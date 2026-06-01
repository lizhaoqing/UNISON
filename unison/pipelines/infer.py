#!/usr/bin/env python
# coding: utf-8
"""
Inference pipeline for UNISON with channel-cat + Qwen2.5-Omni deep fusion.

Channel-Cat approach:
- ref audio is concatenated into the source channel (not as separate ref_latents).
- For zero-shot TTS: source = [ref_latent | zeros], mask = [2 | 0].
  The model sees ref via channel concatenation and learns to generate target audio
  conditioned on the ref portion.
- Optional inpainting trick: at each ODE step, clamp the ref region in latent space
  so the ref portion stays faithful to the original encoding.

Supports three task modes: generation / editing / zeroshotts.
"""

import argparse
import os
import re
from pathlib import Path
from typing import List, Optional
import json

import torch
import torch.nn.functional as F
import torch.distributed as dist
import soundfile as sf
import torchaudio
import torchaudio.functional as AF
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.training_utils import EMAModel
from safetensors.torch import load_file as safe_load_file
from tqdm.auto import tqdm
import yaml
import logging

from unison.models.transformers.backbone import UnisonBackbone
from unison.models.text_encoders.omni_encoder import (
    QwenOmniThinkerExtractor,
    resolve_omni_model_path,
)
from unison.models.text_encoders.qwen3_causal_encoder import (
    Qwen3CausalLMHiddenExtractor,
    resolve_qwen3_model_path,
)
from unison.models.mmaudio.features_utils import FeaturesUtils


# -------------------------
# Defaults / constants
# -------------------------
_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_VAE_CONFIG_PATH = str(_REPO_ROOT / "unison/models/mmaudio/vae_config_44k.yaml")
DEFAULT_OMNI_MODEL_PATH = os.environ.get("QWEN_OMNI_MODEL_PATH", "Qwen/Qwen2.5-Omni-7B")
DEFAULT_MODEL_CONFIG_PATH = str(_REPO_ROOT / "unison/config/D20S0_O_40ch.yaml")

MAX_AUDIO_DURATION = 10.0
AUDIO_FPS          = 31.25
TARGET_FRAMES      = int(MAX_AUDIO_DURATION * AUDIO_FPS)
DEFAULT_TARGET_SAMPLE_RATE = 16000
DEFAULT_VAE_SCALE_FACTOR   = 0.5
REF_DURATION       = 3.0  # seconds
TRAINING_TARGET_RMS = 0.1  # must match GPUAudioProcessorChannelCat.target_rms


# -------------------------
# Logging filter
# -------------------------
class _OmniFilter(logging.Filter):
    def filter(self, record):
        if record.levelno != logging.WARNING:
            return True
        return "System prompt modified" not in record.getMessage()

logging.getLogger().addFilter(_OmniFilter())


# -------------------------
# Model initializers
# -------------------------
def init_text_hidden_extractor(
    text_encoder_type: str,
    omni_model_path: str,
    text_encoder_model_path: Optional[str],
    dit_depth: int,
    device,
    dtype=torch.bfloat16,
    omni_last_layer_idx=-1,
):
    """Load Qwen2.5-Omni or Qwen3 Causal LM; same interface as training script."""
    if text_encoder_type in ("omni", "omni-3b", "omni-7b"):
        omni_path = resolve_omni_model_path(
            text_encoder_type,
            text_encoder_model_path or omni_model_path,
        )
        print(f"  Omni preset={text_encoder_type} path={omni_path}")
        extractor = QwenOmniThinkerExtractor(
            model_path=omni_path,
            dit_depth=dit_depth,
            select_mode="interval",
            device=device,
            dtype=dtype,
            omni_last_layer_idx=omni_last_layer_idx,
        )
    else:
        qwen_path = resolve_qwen3_model_path(text_encoder_type, text_encoder_model_path)
        extractor = Qwen3CausalLMHiddenExtractor(
            model_path=qwen_path,
            dit_depth=dit_depth,
            select_mode="interval",
            device=device,
            dtype=dtype,
            omni_last_layer_idx=omni_last_layer_idx,
        )
    extractor.eval()
    extractor.requires_grad_(False)
    return extractor


def sync_omni_dim_with_text_encoder(model_config, text_extractor):
    """Align model_config['omni_dim'] with loaded text encoder hidden_size."""
    backbone = getattr(text_extractor, "text_backbone", None)
    if backbone is None:
        print("[WARN] Text encoder has no text_backbone; skip omni_dim sync.")
        return
    cfg = getattr(backbone, "config", None)
    hidden = getattr(cfg, "hidden_size", None) if cfg is not None else None
    if hidden is None:
        print("[WARN] Could not read text encoder hidden_size; skip omni_dim sync.")
        return
    prev = model_config.get("omni_dim")
    if prev != hidden:
        print(
            f"[WARN] model_config omni_dim={prev} != text encoder hidden_size={hidden}; "
            f"updating omni_dim to {hidden}."
        )
        model_config["omni_dim"] = hidden


def _resolve_vae_ckpt(path: Optional[str], config_path: str) -> Optional[str]:
    if not path or os.path.isabs(path):
        return path
    return str((Path(config_path).resolve().parent / path).resolve())


def init_audio_vae(config_path):
    """Initialize the MMAudio VAE from a YAML config file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    vae_cfg = config.get("audio_vae", {})
    if vae_cfg.get("tod_vae_ckpt"):
        vae_cfg = {**vae_cfg, "tod_vae_ckpt": _resolve_vae_ckpt(vae_cfg["tod_vae_ckpt"], config_path)}
    if vae_cfg.get("bigvgan_vocoder_ckpt"):
        vae_cfg = {**vae_cfg, "bigvgan_vocoder_ckpt": _resolve_vae_ckpt(vae_cfg["bigvgan_vocoder_ckpt"], config_path)}

    vae = FeaturesUtils(
        tod_vae_ckpt=vae_cfg["tod_vae_ckpt"],
        bigvgan_vocoder_ckpt=vae_cfg.get("bigvgan_vocoder_ckpt"),
        mode=vae_cfg.get("mode", "16k"),
        need_vae_encoder=vae_cfg.get("need_vae_encoder", True),
    )
    scale_factor = vae_cfg.get("vae_scale_factor", DEFAULT_VAE_SCALE_FACTOR)
    sample_rate = vae_cfg.get("sample_rate", 16000)
    return vae, scale_factor, sample_rate


# -------------------------
# Sampling (generation / editing / zero-shot TTS via channel-cat)
# -------------------------
@torch.no_grad()
def sample_latents(
    model,
    scheduler,
    omni_extractor,
    prompts: List[str],
    source_latents=None,
    masks=None,
    num_inference_steps: int = 30,
    guidance_scale: float = 3.5,
    device=torch.device("cuda"),
    target_frames: int = None,
    latent_shape: tuple = None,
    inpaint_mask=None,
    inpaint_source=None,
    noise_init=None,
):
    """
    Sample latents with channel-cat conditioning.

    Audio latent shape: [B, C, T]  (C = VAE latent channels)
      Generation: source_latents=None, masks=None  → zeros are used
      Editing:    source_latents=[B,C,T], masks=[B,1,T]  (mask value 1)
      ZS-TTS:     source_latents=[B,C,T] (ref | zeros), masks=[B,1,T] (2=ref region, 0=target)

    Optional inpainting trick (ZS-TTS only):
      inpaint_mask:   [B, C, T] bool, True at ref positions
      inpaint_source: [B, C, T] clean ref latent (already scaled by vae_scale_factor)
      noise_init:     [B, C, T] initial noise tensor
      At each ODE step the ref region is clamped to keep it faithful to the encoding.
    """
    model.eval()
    batch_size = len(prompts)
    is_edit = source_latents is not None and masks is not None
    C = model.config.in_channels
    is_5d = latent_shape is not None and len(latent_shape) == 3

    # Omni features
    omni_emb_cond, omni_mask_cond, omni_last_emb_cond = omni_extractor(prompts)
    omni_emb_cond  = [e.to(device) for e in omni_emb_cond]
    omni_mask_cond = omni_mask_cond.to(device)
    omni_last_emb_cond = omni_last_emb_cond.to(device)

    if is_edit:
        source_latents = source_latents.to(device=device, dtype=torch.bfloat16)
        masks = masks.to(device=device, dtype=torch.bfloat16)
        if noise_init is not None:
            noise = noise_init.to(device=device, dtype=torch.bfloat16)
        else:
            noise = torch.randn_like(source_latents)
        x = torch.cat([noise, source_latents, masks], dim=1)
    else:
        if is_5d:
            lt, lh, lw = latent_shape
            noise     = torch.randn(batch_size, C, lt, lh, lw, device=device, dtype=torch.bfloat16)
            zero_src  = torch.zeros_like(noise)
            zero_mask = torch.zeros(batch_size, 1, lt, lh, lw, device=device, dtype=torch.bfloat16)
        else:
            T = target_frames if target_frames is not None else TARGET_FRAMES
            noise     = torch.randn(batch_size, C, T, device=device, dtype=torch.bfloat16)
            zero_src  = torch.zeros_like(noise)
            zero_mask = torch.zeros(batch_size, 1, T, device=device, dtype=torch.bfloat16)
        x = torch.cat([noise, zero_src, zero_mask], dim=1)

    # Move inpainting tensors to device
    if inpaint_mask is not None:
        inpaint_mask = inpaint_mask.to(device=device)
    if inpaint_source is not None:
        inpaint_source = inpaint_source.to(device=device, dtype=torch.bfloat16)
    if noise_init is not None:
        noise_init = noise_init.to(device=device, dtype=torch.bfloat16)

    scheduler.set_timesteps(num_inference_steps, device=device)
    do_cfg = guidance_scale is not None and guidance_scale > 1.0

    noise_slice = slice(None, C)

    if do_cfg:
        omni_emb_uncond  = [torch.zeros_like(e) for e in omni_emb_cond]
        omni_mask_uncond = torch.zeros_like(omni_mask_cond)
        omni_last_emb_uncond = torch.zeros_like(omni_last_emb_cond)

        timesteps = scheduler.timesteps
        for step_idx, t in enumerate(tqdm(timesteps, desc="Sampling (CFG)")):
            t_in = (t.float() / scheduler.config.num_train_timesteps).repeat(batch_size).to(device=device, dtype=torch.bfloat16)

            out_unc = model(x=x, t=t_in, duration=None,
                            omni_emb_list=omni_emb_uncond, omni_last_emb=omni_last_emb_uncond,
                            omni_mask=omni_mask_uncond)
            pred_unc = out_unc[0]

            out_cond = model(x=x, t=t_in, duration=None,
                             omni_emb_list=omni_emb_cond, omni_last_emb=omni_last_emb_cond,
                             omni_mask=omni_mask_cond)
            pred_cond = out_cond[0]

            pred = pred_unc + guidance_scale * (pred_cond - pred_unc)
            x[:, noise_slice] = scheduler.step(model_output=pred, timestep=t, sample=x[:, noise_slice]).prev_sample

            # Inpainting trick: clamp ref region after each ODE step
            if inpaint_mask is not None and noise_init is not None and inpaint_source is not None:
                # Compute sigma for the NEXT step (after the update we just did)
                if step_idx + 1 < len(timesteps):
                    t_next = timesteps[step_idx + 1]
                    sigma_next = t_next.float() / scheduler.config.num_train_timesteps
                else:
                    sigma_next = 0.0
                # clamp_val = (1 - sigma_next) * clean + sigma_next * noise
                clamp_val = (1.0 - sigma_next) * inpaint_source + sigma_next * noise_init
                for dim_idx in range(C):
                    x[:, dim_idx][inpaint_mask[:, dim_idx]] = clamp_val[:, dim_idx][inpaint_mask[:, dim_idx]]
    else:
        timesteps = scheduler.timesteps
        for step_idx, t in enumerate(tqdm(timesteps, desc="Sampling")):
            t_in = (t.float() / scheduler.config.num_train_timesteps).repeat(batch_size).to(device=device, dtype=torch.bfloat16)
            out = model(x=x, t=t_in, duration=None,
                        omni_emb_list=omni_emb_cond, omni_last_emb=omni_last_emb_cond,
                        omni_mask=omni_mask_cond)
            pred = out[0]

            x[:, noise_slice] = scheduler.step(model_output=pred, timestep=t, sample=x[:, noise_slice]).prev_sample

            # Inpainting trick: clamp ref region after each ODE step
            if inpaint_mask is not None and noise_init is not None and inpaint_source is not None:
                if step_idx + 1 < len(timesteps):
                    t_next = timesteps[step_idx + 1]
                    sigma_next = t_next.float() / scheduler.config.num_train_timesteps
                else:
                    sigma_next = 0.0
                clamp_val = (1.0 - sigma_next) * inpaint_source + sigma_next * noise_init
                for dim_idx in range(C):
                    x[:, dim_idx][inpaint_mask[:, dim_idx]] = clamp_val[:, dim_idx][inpaint_mask[:, dim_idx]]

    return x[:, noise_slice]


# -------------------------
# Decode & save
# -------------------------
@torch.no_grad()
def decode_and_save(audio_vae, latents, durations, output_paths, sample_rate=None):
    if sample_rate is None:
        sample_rate = DEFAULT_TARGET_SAMPLE_RATE
    device  = next(audio_vae.parameters()).device
    dtype = getattr(audio_vae, 'dtype', torch.bfloat16)
    latents = latents.to(device=device, dtype=dtype)
    audio   = audio_vae.wrapped_decode(latents).cpu()

    for i, path in enumerate(output_paths):
        dur     = min(float(durations[i]), MAX_AUDIO_DURATION)
        max_smp = int(dur * sample_rate)
        wav     = audio[i]
        if wav.dim() > 1:
            wav = wav.mean(0)
        wav = wav[..., :max_smp]
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        sf.write(path, wav.float().numpy(), sample_rate)
        print(f"  Saved: {path}  ({dur:.1f}s)")


@torch.no_grad()
def decode_and_save_full(audio_vae, latents_full, ref_samples,
                         durations, output_paths, sample_rate=None,
                         fade_in_ms: float = 20.0,
                         max_duration: float = None,
                         ref_wav_raw: torch.Tensor = None,
                         ref_output_paths=None):
    """Decode the FULL latent (ref+target), then crop target in waveform space.

    This avoids boundary artifacts that occur when cropping in latent space
    (the decoder's convolutional receptive field needs left-side context).

    The crop point is snapped to the nearest VAE hop boundary so the cut
    aligns with a latent frame edge, then a short fade-in suppresses any
    residual ref tail bleed from the decoder's receptive field.

    If ref_wav_raw is provided (the truncated ref waveform before VAE encode),
    it is saved to ref_output_paths alongside the generated target files. This
    lets you verify exactly what the model heard as the reference.
    """
    if sample_rate is None:
        sample_rate = DEFAULT_TARGET_SAMPLE_RATE
    device = next(audio_vae.parameters()).device
    dtype = getattr(audio_vae, 'dtype', torch.bfloat16)
    latents_full = latents_full.to(device=device, dtype=dtype)
    audio_full = audio_vae.wrapped_decode(latents_full).cpu()

    wav_total = audio_full.shape[-1]
    lat_total = latents_full.shape[-1]
    vae_hop = wav_total // lat_total if lat_total > 0 else 1
    # Snap crop point to nearest VAE hop boundary
    crop_start = round(ref_samples / vae_hop) * vae_hop
    crop_start_s = crop_start / sample_rate

    save_cap = MAX_AUDIO_DURATION if max_duration is None else float(max_duration)

    for i, path in enumerate(output_paths):
        dur = min(float(durations[i]), save_cap)
        max_smp = int(dur * sample_rate)
        wav = audio_full[i]
        if wav.dim() > 1:
            wav = wav.mean(0)
        target_wav = wav[crop_start: crop_start + max_smp].clone()
        fade_len = min(int(fade_in_ms * sample_rate / 1000), target_wav.shape[-1])
        if fade_len > 1:
            target_wav[:fade_len] *= torch.linspace(0.0, 1.0, fade_len)
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        sf.write(path, target_wav.float().numpy(), sample_rate)
        print(f"  Saved (target): {path}  ({dur:.1f}s, crop@{crop_start} = {crop_start_s:.3f}s, fade={fade_in_ms}ms)")

        # Optionally save the truncated ref waveform used as conditioning
        if ref_wav_raw is not None and ref_output_paths is not None and i < len(ref_output_paths):
            ref_path = ref_output_paths[i]
            ref_out = ref_wav_raw.squeeze().cpu().float()
            if ref_out.dim() == 0:
                ref_out = ref_out.unsqueeze(0)
            os.makedirs(os.path.dirname(ref_path) or ".", exist_ok=True)
            sf.write(ref_path, ref_out.numpy(), sample_rate)
            print(f"  Saved (ref):    {ref_path}  ({ref_out.shape[-1] / sample_rate:.3f}s)")


# -------------------------
# Audio utilities
# -------------------------
def _rms_normalize(wav: torch.Tensor, target_rms: float = TRAINING_TARGET_RMS) -> torch.Tensor:
    """Normalize waveform RMS to match training preprocessing."""
    rms = torch.sqrt(torch.mean(wav ** 2))
    if rms > 0:
        wav = wav * (target_rms / (rms + 1e-8))
    return wav


def load_source_audio(audio_path, target_sr=DEFAULT_TARGET_SAMPLE_RATE,
                      target_length=None, max_duration=MAX_AUDIO_DURATION, device=None):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wav, sr = torchaudio.load(audio_path)
    if wav.shape[0] > 1:
        wav = wav.mean(0, keepdim=True)
    if sr != target_sr:
        wav = AF.resample(wav, sr, target_sr)
    wav = _rms_normalize(wav)
    L = target_length or int(max_duration * target_sr)
    if wav.shape[-1] > L:
        wav = wav[..., :L]
    elif wav.shape[-1] < L:
        wav = F.pad(wav, (0, L - wav.shape[-1]))
    return wav.to(device)


_whisper_model = None

def get_whisper_model(model_size="base"):
    global _whisper_model
    if _whisper_model is None:
        try:
            import whisper
        except ImportError:
            raise ImportError(
                "openai-whisper is required for auto-transcription. "
                "Install with: pip install openai-whisper"
            )
        print(f"  Loading Whisper ({model_size}) for auto-transcription...")
        _whisper_model = whisper.load_model(model_size)
    return _whisper_model


def transcribe_ref_audio(wav_tensor, sr, whisper_size="base"):
    """Auto-transcribe a (possibly truncated) waveform tensor using Whisper.
    
    Args:
        wav_tensor: [1, T] or [T] waveform on any device / sample rate
        sr: sample rate of wav_tensor
    """
    model = get_whisper_model(whisper_size)
    wav_16k = wav_tensor.squeeze().cpu().float()
    if sr != 16000:
        wav_16k = AF.resample(wav_16k.unsqueeze(0), sr, 16000).squeeze(0)
    audio_np = wav_16k.numpy()
    result = model.transcribe(audio_np, fp16=torch.cuda.is_available())
    text = result["text"].strip()
    print(f"    [ASR] ref_text = \"{text}\"")
    return text


# --- Silence-aware ref cut + tail padding helpers --------------------------
#
# Why these exist: training does a hard cut at a random sample for ref/target
# split, so the model has learned to *continue the on-going acoustic state* at
# the cut point. At inference, if the user feeds a hard-cut ref that ends mid
# word / mid syllable, the model faithfully extends that ref tail into the
# target generation (ref-tail bleed into output). Two cheap fixes:
#   1. Snap the cut backward to the nearest silent frame (energy-based VAD).
#   2. Append a short silence pad after the ref so the model sees a calm
#      starting state for target generation.

def _find_silence_cut(
    wav: torch.Tensor,
    sr: int,
    target_len: int,
    search_back_s: float = 0.5,
    win_ms: float = 20.0,
    threshold_db: float = -40.0,
) -> int:
    """Snap target_len backward to the last silent frame within search_back_s.

    Returns the cut sample index in [target_len - search_back_s*sr, target_len].
    Falls back to target_len if no silent frame is found in the search window.
    """
    if wav.dim() == 2:
        mono = wav.mean(dim=0)
    else:
        mono = wav

    total = mono.shape[-1]
    if target_len <= 0 or total == 0:
        return max(0, target_len)
    if total <= target_len:
        return total

    win = max(int(win_ms * sr / 1000), 1)
    hop = max(win // 2, 1)
    search_back = max(int(search_back_s * sr), win)
    search_start = max(0, target_len - search_back)
    search_end = target_len
    if search_end - search_start < win:
        return target_len

    seg = mono[search_start:search_end].float()
    frames = seg.unfold(-1, win, hop)                       # [n_frames, win]
    rms = frames.pow(2).mean(-1).sqrt()
    rms_db = 20.0 * torch.log10(rms.clamp(min=1e-10))
    is_silence = rms_db < threshold_db
    if not bool(is_silence.any()):
        return target_len

    silence_idx = torch.where(is_silence)[0]
    last = int(silence_idx[-1].item())
    cut = search_start + last * hop + win // 2
    return max(1, min(cut, target_len))


# --- Trailing-punctuation strip + ref/target join helpers ------------------
#
# Whisper transcription always emits punctuation; for a ref hard-cut mid
# sentence that produces things like "Hello there.", which (a) tells the
# model the speaker has finished a thought and (b) double-punctuates when we
# concatenate with target text. Strip the trailing punct and re-insert a
# soft "continue" comma between ref and target.

_TRAILING_PUNCT_CHARS = (
    " \t\r\n"
    ".,;:!?\"'`-—…"
    "。，；：！？、…—"
    "\u3002\uff0c\uff1b\uff1a\uff01\uff1f\u3001"
    "“”‘’《》「」『』"
)
_TRAILING_PUNCT_RE = re.compile(
    "[" + re.escape(_TRAILING_PUNCT_CHARS) + "]+$"
)
_CJK_RE = re.compile(r"[\u3400-\u9fff\uff00-\uffef]")


def _strip_trailing_punct(text: str) -> str:
    if not text:
        return ""
    return _TRAILING_PUNCT_RE.sub("", text).rstrip()


def _has_cjk(text: str) -> bool:
    return bool(_CJK_RE.search(text or ""))


def join_ref_target_text(ref_text: str, target_text: str) -> str:
    """Strip ref's trailing punctuation, then insert a continuation comma
    between ref_text and target_text (Chinese 「，」 if either side is CJK,
    else English ", ")."""
    ref_text = _strip_trailing_punct(ref_text or "")
    target_text = (target_text or "").strip()
    if not ref_text:
        return target_text
    if not target_text:
        return ref_text
    sep = "，" if (_has_cjk(ref_text) or _has_cjk(target_text)) else ", "
    return f"{ref_text}{sep}{target_text}"


def load_ref_audio(
    audio_path,
    target_sr=DEFAULT_TARGET_SAMPLE_RATE,
    max_ref_duration=None,
    device=None,
    silence_search_s: float = 0.5,
    silence_threshold_db: float = -40.0,
    tail_pad_s: float = 0.1,
):
    """Load reference audio, optionally truncate to max_ref_duration.

    If silence_search_s > 0 and max_ref_duration is set, the cut sample is
    snapped backward to the last silent frame within
    [target_len - silence_search_s*sr, target_len]. After RMS-normalizing
    the cut waveform, ``tail_pad_s`` of true silence is appended so the
    model starts target generation from a calm state instead of continuing
    the ref's last formant.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wav, sr = torchaudio.load(audio_path)
    if wav.shape[0] > 1:
        wav = wav.mean(0, keepdim=True)
    if sr != target_sr:
        wav = AF.resample(wav, sr, target_sr)

    if max_ref_duration is not None:
        ref_len = int(max_ref_duration * target_sr)
        if wav.shape[-1] > ref_len:
            if silence_search_s > 0:
                cut = _find_silence_cut(
                    wav, target_sr, ref_len,
                    search_back_s=silence_search_s,
                    threshold_db=silence_threshold_db,
                )
                cut_dur = cut / target_sr
                if cut < ref_len:
                    print(
                        f"    [ref-cut] snapped {max_ref_duration:.2f}s -> "
                        f"{cut_dur:.2f}s at silence (search_back={silence_search_s:.2f}s, "
                        f"thresh={silence_threshold_db:.0f}dB)"
                    )
                wav = wav[..., :cut]
            else:
                wav = wav[..., :ref_len]

    wav = _rms_normalize(wav)

    if tail_pad_s > 0:
        pad_len = int(tail_pad_s * target_sr)
        if pad_len > 0:
            wav = F.pad(wav, (0, pad_len))

    return wav.to(device)


def make_edit_mask(waveform_length, device):
    return torch.ones(1, 1, waveform_length, device=device, dtype=torch.float32)


def downsample_mask(mask_wav, latent_length):
    T = mask_wav.shape[-1]
    if T != latent_length:
        m = F.adaptive_avg_pool1d(mask_wav.float(), latent_length)
        return (m > 0.5).float()
    return mask_wav.float()


# -------------------------
# CLI
# -------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="UNISON: inference pipeline (generation / editing / zero-shot TTS)"
    )
    # Model
    p.add_argument("--model_ckpt",      type=str, required=True)
    p.add_argument("--vae_config",      type=str, default=DEFAULT_VAE_CONFIG_PATH)
    p.add_argument("--omni_model_path", type=str, default=DEFAULT_OMNI_MODEL_PATH,
                   help="Qwen2.5-Omni checkpoint. Used when --text_encoder_type=omni, "
                        "or as local override for omni-3b/omni-7b presets.")
    p.add_argument(
        "--text_encoder_type",
        type=str,
        default="omni",
        choices=[
            "omni", "omni-3b", "omni-7b",
            "qwen3-4b-instruct", "qwen3-1.7b", "qwen3-0.6b",
        ],
        help="Must match training: "
             "'omni' = Qwen2.5-Omni with system prompt (path from --omni_model_path); "
             "'omni-3b' / 'omni-7b' = Qwen2.5-Omni presets (system prompt kept); "
             "'qwen3-*' = Qwen3 Causal LM presets (no system prompt).",
    )
    p.add_argument(
        "--text_encoder_model_path",
        type=str,
        default=None,
        help="Optional local path override for qwen3-* / omni-* presets.",
    )
    p.add_argument("--dit_depth",       type=int, default=None,
                   help="(deprecated) Auto-computed from model_config. Only set to override.")
    p.add_argument("--model_config",    type=str, default=DEFAULT_MODEL_CONFIG_PATH)

    # Task mode
    p.add_argument("--task_mode", type=str,
                   choices=["generation", "editing", "zeroshotts", "all"],
                   default="all")

    # Generation
    p.add_argument("--gen_prompt",      type=str, default=None,
                   help="Single generation prompt string.")
    p.add_argument("--gen_prompt_list", type=str, default=None,
                   help="Path to a text file with one prompt per line.")
    p.add_argument("--gen_duration",    type=float, default=10.0)

    # Editing
    p.add_argument("--edit_prompt",       type=str, default=None,
                   help="Single editing prompt (requires --edit_source_audio).")
    p.add_argument("--edit_source_audio", type=str, default=None,
                   help="Source audio path for single editing task.")
    p.add_argument("--edit_config",       type=str, default=None,
                   help="Path to JSON list of {prompt, source_audio} objects.")

    # Zero-shot TTS
    p.add_argument("--zs_prompt",      type=str, default=None,
                   help="Target text for single zero-shot TTS (what to say).")
    p.add_argument("--zs_ref_audio",   type=str, default=None,
                   help="Reference audio for single zero-shot TTS.")
    p.add_argument("--zs_ref_text",    type=str, default=None,
                   help="Transcription of ref audio. If omitted, auto-transcribed via Whisper.")
    p.add_argument("--zs_config",      type=str, default=None,
                   help="Path to JSON list of {target_text, ref_audio, ref_text(optional)} objects.")
    p.add_argument("--zs_duration",    type=float, default=10.0,
                   help="Max total duration (ref+target) in seconds.")
    p.add_argument("--ref_duration",   type=float, default=REF_DURATION,
                   help="Max reference audio duration in seconds (default: 3.0)")
    p.add_argument("--whisper_model",  type=str, default="base",
                   help="Whisper model size for auto-transcription (tiny/base/small/medium/large)")
    # Ref tail cleanup (anti-tail-bleed). Set --ref_silence_search_s 0 and
    # --ref_tail_pad_s 0 to fully disable.
    p.add_argument("--ref_silence_search_s", type=float, default=0.5,
                   help="Search backward this many seconds from --ref_duration to "
                        "snap the cut at a silent frame. 0 disables silence-aware cut.")
    p.add_argument("--ref_silence_threshold_db", type=float, default=-40.0,
                   help="Frames with RMS below this dB are considered silence.")
    p.add_argument("--ref_tail_pad_s", type=float, default=0.1,
                   help="Seconds of true silence appended after ref to reset the "
                        "model's starting acoustic state. 0 disables.")
    p.add_argument("--ref_text_join", type=str, default="auto",
                   choices=["auto", "comma", "space"],
                   help="How to join ref_text and target_text after stripping ref's "
                        "trailing punctuation. 'auto' uses 「，」 for CJK and ', ' "
                        "otherwise; 'comma' forces a comma; 'space' forces a single "
                        "space (legacy behavior).")

    # Inpainting trick
    p.add_argument("--use_inpainting", action="store_true", default=False,
                   help="Use inpainting trick for zero-shot TTS (clamp ref xt each step)")
    p.add_argument("--mask_zs_after_encode", action="store_true", default=False,
                   help="Zero out target region of source latent after VAE encoding. "
                        "Must match training flag.")

    # Inference
    p.add_argument("--num_inference_steps", type=int,   default=50)
    p.add_argument("--guidance_scale",      type=float, default=4.5)
    p.add_argument("--output_dir",          type=str,   default="./outputs/timed_channelcat_infer")
    p.add_argument("--seed",                type=int,   default=42)
    p.add_argument("--device",              type=str,   default="cuda")

    return p.parse_args()


def _load_model(args, device, model_config):
    mcfg = dict(model_config)
    mcfg.pop("omni_last_layer_idx", None)
    mcfg["concat_condition"] = True

    model = UnisonBackbone(**mcfg)

    ckpt_path = args.model_ckpt
    if os.path.isdir(ckpt_path):
        for name in ("ema_model.pt", "model.safetensors", "pytorch_model.bin"):
            candidate = os.path.join(ckpt_path, name)
            if os.path.exists(candidate):
                ckpt_path = candidate
                break

    print(f"Loading checkpoint: {ckpt_path}")
    if ckpt_path.endswith(".safetensors"):
        sd = safe_load_file(ckpt_path, device="cpu")
    else:
        sd = torch.load(ckpt_path, map_location="cpu")

    is_ema = "shadow_params" in sd or any(k.startswith("shadow_params") for k in sd)
    if is_ema:
        print(">>> EMA wrapper detected - unwrapping...")
        ema = EMAModel(model.parameters(), decay=0.9999,
                       model_cls=UnisonBackbone, model_config=mcfg)
        ema.load_state_dict(sd)
        ema.copy_to(model.parameters())
        print(">>> EMA weights injected.")
    else:
        ignore = {'decay','num_updates','min_decay','update_after_step',
                  'use_ema_warmup','inv_gamma','power','shadow_params'}
        model.load_state_dict({k: v for k, v in sd.items() if k not in ignore}, strict=False)

    model.to(device=device, dtype=torch.bfloat16).eval()
    return model


def main():
    args   = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    print("=" * 60)
    print("UNISON Inference")
    print("=" * 60)

    # ---------- Compute dit_depth ----------
    with open(args.model_config) as _f:
        _mcfg = yaml.safe_load(_f)
    dit_depth = args.dit_depth
    if dit_depth is None:
        dit_depth = _mcfg.get("mm_double_blocks_depth", 0) + _mcfg.get("mm_single_blocks_depth", 0)
    print(f"  dit_depth = {dit_depth} (double={_mcfg.get('mm_double_blocks_depth')}, single={_mcfg.get('mm_single_blocks_depth')})")

    # ---------- Initialize models ----------
    omni_last_layer_idx = _mcfg.get("omni_last_layer_idx", -1)
    print(
        f"  text_encoder_type={args.text_encoder_type} "
        f"(omni_path={args.omni_model_path if args.text_encoder_type.startswith('omni') else 'N/A'}, "
        f"text_encoder_model_path={args.text_encoder_model_path})"
    )
    omni_extractor = init_text_hidden_extractor(
        args.text_encoder_type,
        args.omni_model_path,
        args.text_encoder_model_path,
        dit_depth,
        device,
        dtype=torch.bfloat16,
        omni_last_layer_idx=omni_last_layer_idx,
    )
    sync_omni_dim_with_text_encoder(_mcfg, omni_extractor)
    model = _load_model(args, device, _mcfg)
    audio_vae, vae_scale_factor, vae_sample_rate = init_audio_vae(args.vae_config)
    audio_vae.to(device).eval()
    scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000)
    os.makedirs(args.output_dir, exist_ok=True)

    # Probe actual latent length via a dummy encode.
    # We do NOT use a fixed vae_hop constant because the effective hop for the
    # MMAudio 44k VAE (mel hop 512 + 1D encoder 2× downsample = hop 1024) differs
    # from what one might hard-code, causing 2× too-long and near-silent outputs.
    gen_target_frames = None
    dummy_len = int(MAX_AUDIO_DURATION * vae_sample_rate)
    with torch.no_grad():
        dummy_wav = torch.zeros(1, dummy_len, device=device)
        dummy_lat = audio_vae.wrapped_encode(dummy_wav)
        gen_target_frames = int(dummy_lat.shape[-1])
        del dummy_wav, dummy_lat
        torch.cuda.empty_cache()
    vae_hop = dummy_len // max(gen_target_frames, 1)
    audio_fps = vae_sample_rate / max(vae_hop, 1)
    if hasattr(audio_vae, 'hop_length'):
        reported_hop = int(audio_vae.hop_length)
        hop_note = f", reported_hop={reported_hop}"
        if reported_hop != vae_hop:
            hop_note += f" (MISMATCH with probed {vae_hop})"
    else:
        hop_note = " (no hop_length attr; probed from dummy encode)"
    print(
        f"  VAE: probed_hop={vae_hop}, latent FPS={audio_fps:.2f}, "
        f"gen frames ({MAX_AUDIO_DURATION}s)={gen_target_frames}{hop_note}"
    )

    print(f"  VAE scale factor: {vae_scale_factor}, sample rate: {vae_sample_rate}")
    print(f"  Inpainting trick: {'ON' if args.use_inpainting else 'OFF'}")
    print(f"  mask_zs_after_encode: {'ON' if args.mask_zs_after_encode else 'OFF'}")

    # Checkpoint name prefix
    if os.path.isdir(args.model_ckpt):
        ckpt_tag = os.path.basename(args.model_ckpt) + "_"
    else:
        ckpt_tag = os.path.splitext(os.path.basename(args.model_ckpt))[0] + "_"
    ckpt_tag = ""
    params_tag = f"s{args.num_inference_steps}_cfg{args.guidance_scale}_seed{args.seed}"

    # ---------- Generation tasks ----------
    if args.task_mode in ("generation", "all"):
        print("\n[Generation Tasks]")
        if args.gen_prompt_list:
            with open(args.gen_prompt_list, encoding="utf-8") as f:
                gen_prompts = [l.strip() for l in f if l.strip() and not l.startswith("#")]
        elif args.gen_prompt:
            gen_prompts = [args.gen_prompt]
        else:
            gen_prompts = []

        for i, prompt in enumerate(gen_prompts):
            print(f"\n  [{i+1}/{len(gen_prompts)}] {prompt[:70]}{'...' if len(prompt) > 70 else ''}")
            latents = sample_latents(model, scheduler, omni_extractor, [prompt],
                                     num_inference_steps=args.num_inference_steps,
                                     guidance_scale=args.guidance_scale, device=device,
                                     target_frames=gen_target_frames,
                                     latent_shape=None)
            latents = latents * (1.0 / vae_scale_factor)
            safe = "".join(c if c.isalnum() or c in " -_" else "" for c in prompt[:70]).strip().replace(" ", "_")
            out_path = os.path.join(args.output_dir, f"{ckpt_tag}gen_{safe}_{params_tag}_{i:03d}.wav")
            decode_and_save(audio_vae, latents, [args.gen_duration], [out_path], sample_rate=vae_sample_rate)

    # ---------- Editing tasks ----------
    if args.task_mode in ("editing", "all"):
        print("\n[Editing Tasks]")
        if args.edit_config:
            with open(args.edit_config, encoding="utf-8") as f:
                edit_tasks = json.load(f)
        elif args.edit_prompt and args.edit_source_audio:
            edit_tasks = [{"prompt": args.edit_prompt, "source_audio": args.edit_source_audio}]
        else:
            edit_tasks = []

        tgt_wav_len = int(MAX_AUDIO_DURATION * vae_sample_rate)
        for i, task in enumerate(edit_tasks):
            prompt   = task["prompt"]
            src_path = task["source_audio"]
            print(f"\n  [{i+1}/{len(edit_tasks)}] {prompt[:70]}{'...' if len(prompt) > 70 else ''}")
            print(f"    src: {src_path}")

            src_wav    = load_source_audio(src_path, target_sr=vae_sample_rate,
                                           target_length=tgt_wav_len, device=device)
            src_latent = audio_vae.wrapped_encode(src_wav) * vae_scale_factor

            if src_latent.dim() == 5:
                mask_lat = torch.ones(1, 1, *src_latent.shape[2:], device=device, dtype=torch.float32)
            else:
                mask_wav = make_edit_mask(src_wav.shape[-1], device)
                lat_len  = src_latent.shape[-1]
                mask_lat = downsample_mask(mask_wav, lat_len)

            latents = sample_latents(model, scheduler, omni_extractor, [prompt],
                                     source_latents=src_latent, masks=mask_lat,
                                     num_inference_steps=args.num_inference_steps,
                                     guidance_scale=args.guidance_scale, device=device)
            latents = latents * (1.0 / vae_scale_factor)
            dur     = src_wav.shape[-1] / vae_sample_rate
            safe    = "".join(c if c.isalnum() or c in " -_" else "" for c in prompt[:70]).strip().replace(" ", "_")
            src_tag = Path(src_path).stem
            out_path = os.path.join(args.output_dir, f"{ckpt_tag}edit_{safe}_{src_tag}_{params_tag}_{i:03d}.wav")
            decode_and_save(audio_vae, latents, [dur], [out_path], sample_rate=vae_sample_rate)

    # ---------- Zero-shot TTS tasks (Channel-Cat) ----------
    if args.task_mode in ("zeroshotts", "all"):
        print("\n[Zero-shot TTS Tasks (Channel-Cat)]")
        if args.zs_config:
            with open(args.zs_config, encoding="utf-8") as f:
                zs_tasks = json.load(f)
        elif args.zs_prompt and args.zs_ref_audio:
            zs_tasks = [{"target_text": args.zs_prompt, "ref_audio": args.zs_ref_audio,
                         "ref_text": args.zs_ref_text}]
        else:
            zs_tasks = []

        max_total_duration = args.zs_duration

        for i, task in enumerate(zs_tasks):
            target_text = task.get("target_text") or task.get("prompt", "")
            ref_path    = task["ref_audio"]
            ref_text    = task.get("ref_text")
            duration    = task.get("duration", max_total_duration)

            print(f"\n  [{i+1}/{len(zs_tasks)}] target: {target_text[:70]}{'...' if len(target_text) > 70 else ''}")
            print(f"    ref: {ref_path}")

            # Load ref audio first (truncate to ref_duration with silence-aware
            # snap + tail silence pad to suppress ref-tail leakage).
            ref_wav = load_ref_audio(
                ref_path, target_sr=vae_sample_rate,
                max_ref_duration=args.ref_duration, device=device,
                silence_search_s=args.ref_silence_search_s,
                silence_threshold_db=args.ref_silence_threshold_db,
                tail_pad_s=args.ref_tail_pad_s,
            )
            ref_audio_duration = ref_wav.shape[-1] / vae_sample_rate

            # Auto-transcribe the *truncated* ref audio if ref_text not provided.
            # Pass the speech-only portion (exclude tail silence pad) to Whisper
            # so it doesn't hallucinate trailing tokens on the silence.
            speech_samples = ref_wav.shape[-1] - int(args.ref_tail_pad_s * vae_sample_rate)
            speech_samples = max(speech_samples, 1)
            if not ref_text:
                ref_text = transcribe_ref_audio(
                    ref_wav[..., :speech_samples],
                    sr=vae_sample_rate,
                    whisper_size=args.whisper_model,
                )
            else:
                print(f"    ref_text = \"{ref_text}\"")

            # Ensure total duration fits within max
            if ref_audio_duration + 1.0 > duration:
                print(f"    [WARN] ref ({ref_audio_duration:.1f}s) + min_target (1.0s) > duration ({duration:.1f}s), skipping.")
                continue
            print(f"    ref_duration={ref_audio_duration:.2f}s, total_duration={duration:.1f}s, target_duration={duration - ref_audio_duration:.2f}s")

            # Build combined prompt: strip ref's trailing punctuation, then
            # join with a continuation comma so the model reads ref as
            # "paused mid-thought" rather than "fully completed sentence".
            if args.ref_text_join == "space":
                combined_text = f"{_strip_trailing_punct(ref_text)} {target_text.strip()}".strip()
            elif args.ref_text_join == "comma":
                ref_clean = _strip_trailing_punct(ref_text)
                tgt_clean = target_text.strip()
                if ref_clean and tgt_clean:
                    sep = "，" if (_has_cjk(ref_clean) or _has_cjk(tgt_clean)) else ", "
                    combined_text = f"{ref_clean}{sep}{tgt_clean}"
                else:
                    combined_text = ref_clean or tgt_clean
            else:  # auto
                combined_text = join_ref_target_text(ref_text, target_text)
            prompt = f"[Speech with voice] {combined_text}"
            print(f"    prompt = {prompt[:100]}{'...' if len(prompt) > 100 else ''}")

            # Build source waveform [ref | zeros] and encode as one piece
            total_wav_len = int(duration * vae_sample_rate)
            ref_wav_1d = ref_wav.squeeze(0) if ref_wav.dim() == 2 else ref_wav
            ref_wav_len = min(ref_wav_1d.shape[-1], total_wav_len)

            source_wav = torch.zeros(1, total_wav_len, device=device)
            source_wav[:, :ref_wav_len] = ref_wav_1d[:ref_wav_len]
            with torch.no_grad():
                source_latent = audio_vae.wrapped_encode(source_wav) * vae_scale_factor
            C = source_latent.shape[1]
            total_lat_len = source_latent.shape[-1]

            # Compute ref latent length via mask interpolation (same as training)
            mask_wav = torch.zeros(1, 1, total_wav_len, device=device)
            mask_wav[:, :, :ref_wav_len] = 2.0
            mask_latent = F.interpolate(mask_wav, size=total_lat_len, mode='nearest').to(torch.bfloat16)

            # mask_zs_after_encode: zero target region in source latent
            if args.mask_zs_after_encode:
                ref_region = (mask_latent > 1.5).float()
                source_latent = source_latent * ref_region.expand_as(source_latent)

            # Prepare inpainting data
            inpaint_mask_arg = None
            inpaint_source_arg = None
            noise_init_arg = None

            if args.use_inpainting:
                noise_init_arg = torch.randn(1, C, total_lat_len, device=device, dtype=torch.bfloat16)
                inpaint_source_arg = source_latent[:, :C, :].clone()
                inpaint_mask_1d = (mask_latent[:, 0, :] > 1.5)
                inpaint_mask_arg = inpaint_mask_1d.unsqueeze(1).expand(-1, C, -1)

            latents = sample_latents(model, scheduler, omni_extractor, [prompt],
                                     source_latents=source_latent, masks=mask_latent,
                                     num_inference_steps=args.num_inference_steps,
                                     guidance_scale=args.guidance_scale, device=device,
                                     inpaint_mask=inpaint_mask_arg,
                                     inpaint_source=inpaint_source_arg,
                                     noise_init=noise_init_arg)

            # Decode the FULL latent (ref+target) so the decoder has
            # boundary context, then crop the target waveform.
            # Cropping in latent space would cut the decoder's left-side
            # receptive field at the boundary → artifacts / "tail sound".
            latents_full = latents * (1.0 / vae_scale_factor)
            ref_samples = int(ref_audio_duration * vae_sample_rate)

            tgt_duration = duration - ref_audio_duration
            safe = "".join(c if c.isalnum() or c in " -_" else "" for c in target_text[:70]).strip().replace(" ", "_")
            ref_tag = Path(ref_path).stem
            out_path = os.path.join(args.output_dir, f"{ckpt_tag}zstts_{ref_tag}_{safe}_{params_tag}_{i:03d}.wav")
            ref_out_path = os.path.join(args.output_dir, f"{ckpt_tag}ref_{ref_tag}_{params_tag}_{i:03d}.wav")
            decode_and_save_full(audio_vae, latents_full, ref_samples,
                                [tgt_duration], [out_path], sample_rate=vae_sample_rate,
                                ref_wav_raw=ref_wav, ref_output_paths=[ref_out_path])

    print(f"\nDone. Outputs -> {args.output_dir}")
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
