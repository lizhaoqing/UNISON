#!/usr/bin/env python
# coding: utf-8
"""
UNISON fine-tuning example script.

This script provides a minimal but complete fine-tuning example using a small
set of audio samples supplied in data/train/metadata.jsonl.  It is intended as a
starting point for fine-tuning UNISON on your own data.

Usage
-----
Single-GPU smoke test:

    python unison/pipelines/train.py \
        --model_config  unison/config/D20S0_O_40ch.yaml \
        --vae_config    unison/models/mmaudio/vae_config_44k.yaml \
        --omni_model_path $QWEN_OMNI_MODEL_PATH \
        --metadata      data/train/metadata.jsonl \
        --output_dir    outputs/finetune_example \
        --max_train_steps 200

Multi-GPU (accelerate):

    accelerate launch \\
        --config_file unison/config/training/accelerator_config.yaml \\
        --num_processes 4 \\
        unison/pipelines/train.py [args]

Data format
-----------
The metadata JSONL file (--metadata) must contain one JSON object per line with
at least the following keys:

    audio_path  : str    Path to .wav file (relative to repo root or absolute).
    caption     : str    Text description for this audio clip.
    duration    : float  Clip duration in seconds (used for padding/trimming).

See data/train/metadata.jsonl for an example.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn.functional as F
import torchaudio
import torchaudio.functional as AF
import yaml
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs, ProjectConfiguration, set_seed
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.training_utils import EMAModel, compute_density_for_timestep_sampling
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import get_scheduler

from unison.models.transformers.backbone import UnisonBackbone
from unison.models.text_encoders.omni_encoder import (
    QwenOmniThinkerExtractor,
    resolve_omni_model_path,
)
from unison.models.mmaudio.features_utils import FeaturesUtils
from unison.utils.diffusers_compat import patch_transformers_deepspeed

patch_transformers_deepspeed()

logger = get_logger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_VAE_SCALE_FACTOR = 0.5


# ---------------------------------------------------------------------------
# Suppress noisy Qwen warnings that are harmless in training
# ---------------------------------------------------------------------------
class _QwenWarningFilter(logging.Filter):
    _SUPPRESSED = (
        "System prompt modified",
        "audio output may not work",
        "mrope_section",
    )
    def filter(self, record):
        if record.levelno == logging.WARNING:
            msg = record.getMessage()
            if any(s in msg for s in self._SUPPRESSED):
                return False
        return True

logging.getLogger().addFilter(_QwenWarningFilter())


# ---------------------------------------------------------------------------
# Tee stdout/stderr to a per-rank log file
# ---------------------------------------------------------------------------
class _TeeStream:
    """Duplicate writes to both the original stream and a log file."""
    def __init__(self, original, log_fp):
        self._orig   = original
        self._log_fp = log_fp

    def write(self, msg):
        self._orig.write(msg)
        if msg and msg.strip():
            # Skip tqdm progress-bar lines
            if "\r" not in msg and "it/s]" not in msg and "s/it]" not in msg:
                self._log_fp.write(msg.strip() + "\n")
                self._log_fp.flush()

    def flush(self):
        self._orig.flush()

    def __getattr__(self, name):
        return getattr(self._orig, name)


def _setup_log_tee(output_dir: str, rank: int):
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, f"rank_{rank}.log")
    log_fp   = open(log_path, "w", encoding="utf-8")
    log_fp.write(f"\n{'='*60}\n[{datetime.now()}] Training started (rank {rank})\n{'='*60}\n")
    log_fp.flush()

    # Tee stdout/stderr (catches print() calls)
    sys.stdout = _TeeStream(sys.stdout, log_fp)
    sys.stderr = _TeeStream(sys.stderr, log_fp)

    # Also attach a FileHandler to the root logger so that logger.info/warning
    # calls are captured regardless of when the StreamHandler was created.
    fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter("%(asctime)s  %(levelname)s  %(message)s"))
    fh.addFilter(_QwenWarningFilter())
    logging.getLogger().addHandler(fh)

    return log_fp


# ---------------------------------------------------------------------------
# Save all training parameters to JSON for reproducibility
# ---------------------------------------------------------------------------
def save_training_config(args, model_config: dict, output_dir: str):
    config = {
        "timestamp":    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "args":         {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "model_config": model_config,
    }
    path = os.path.join(output_dir, "training_config.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False, default=str)
    return path


# ---------------------------------------------------------------------------
# VAE loader
# ---------------------------------------------------------------------------

def _resolve_vae_ckpt(path: Optional[str], config_path: str) -> Optional[str]:
    if not path or os.path.isabs(path):
        return path
    return str((Path(config_path).resolve().parent / path).resolve())


def init_audio_vae(config_path: str):
    """Load the MMAudio VAE from a YAML config file."""
    with open(config_path) as f:
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


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class AudioDataset(Dataset):
    """Simple dataset that reads audio files listed in a JSONL metadata file.

    Each item in the JSONL file must have:
        audio_path  : path to a .wav file (relative to repo_root or absolute)
        caption     : text description
        duration    : clip length in seconds
    """

    def __init__(
        self,
        metadata_path: str,
        sample_rate: int,
        max_duration: float = 10.0,
        repo_root: Path = _REPO_ROOT,
    ):
        self.sample_rate = sample_rate
        self.max_samples = int(max_duration * sample_rate)
        self.repo_root = repo_root

        with open(metadata_path) as f:
            self.records = [json.loads(line) for line in f if line.strip()]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        rec = self.records[idx]

        audio_path = rec["audio_path"]
        if not os.path.isabs(audio_path):
            audio_path = str(self.repo_root / audio_path)

        wav, sr = torchaudio.load(audio_path)
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        if sr != self.sample_rate:
            wav = AF.resample(wav, sr, self.sample_rate)

        # Trim or pad to max_samples
        wav = wav[0]  # [T]
        if wav.shape[0] > self.max_samples:
            wav = wav[: self.max_samples]
        else:
            wav = F.pad(wav, (0, self.max_samples - wav.shape[0]))

        return {
            "wav": wav,            # [T]  float32 waveform
            "caption": rec["caption"],
            "duration": float(rec.get("duration", self.max_samples / self.sample_rate)),
        }


def collate_fn(batch: List[dict]) -> dict:
    wavs      = torch.stack([b["wav"]      for b in batch])
    captions  = [b["caption"]  for b in batch]
    durations = [b["duration"] for b in batch]
    return {"wav": wavs, "caption": captions, "duration": durations}


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="UNISON fine-tuning example")

    p.add_argument("--metadata",        type=str,
                   default=str(_REPO_ROOT / "data/train/metadata.jsonl"),
                   help="Path to JSONL metadata file listing training samples.")
    p.add_argument("--output_dir",      type=str, default="outputs/finetune_example")
    p.add_argument("--model_config",    type=str,
                   default=str(_REPO_ROOT / "unison/config/D20S0_O_40ch.yaml"))
    p.add_argument("--vae_config",      type=str,
                   default=str(_REPO_ROOT / "unison/models/mmaudio/vae_config_44k.yaml"))
    p.add_argument("--omni_model_path", type=str, default="Qwen/Qwen2.5-Omni-7B",
                   help="Path or HF hub ID for Qwen2.5-Omni-7B.")
    p.add_argument("--pretrained_model_path", type=str, default=None,
                   help="Path to a pretrained UNISON checkpoint directory or .pt/.safetensors file.")
    p.add_argument("--resume_from_checkpoint", type=str, default=None,
                   help="'latest' or explicit checkpoint folder path to resume training.")

    p.add_argument("--max_duration",    type=float, default=10.0)
    p.add_argument("--batch_size",      type=int,   default=2)
    p.add_argument("--dataloader_num_workers", type=int, default=0)
    p.add_argument("--learning_rate",   type=float, default=1e-5)
    p.add_argument("--max_train_steps", type=int,   default=200)
    p.add_argument("--lr_warmup_steps", type=int,   default=10)
    p.add_argument("--weight_decay",    type=float, default=0.01)
    p.add_argument("--max_grad_norm",   type=float, default=1.0)
    p.add_argument("--mixed_precision", type=str,   default="bf16",
                   choices=["no", "fp16", "bf16"])
    p.add_argument("--seed",            type=int,   default=42)
    p.add_argument("--prob_uncond",     type=float, default=0.1,
                   help="Probability of dropping text condition for CFG training.")
    p.add_argument("--use_ema",         action="store_true", default=False)
    p.add_argument("--ema_decay",       type=float, default=0.999)
    p.add_argument("--checkpointing_steps", type=int, default=100)
    p.add_argument("--logging_steps",   type=int,   default=10,
                   help="Log loss/lr/grad_norm every N steps.")
    p.add_argument("--report_to",       type=str,   default="tensorboard",
                   choices=["tensorboard", "wandb", "none"],
                   help="Tracker backend for accelerator.log().")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        log_with=args.report_to if args.report_to != "none" else None,
        project_config=ProjectConfiguration(project_dir=args.output_dir),
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=False)],
    )

    set_seed(args.seed + accelerator.process_index)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")

    os.makedirs(args.output_dir, exist_ok=True)
    # Tee output to log file — main process only
    _log_fp = None
    if accelerator.is_main_process:
        _log_fp = _setup_log_tee(args.output_dir, 0)

    # -----------------------------------------------------------------------
    # Load model config
    # -----------------------------------------------------------------------
    with open(args.model_config) as f:
        model_config = yaml.safe_load(f)

    dit_depth = (model_config.get("mm_double_blocks_depth", 0)
                 + model_config.get("mm_single_blocks_depth", 0))
    logger.info(f"DiT depth: {dit_depth}")

    # -----------------------------------------------------------------------
    # Audio VAE (frozen)
    # -----------------------------------------------------------------------
    audio_vae, vae_scale_factor, vae_sample_rate = init_audio_vae(args.vae_config)
    audio_vae = audio_vae.to(accelerator.device).eval()
    for param in audio_vae.parameters():
        param.requires_grad_(False)
    logger.info(f"VAE loaded  scale_factor={vae_scale_factor}  sr={vae_sample_rate}")

    # -----------------------------------------------------------------------
    # Text encoder (frozen)
    # -----------------------------------------------------------------------
    omni_path = resolve_omni_model_path("omni-7b", args.omni_model_path)
    omni_last_layer_idx = model_config.get("omni_last_layer_idx", -1)
    text_encoder = QwenOmniThinkerExtractor(
        model_path=omni_path,
        dit_depth=dit_depth,
        select_mode="interval",
        device=accelerator.device,
        dtype=torch.bfloat16,
        omni_last_layer_idx=omni_last_layer_idx,
    ).eval().requires_grad_(False)
    logger.info(f"Text encoder loaded (omni_last_layer_idx={omni_last_layer_idx}).")

    # -----------------------------------------------------------------------
    # DiT backbone (trainable)
    # -----------------------------------------------------------------------
    backbone_cfg = {**model_config, "concat_condition": True}
    backbone_cfg.pop("omni_last_layer_idx", None)
    model = UnisonBackbone(**backbone_cfg)

    if args.pretrained_model_path:
        ckpt = args.pretrained_model_path
        if os.path.isdir(ckpt):
            for name in ("ema_model.pt", "model.safetensors", "pytorch_model.bin"):
                cand = os.path.join(ckpt, name)
                if os.path.exists(cand):
                    ckpt = cand
                    break
        logger.info(f"Loading pretrained weights from {ckpt}")
        if ckpt.endswith(".safetensors"):
            from safetensors.torch import load_file as safe_load
            sd = safe_load(ckpt, device="cpu")
        else:
            sd = torch.load(ckpt, map_location="cpu")
        is_ema = any(k.startswith("shadow_params") for k in sd)
        if is_ema:
            ema_tmp = EMAModel(model.parameters(), decay=0.9999,
                               model_cls=UnisonBackbone, model_config=backbone_cfg)
            ema_tmp.load_state_dict(sd)
            ema_tmp.copy_to(model.parameters())
        else:
            model.load_state_dict(
                {k: v for k, v in sd.items()
                 if k not in {"decay", "num_updates", "shadow_params"}},
                strict=False,
            )

    # Ensure model parameters are float32 for mixed-precision training.
    # Pretrained safetensors may be stored in bf16; cast back so that accelerate's
    # autocast handles dtype conversion correctly during the forward pass.
    model.float()

    model.count_parameters(verbose=True)

    # -----------------------------------------------------------------------
    # EMA (optional)
    # -----------------------------------------------------------------------
    ema_model = None
    if args.use_ema:
        ema_model = EMAModel(
            model.parameters(), decay=args.ema_decay,
            model_cls=UnisonBackbone, model_config=backbone_cfg,
        )

    # -----------------------------------------------------------------------
    # Dataset and dataloader
    # -----------------------------------------------------------------------
    train_dataset = AudioDataset(
        metadata_path=args.metadata,
        sample_rate=vae_sample_rate,
        max_duration=args.max_duration,
    )
    logger.info(f"Training on {len(train_dataset)} samples from {args.metadata}")

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.dataloader_num_workers,
        collate_fn=collate_fn,
        drop_last=False,
    )

    # -----------------------------------------------------------------------
    # Optimizer and LR scheduler
    # -----------------------------------------------------------------------
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    lr_scheduler = get_scheduler(
        "cosine",
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=args.max_train_steps,
    )

    # -----------------------------------------------------------------------
    # Accelerate prepare  (must happen before computing num_epochs so that
    # len(train_dataloader) reflects the per-rank length after distribution)
    # -----------------------------------------------------------------------
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    # Compute num_epochs AFTER prepare so len(train_dataloader) is the
    # per-rank step count (divided by num_processes by DistributedSampler).
    num_epochs = math.ceil(args.max_train_steps / max(len(train_dataloader), 1))
    if ema_model:
        ema_model.to(accelerator.device)

    scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000)

    # -----------------------------------------------------------------------
    # Resume from checkpoint
    # -----------------------------------------------------------------------
    global_step = 0
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint == "latest":
            dirs = [d for d in Path(args.output_dir).iterdir()
                    if d.is_dir() and d.name.startswith("checkpoint-")]
            if dirs:
                ckpt_dir = max(dirs, key=lambda d: int(d.name.split("-")[-1]))
                accelerator.load_state(str(ckpt_dir))
                global_step = int(ckpt_dir.name.split("-")[-1])
                logger.info(f"Resumed from {ckpt_dir}")
        else:
            accelerator.load_state(args.resume_from_checkpoint)
            global_step = int(Path(args.resume_from_checkpoint).name.split("-")[-1])

    if accelerator.is_main_process:
        # Save all training parameters for reproducibility
        cfg_path = save_training_config(args, model_config, args.output_dir)
        logger.info(f"Training config saved to {cfg_path}")
        # Sanitise config for tracker (lists → comma-separated strings)
        tracker_cfg = {k: (",".join(v) if isinstance(v, list) else v)
                       for k, v in vars(args).items()}
        accelerator.init_trackers("unison_finetune", config=tracker_cfg)

    # -----------------------------------------------------------------------
    # Training loop
    # -----------------------------------------------------------------------
    logger.info(
        f"Starting training: {args.max_train_steps} steps, "
        f"batch_size={args.batch_size} × {accelerator.num_processes} GPUs, "
        f"mixed_precision={args.mixed_precision}"
    )
    omni_dim = model_config.get("omni_dim", 3584)

    progress = tqdm(
        total=args.max_train_steps,
        initial=global_step,
        disable=not accelerator.is_local_main_process,
        desc="Training",
    )

    running_loss = 0.0   # loss over the last logging_steps window
    total_loss   = 0.0   # cumulative loss over all steps (for global avg)
    step_t0      = time.time()

    for epoch in range(num_epochs):
        model.train()
        for batch in train_dataloader:
            wavs      = batch["wav"].to(accelerator.device)       # [B, T_wav]
            captions  = batch["caption"]

            B = wavs.shape[0]

            # Encode audio waveforms to latents on-the-fly (float32)
            with torch.no_grad():
                latents = audio_vae.wrapped_encode(wavs) * vae_scale_factor
                latents = latents.float()

            C = latents.shape[1]
            T = latents.shape[2]

            # Pure text-to-audio generation: source = zeros, mask = zeros
            source_latent = torch.zeros_like(latents)
            mask = torch.zeros(B, 1, T, device=accelerator.device, dtype=torch.float32)

            # CFG dropout
            drop = torch.rand(B, device=accelerator.device) < args.prob_uncond
            effective_captions = ["" if drop[i] else captions[i] for i in range(B)]

            # Text conditioning (float32 — cast from bf16 text encoder output)
            with torch.no_grad():
                result = text_encoder(effective_captions)
            omni_emb_list, omni_mask, *extra = result
            omni_last_emb = extra[0] if extra else torch.zeros(
                B, 1, omni_dim, device=accelerator.device, dtype=torch.float32
            )
            omni_emb_list  = [e.float() for e in omni_emb_list]
            omni_last_emb  = omni_last_emb.float()

            # Flow-matching noise schedule
            noise = torch.randn_like(latents)
            u = compute_density_for_timestep_sampling(
                weighting_scheme="logit_normal",
                batch_size=B,
                logit_mean=0.0,
                logit_std=1.0,
                mode_scale=1.29,
            )
            t = u.to(device=accelerator.device, dtype=torch.float32)
            noisy = (1.0 - t[:, None, None]) * latents + t[:, None, None] * noise

            # Model input: [noisy_target | source | mask] along channel dim
            x = torch.cat([noisy, source_latent, mask], dim=1)

            # Forward pass under accelerator.autocast():
            #   - model parameters stay float32 (optimizer updates in float32)
            #   - float32 inputs are auto-cast to bf16 inside the context
            #   - FlashAttention runs in bf16 as required
            #   - pred is bf16; cast back to float32 for the loss
            with accelerator.autocast():
                pred = model(
                    x=x,
                    t=t,
                    duration=None,
                    omni_emb_list=omni_emb_list,
                    omni_last_emb=omni_last_emb,
                    omni_mask=omni_mask,
                )[0]

            v_target = noise - latents  # flow-matching velocity target (float32)
            loss = F.mse_loss(pred.float(), v_target)

            accelerator.backward(loss)
            grad_norm = accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            global_step  += 1
            running_loss += loss.item()
            total_loss   += loss.item()
            progress.update(1)

            if ema_model and global_step % 10 == 0:
                ema_model.step(model.parameters())

            # ---- Periodic logging ----
            if global_step % args.logging_steps == 0 and accelerator.is_main_process:
                window_avg = running_loss / args.logging_steps
                total_avg  = total_loss  / global_step
                elapsed    = time.time() - step_t0
                sps        = args.logging_steps / max(elapsed, 1e-6)
                lr_now     = lr_scheduler.get_last_lr()[0]
                gn         = grad_norm.item() if isinstance(grad_norm, torch.Tensor) else (grad_norm or 0.0)

                logs = {
                    "train/loss":           loss.item(),
                    "train/avg_loss":       window_avg,
                    "train/total_avg_loss": total_avg,
                    "train/lr":             lr_now,
                    "train/grad_norm":      gn,
                    "train/steps_per_sec":  sps,
                }
                accelerator.log(logs, step=global_step)
                progress.set_postfix(
                    loss=f"{window_avg:.4f}",
                    avg=f"{total_avg:.4f}",
                    lr=f"{lr_now:.2e}",
                    gn=f"{gn:.3f}",
                )
                logger.info(
                    f"[step {global_step}] loss={window_avg:.4f} | total_avg={total_avg:.4f} | "
                    f"lr={lr_now:.2e} | grad_norm={gn:.3f} | {sps:.2f} steps/s"
                )
                running_loss = 0.0
                step_t0      = time.time()

            # ---- Checkpoint ----
            if global_step % args.checkpointing_steps == 0:
                save_dir = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                accelerator.save_state(save_dir)
                if ema_model and accelerator.is_main_process:
                    torch.save(ema_model.state_dict(), os.path.join(save_dir, "ema_model.pt"))
                logger.info(f"Saved checkpoint: {save_dir}")

            if global_step >= args.max_train_steps:
                break

        if global_step >= args.max_train_steps:
            break

    progress.close()
    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        final_dir = os.path.join(args.output_dir, "checkpoint-final")
        accelerator.save_state(final_dir)
        if ema_model:
            torch.save(ema_model.state_dict(), os.path.join(final_dir, "ema_model.pt"))
        logger.info(f"Training complete. Final checkpoint: {final_dir}")

    accelerator.end_training()


if __name__ == "__main__":
    main()
