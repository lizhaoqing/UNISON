"""
Qwen2.5-Omni text encoder with layer-wise hidden state extraction for deep LLM fusion.

Extracts hidden states from uniformly-sampled layers of a frozen Qwen2.5-Omni-7B
text backbone. Each extracted state is injected into the corresponding MM-DiT double-
stream block via a learned linear projection (deep fusion).

Key design:
  - Only the Thinker's language model is retained; vision, audio tower, and Talker
    sub-modules are dropped immediately after loading to save GPU memory.
  - A domain-specific system prompt is prepended; its token span is sliced off from
    every extracted hidden state so the DiT only receives user-prompt tokens.
  - padding_side='right' ensures the system prompt is always at the front (index 0),
    making the slice offset stable across batches.
"""

import gc
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor


OMNI_PRESET_MODEL_IDS = {
    "omni-3b": "Qwen/Qwen2.5-Omni-3B",
    "omni-7b": "Qwen/Qwen2.5-Omni-7B",
}


def resolve_omni_model_path(preset: str, model_path: Optional[str]) -> str:
    """Resolve Qwen2.5-Omni checkpoint path.

    - preset == 'omni'         -> use `model_path` directly (user-supplied path).
    - preset == 'omni-3b/7b'  -> use `model_path` if given, else fall back to HF hub id.
    """
    if preset == "omni":
        if not model_path:
            raise ValueError(
                "text_encoder_type='omni' requires --omni_model_path to be set."
            )
        return model_path
    if preset not in OMNI_PRESET_MODEL_IDS:
        raise ValueError(
            f"Unknown Omni preset {preset!r}. "
            f"Choose one of {list(OMNI_PRESET_MODEL_IDS) + ['omni']}."
        )
    return model_path or OMNI_PRESET_MODEL_IDS[preset]


class QwenOmniThinkerExtractor(nn.Module):
    """Frozen Qwen2.5-Omni Thinker language model used as a text-feature extractor.

    Returns per-layer hidden states for deep fusion into the DiT backbone.
    """

    def __init__(
        self,
        model_path: str = "Qwen/Qwen2.5-Omni-7B",
        dit_depth: int = 20,
        select_mode: str = "interval",
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        omni_last_layer_idx: Optional[int] = None,
    ):
        """
        Args:
            model_path:          Path to Qwen2.5-Omni model (local or HF hub).
            dit_depth:           Number of MM-DiT double-stream blocks; determines
                                 how many LLM layers to sample.
            select_mode:         'interval' = uniformly-spaced layers;
                                 'last'     = last dit_depth layers.
            device:              Device to place the text backbone on.
            dtype:               Computation dtype (bfloat16 recommended).
            omni_last_layer_idx: If set, also return a single "global" embedding from
                                 this LLM layer index (0-indexed from end when negative).
                                 None = only return per-block hidden states.
        """
        super().__init__()
        self.device = device
        self.dtype = dtype

        print(f"Loading Qwen2.5-Omni Thinker from {model_path}...")

        # Load full model then discard everything except the language backbone.
        full_model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=dtype,
            device_map="cpu",
            attn_implementation="flash_attention_2",
        )
        self.text_backbone = full_model.thinker.model.eval()

        for param in self.text_backbone.parameters():
            param.requires_grad = False

        print("Dropping unused sub-modules (talker, token2wav, visual, audio_tower)...")
        del full_model.talker
        del full_model.token2wav
        del full_model.thinker.visual
        del full_model.thinker.audio_tower
        del full_model

        gc.collect()

        print(f"Moving text backbone to {device}...")
        self.text_backbone = self.text_backbone.to(device)
        torch.cuda.empty_cache()

        # Load processor; force right-side padding so the system prompt is always
        # at index 0, making the slice offset constant.
        self.processor = Qwen2_5OmniProcessor.from_pretrained(model_path)
        self.processor.tokenizer.padding_side = "right"

        # Domain-specific system prompt for audio generation conditioning.
        self.system_prompt_content = [{
            "type": "text",
            "text": (
                "You are an expert Audio Director and Sound Engineer. "
                "Your task is to mentally simulate and analyze the acoustic scene described in the text. "
                "Pay attention to the following three layers: "
                "1. Voice & Prosody: If there is speech, identify the speaker's gender, age, emotion, accent, and exact spoken content. "
                "2. Environmental Sounds: Identify background ambience, distinct sound effects, and their textures. "
                "3. Spatial Mix: Determine how the voice and background sounds overlap and interact in the scene. "
                "Extract deep semantic representations to guide the generation of high-fidelity speech, audio, or a mixture of both."
            )
        }]

        # Pre-compute system-prompt token length so we can slice it off during forward.
        print("Pre-computing system prompt token length...")
        sys_str = self.processor.apply_chat_template(
            [{"role": "system", "content": self.system_prompt_content}],
            add_generation_prompt=False,
            tokenize=False,
        )
        sys_tokens = self.processor(
            sys_str,
            return_tensors="pt",
            padding=False,
            truncation=False,
            add_special_tokens=False,
        )
        self.system_token_len = sys_tokens.input_ids.shape[1]
        print(f"System prompt length: {self.system_token_len} tokens (will be sliced off in forward)")

        # Map LLM layer indices to DiT block indices.
        self.total_llm_layers = len(self.text_backbone.layers)
        self.layer_indices = self._get_layer_indices(self.total_llm_layers, dit_depth, select_mode)
        print(f"Selected LLM layer indices: {self.layer_indices}")

        # omni_last_layer_idx: index into hidden_states tuple for an optional global
        # embedding (e.g. -1 = last layer, -2 = second-to-last).
        # When None, forward() returns (extracted_states, mask) only.
        self.omni_last_layer_idx = omni_last_layer_idx
        if omni_last_layer_idx is not None:
            resolved = self.total_llm_layers + 1 + omni_last_layer_idx
            print(f"Global embedding from LLM layer index {omni_last_layer_idx} (= layer {resolved} of {self.total_llm_layers})")

    def _get_layer_indices(self, total_llm: int, dit_depth: int, mode: str):
        """Return list of LLM layer indices to sample for each DiT block."""
        available = list(range(1, total_llm + 1))
        if total_llm >= dit_depth:
            if mode == "interval":
                idxs = np.linspace(0, len(available) - 1, dit_depth, dtype=int)
                return [available[i] for i in idxs]
            else:
                return available[-dit_depth:]
        else:
            # Repeat last layer to pad up to dit_depth.
            return available + [available[-1]] * (dit_depth - total_llm)

    @torch.no_grad()
    def forward(self, text_list: list):
        """Extract per-block hidden states for a batch of text prompts.

        Args:
            text_list: List of prompt strings.

        Returns:
            extracted_states: List[Tensor], length == dit_depth.
                              Each tensor is [B, user_seq_len, hidden_dim].
            user_attention_mask: [B, user_seq_len] — system-prompt tokens removed.
            omni_last_emb (only if omni_last_layer_idx is set):
                              [B, user_seq_len, hidden_dim] from a single chosen layer.
        """
        batch_conversations = [
            [
                {"role": "system", "content": self.system_prompt_content},
                {"role": "user", "content": [{"type": "text", "text": txt}]},
            ]
            for txt in text_list
        ]

        text_prompts = self.processor.apply_chat_template(
            batch_conversations,
            add_generation_prompt=True,
            tokenize=False,
        )

        # Tokenize with right-padding: layout is [System][User][Pad…]
        inputs = self.processor(
            text_prompts,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=256,
            add_special_tokens=False,
        ).to(self.device)

        outputs = self.text_backbone(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            output_hidden_states=True,
            return_dict=True,
            use_cache=False,
        )

        all_states = outputs.hidden_states

        # Slice off system-prompt tokens from each selected layer.
        extracted_states = [
            all_states[idx].to(self.dtype)[:, self.system_token_len:, :]
            for idx in self.layer_indices
        ]
        user_attention_mask = inputs.attention_mask[:, self.system_token_len:]

        if self.omni_last_layer_idx is not None:
            omni_last_emb = all_states[self.omni_last_layer_idx].to(self.dtype)[:, self.system_token_len:, :]
            return extracted_states, user_attention_mask, omni_last_emb

        return extracted_states, user_attention_mask
