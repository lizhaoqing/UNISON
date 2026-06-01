"""
Qwen3 (Causal LM) hidden-state extractor for training / inference.

Same output contract as QwenOmniThinkerExtractor.forward:
  - extracted_states: List[Tensor], one per selected layer (dit_depth layers)
  - user_attention_mask: Tensor [B, L]
  - omni_last_emb: optional Tensor from omni_last_layer_idx

No system prompt: user text is encoded via chat template (single user turn) when
the tokenizer defines chat_template; otherwise plain encoding.

Requires transformers >= 4.51 for model_type qwen3.

Preset HF repo IDs (override with model_path):
  - qwen3-4b-instruct  -> Qwen/Qwen3-4B-Instruct-2507  (hidden 2560, 36 layers)
  - qwen3-1.7b         -> Qwen/Qwen3-1.7B              (hidden 2048, 28 layers)
  - qwen3-0.6b         -> Qwen/Qwen3-0.6B              (hidden 1024, 28 layers)

Set model_config omni_dim to the chosen model's hidden_size (see config.json).
"""

from __future__ import annotations

import gc
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer


# Preset name -> default Hugging Face model id
QWEN3_PRESET_MODEL_IDS = {
    "qwen3-4b-instruct": "Qwen/Qwen3-4B-Instruct-2507",
    "qwen3-1.7b": "Qwen/Qwen3-1.7B",
    "qwen3-0.6b": "Qwen/Qwen3-0.6B",
}


def resolve_qwen3_model_path(preset: str, model_path: Optional[str]) -> str:
    """Return checkpoint path: explicit model_path wins, else preset default."""
    if model_path:
        return model_path
    if preset not in QWEN3_PRESET_MODEL_IDS:
        raise ValueError(
            f"Unknown Qwen3 preset {preset!r}. "
            f"Choose one of {list(QWEN3_PRESET_MODEL_IDS)} or pass model_path."
        )
    return QWEN3_PRESET_MODEL_IDS[preset]


def _load_attn_implementation():
    try:
        import flash_attn  # noqa: F401

        return "flash_attention_2"
    except Exception:
        return "sdpa"


class Qwen3CausalLMHiddenExtractor(nn.Module):
    """Extract multi-layer hidden states from Qwen3ForCausalLM (text backbone only)."""

    def __init__(
        self,
        model_path: str,
        dit_depth: int = 20,
        select_mode: str = "interval",
        device="cuda",
        dtype=torch.bfloat16,
        omni_last_layer_idx: Optional[int] = None,
        max_length: int = 256,
    ):
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.max_length = max_length
        self.omni_last_layer_idx = omni_last_layer_idx

        attn_impl = _load_attn_implementation()
        print(f"Loading Qwen3 Causal LM from {model_path} (attn={attn_impl})...")

        full_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            device_map="cpu",
            attn_implementation=attn_impl,
        )
        self.text_backbone = full_model.model.eval()
        for p in self.text_backbone.parameters():
            p.requires_grad = False

        del full_model
        gc.collect()
        print(f"Moving Qwen3 backbone to {device}...")
        self.text_backbone = self.text_backbone.to(device)
        torch.cuda.empty_cache()

        hs = getattr(self.text_backbone.config, "hidden_size", None)
        if hs is not None:
            print(
                f"Qwen3 hidden_size={hs} — set model_config omni_dim to this value "
                f"(e.g. 2560 / 2048 / 1024 for 4B-Instruct / 1.7B / 0.6B)."
            )

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"

        self.total_llm_layers = len(self.text_backbone.layers)
        self.layer_indices = self._get_layer_indices(self.total_llm_layers, dit_depth, select_mode)
        print(f"Selected layer indices: {self.layer_indices}")

        if omni_last_layer_idx is not None:
            print(
                f"omni_last_layer_idx={omni_last_layer_idx} "
                f"(hidden_states index; negative = from end)"
            )

    def _get_layer_indices(self, total_llm: int, dit_depth: int, mode: str) -> List[int]:
        available_indices = list(range(1, total_llm + 1))
        selected_indices = []
        if total_llm >= dit_depth:
            if mode == "interval":
                idx_array = np.linspace(0, len(available_indices) - 1, dit_depth, dtype=int)
                selected_indices = [available_indices[i] for i in idx_array]
            else:
                selected_indices = available_indices[-dit_depth:]
        else:
            selected_indices = available_indices[:]
            missing_count = dit_depth - total_llm
            selected_indices.extend([available_indices[-1]] * missing_count)
        return selected_indices

    def _encode_texts(self, text_list: List[str]) -> torch.Tensor:
        """Batch-encode strings to input_ids + attention_mask on device."""
        use_chat = getattr(self.tokenizer, "chat_template", None) is not None
        if use_chat:
            prompts = []
            for txt in text_list:
                messages = [{"role": "user", "content": txt}]
                try:
                    p = self.tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                        enable_thinking=False,
                    )
                except TypeError:
                    p = self.tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                prompts.append(p)
        else:
            prompts = list(text_list)

        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            add_special_tokens=False if use_chat else True,
        ).to(self.device)
        return inputs

    @torch.no_grad()
    def forward(self, text_list: List[str]):
        """
        Args:
            text_list: batch of user strings
        Returns:
            extracted_states, user_attention_mask[, omni_last_emb]
        """
        inputs = self._encode_texts(text_list)
        outputs = self.text_backbone(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            output_hidden_states=True,
            return_dict=True,
            use_cache=False,
        )
        all_states = outputs.hidden_states

        extracted_states = []
        for idx in self.layer_indices:
            feat = all_states[idx].to(self.dtype)
            extracted_states.append(feat)

        user_attention_mask = inputs.attention_mask

        if self.omni_last_layer_idx is not None:
            omni_last_emb = all_states[self.omni_last_layer_idx].to(self.dtype)
            return extracted_states, user_attention_mask, omni_last_emb

        return extracted_states, user_attention_mask


if __name__ == "__main__":
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen3-0.6B"
    ex = Qwen3CausalLMHiddenExtractor(
        model_path=path,
        dit_depth=8,
        device="cuda" if torch.cuda.is_available() else "cpu",
        dtype=torch.bfloat16,
        omni_last_layer_idx=-1,
    )
    out = ex(["hello", "test audio scene"])
    if len(out) == 3:
        a, m, last = out
    else:
        a, m = out
        last = None
    print("layers", len(a), "feat0", a[0].shape, "mask", m.shape, "last", None if last is None else last.shape)
