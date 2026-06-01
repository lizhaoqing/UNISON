# UnisonBackbone — MM-DiT backbone for UNISON.
#
# Task identity and source audio are encoded entirely through the input tensor:
#   model input shape: [B, 2*C+1, T]
#     [:, :C, :]   — noisy target latent
#     [:, C:2C, :] — source/reference latent (zeros for pure generation)
#     [:, 2C, :]   — task mask (0=generate, 1=edit, 2=zero-shot-TTS reference region)
#
# The UniversalPatchEmbed layer (concat_condition=True) handles the 2C+1 → hidden_size
# projection; C is set by in_channels in the model config and matches the VAE latent dim.
#
# Deep LLM fusion: each MM-DiT block receives the hidden state from the corresponding
# layer of a frozen MLLM (Qwen2.5-Omni or Qwen3) via a learned linear projection,
# providing depth-matched semantic conditioning.

from typing import Any, List, Tuple, Optional, Union, Dict

import torch
import torch.nn as nn
from einops import rearrange
from loguru import logger

from diffusers.models import ModelMixin
from diffusers.configuration_utils import ConfigMixin, register_to_config

from .modules.activation_layers import get_activation_layer
from .modules.norm_layers import get_norm_layer
from .modules.embed_layers import (
    TimestepEmbedder,
    PatchEmbed,
    VisionProjection,
    UniversalPatchEmbed,
    DurationEmbedder
)
from .modules.attention import parallel_attention
from .modules.posemb_layers import apply_rotary_emb, get_nd_rotary_pos_embed, get_audio_rotary_pos_embed
from .modules.mlp_layers import MLP, MLPEmbedder, FinalLayer, LinearWarpforSingle
from .modules.modulate_layers import ModulateDiT, modulate, apply_gate
from .modules.token_refiner import SingleTokenRefiner

from unison.utils.communications import all_gather
from unison.utils.infer_utils import torch_compile_wrapper

from unison.commons.parallel_states import get_parallel_state

class MMDoubleStreamBlock(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        heads_num: int,
        mlp_width_ratio: float,
        mlp_act_type: str = "gelu_tanh",
        attn_mode: str = None,
        qk_norm: bool = True,
        qk_norm_type: str = "rms",
        qkv_bias: bool = False,
        has_text_ffn: bool = True,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        self.deterministic = False
        self.heads_num = heads_num
        self.attn_mode = attn_mode
        self.has_text_ffn = has_text_ffn

        head_dim = hidden_size // heads_num
        mlp_hidden_dim = int(hidden_size * mlp_width_ratio)

        self.img_mod = ModulateDiT(
            hidden_size, factor=6, act_layer=get_activation_layer("silu"), **factory_kwargs
        )
        self.img_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6, **factory_kwargs)
        self.img_attn_q = nn.Linear(hidden_size, hidden_size, bias=qkv_bias, **factory_kwargs)
        self.img_attn_k = nn.Linear(hidden_size, hidden_size, bias=qkv_bias, **factory_kwargs)
        self.img_attn_v = nn.Linear(hidden_size, hidden_size, bias=qkv_bias, **factory_kwargs)

        qk_norm_layer = get_norm_layer(qk_norm_type)
        self.img_attn_q_norm = (
            qk_norm_layer(head_dim, elementwise_affine=True, eps=1e-6, **factory_kwargs) if qk_norm else nn.Identity()
        )
        self.img_attn_k_norm = (
            qk_norm_layer(head_dim, elementwise_affine=True, eps=1e-6, **factory_kwargs) if qk_norm else nn.Identity()
        )
        self.img_attn_proj = nn.Linear(hidden_size, hidden_size, bias=qkv_bias, **factory_kwargs)

        self.img_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6, **factory_kwargs)
        self.img_mlp = MLP(hidden_size, mlp_hidden_dim, act_layer=get_activation_layer(mlp_act_type), bias=True, **factory_kwargs)

        txt_mod_factor = 6 if has_text_ffn else 2
        self.txt_mod = ModulateDiT(
            hidden_size, factor=txt_mod_factor, act_layer=get_activation_layer("silu"), **factory_kwargs
        )
        self.txt_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6, **factory_kwargs)

        self.txt_attn_q = nn.Linear(hidden_size, hidden_size, bias=qkv_bias, **factory_kwargs)
        self.txt_attn_k = nn.Linear(hidden_size, hidden_size, bias=qkv_bias, **factory_kwargs)
        self.txt_attn_v = nn.Linear(hidden_size, hidden_size, bias=qkv_bias, **factory_kwargs)

        self.txt_attn_q_norm = (
            qk_norm_layer(head_dim, elementwise_affine=True, eps=1e-6, **factory_kwargs) if qk_norm else nn.Identity()
        )
        self.txt_attn_k_norm = (
            qk_norm_layer(head_dim, elementwise_affine=True, eps=1e-6, **factory_kwargs) if qk_norm else nn.Identity()
        )

        if has_text_ffn:
            self.txt_attn_proj = nn.Linear(hidden_size, hidden_size, bias=qkv_bias, **factory_kwargs)
            self.txt_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6, **factory_kwargs)
            self.txt_mlp = MLP(hidden_size, mlp_hidden_dim, act_layer=get_activation_layer(mlp_act_type), bias=True, **factory_kwargs)

        self.hybrid_seq_parallel_attn = None

    def enable_deterministic(self):
        self.deterministic = True

    def disable_deterministic(self):
        self.deterministic = False

    @torch_compile_wrapper()
    def forward(
        self,
        img: torch.Tensor,
        txt: torch.Tensor,
        vec: torch.Tensor,
        freqs_cis: tuple = None,
        text_mask=None,
        attn_param=None,
        is_flash=False,
        block_idx=None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        (
            img_mod1_shift,
            img_mod1_scale,
            img_mod1_gate,
            img_mod2_shift,
            img_mod2_scale,
            img_mod2_gate,
        ) = self.img_mod(vec).chunk(6, dim=-1)

        if self.has_text_ffn:
            (
                txt_mod1_shift,
                txt_mod1_scale,
                txt_mod1_gate,
                txt_mod2_shift,
                txt_mod2_scale,
                txt_mod2_gate,
            ) = self.txt_mod(vec).chunk(6, dim=-1)
        else:
            txt_mod1_shift, txt_mod1_scale = self.txt_mod(vec).chunk(2, dim=-1)

        img_modulated = self.img_norm1(img)
        img_modulated = modulate(img_modulated, shift=img_mod1_shift, scale=img_mod1_scale)

        img_q = self.img_attn_q(img_modulated)
        img_k = self.img_attn_k(img_modulated)
        img_v = self.img_attn_v(img_modulated)
        img_q = rearrange(img_q, "B L (H D) -> B L H D", H=self.heads_num)
        img_k = rearrange(img_k, "B L (H D) -> B L H D", H=self.heads_num)
        img_v = rearrange(img_v, "B L (H D) -> B L H D", H=self.heads_num)
        img_q = self.img_attn_q_norm(img_q).to(img_v)
        img_k = self.img_attn_k_norm(img_k).to(img_v)

        if freqs_cis is not None:
            img_qq, img_kk = apply_rotary_emb(img_q, img_k, freqs_cis, head_first=False)
            assert (
                img_qq.shape == img_q.shape and img_kk.shape == img_k.shape
            ), f"img_kk: {img_qq.shape}, img_q: {img_q.shape}, img_kk: {img_kk.shape}, img_k: {img_k.shape}"
            img_q, img_k = img_qq, img_kk

        txt_modulated = self.txt_norm1(txt)
        txt_modulated = modulate(txt_modulated, shift=txt_mod1_shift, scale=txt_mod1_scale)
        txt_q = self.txt_attn_q(txt_modulated)
        txt_k = self.txt_attn_k(txt_modulated)
        txt_v = self.txt_attn_v(txt_modulated)
        txt_q = rearrange(txt_q, "B L (H D) -> B L H D", H=self.heads_num)
        txt_k = rearrange(txt_k, "B L (H D) -> B L H D", H=self.heads_num)
        txt_v = rearrange(txt_v, "B L (H D) -> B L H D", H=self.heads_num)
        txt_q = self.txt_attn_q_norm(txt_q).to(txt_v)
        txt_k = self.txt_attn_k_norm(txt_k).to(txt_v)

        attn_mode = 'flash' if is_flash else self.attn_mode
        attn = parallel_attention(
            (img_q, txt_q),
            (img_k, txt_k),
            (img_v, txt_v),
            img_q_len=img_q.shape[1],
            img_kv_len=img_k.shape[1],
            text_mask=text_mask,
            attn_mode=attn_mode,
            attn_param=attn_param,
            block_idx=block_idx,
        )

        img_attn, txt_attn = attn[:, :img_q.shape[1]].contiguous(), attn[:, img_q.shape[1]:].contiguous()

        img = img + apply_gate(self.img_attn_proj(img_attn), gate=img_mod1_gate)
        img = img + apply_gate(
            self.img_mlp(
                modulate(self.img_norm2(img), shift=img_mod2_shift, scale=img_mod2_scale)
            ),
            gate=img_mod2_gate,
        )

        if self.has_text_ffn:
            txt = txt + apply_gate(self.txt_attn_proj(txt_attn), gate=txt_mod1_gate)
            txt = txt + apply_gate(
                self.txt_mlp(modulate(self.txt_norm2(txt), shift=txt_mod2_shift, scale=txt_mod2_scale)),
                gate=txt_mod2_gate,
            )

        return img, txt


class MMSingleStreamBlock(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        heads_num: int,
        mlp_width_ratio: float = 4.0,
        mlp_act_type: str = "gelu_tanh",
        attn_mode: str = None,
        qk_norm: bool = True,
        qk_norm_type: str = "rms",
        qk_scale: float = None,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        self.deterministic = False
        self.attn_mode = attn_mode

        self.hidden_size = hidden_size
        self.heads_num = heads_num
        head_dim = hidden_size // heads_num
        mlp_hidden_dim = int(hidden_size * mlp_width_ratio)
        self.mlp_hidden_dim = mlp_hidden_dim
        self.scale = qk_scale or head_dim ** -0.5

        self.linear1_q = nn.Linear(hidden_size, hidden_size, **factory_kwargs)
        self.linear1_k = nn.Linear(hidden_size, hidden_size, **factory_kwargs)
        self.linear1_v = nn.Linear(hidden_size, hidden_size, **factory_kwargs)
        self.linear1_mlp = nn.Linear(hidden_size, mlp_hidden_dim, **factory_kwargs)
        self.linear2 = LinearWarpforSingle(hidden_size + mlp_hidden_dim, hidden_size, bias=True, **factory_kwargs)
        self.mlp_act = get_activation_layer(mlp_act_type)()

        qk_norm_layer = get_norm_layer(qk_norm_type)
        self.q_norm = (
            qk_norm_layer(head_dim, elementwise_affine=True, eps=1e-6, **factory_kwargs) if qk_norm else nn.Identity()
        )
        self.k_norm = (
            qk_norm_layer(head_dim, elementwise_affine=True, eps=1e-6, **factory_kwargs) if qk_norm else nn.Identity()
        )

        self.pre_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6, **factory_kwargs)
        self.modulation = ModulateDiT(hidden_size, factor=3, act_layer=get_activation_layer("silu"), **factory_kwargs)
        self.hybrid_seq_parallel_attn = None

    def enable_deterministic(self):
        self.deterministic = True

    def disable_deterministic(self):
        self.deterministic = False

    def forward(
        self,
        x: torch.Tensor,
        vec: torch.Tensor,
        txt_len: int,
        freqs_cis: Tuple[torch.Tensor, torch.Tensor] = None,
        text_mask=None,
        attn_param=None,
        is_flash=False,
    ) -> torch.Tensor:
        mod_shift, mod_scale, mod_gate = self.modulation(vec).chunk(3, dim=-1)
        x_mod = modulate(self.pre_norm(x), shift=mod_shift, scale=mod_scale)

        q = self.linear1_q(x_mod)
        k = self.linear1_k(x_mod)
        v = self.linear1_v(x_mod)

        q = rearrange(q, "B L (H D) -> B L H D", H=self.heads_num)
        k = rearrange(k, "B L (H D) -> B L H D", H=self.heads_num)
        v = rearrange(v, "B L (H D) -> B L H D", H=self.heads_num)

        mlp = self.linear1_mlp(x_mod)

        q = self.q_norm(q).to(v)
        k = self.k_norm(k).to(v)

        img_q, txt_q = q[:, :-txt_len, :, :], q[:, -txt_len:, :, :]
        img_k, txt_k = k[:, :-txt_len, :, :], k[:, -txt_len:, :, :]
        img_v, txt_v = v[:, :-txt_len, :, :], v[:, -txt_len:, :, :]
        img_qq, img_kk = apply_rotary_emb(img_q, img_k, freqs_cis, head_first=False)
        assert (
            img_qq.shape == img_q.shape and img_kk.shape == img_k.shape
        ), f"img_kk: {img_qq.shape}, img_q: {img_q.shape}, img_kk: {img_kk.shape}, img_k: {img_k.shape}"
        img_q, img_k = img_qq, img_kk

        if is_flash:
            attn_mode = 'flash'
        else:
            attn_mode = self.attn_mode
        attn = parallel_attention(
            (img_q, txt_q),
            (img_k, txt_k),
            (img_v, txt_v),
            img_q_len=img_q.shape[1],
            img_kv_len=img_k.shape[1],
            text_mask=text_mask,
            attn_mode=attn_mode,
            attn_param=attn_param,
        )
        output = self.linear2(attn, self.mlp_act(mlp))

        return x + apply_gate(output, gate=mod_gate)


class UnisonBackbone(ModelMixin, ConfigMixin):
    """Channel-cat backbone: no ref_latents, no ref-isolated AdaLN, continuous RoPE."""

    @register_to_config
    def __init__(
        self,
        in_channels: int = 48,
        out_channels: int = None,
        patch_size: list = [1, 2, 2],
        hidden_size: int = 1024,
        heads_num: int = 8,
        concat_condition: bool = False,
        mlp_act_type: str = "gelu_tanh",
        mlp_width_ratio: float = 4.0,
        mm_double_blocks_depth: int = 18,
        mm_single_blocks_depth: int = 0,
        qkv_bias: bool = True,
        qk_norm: bool = True,
        qk_norm_type: str = "rms",
        attn_mode: str = "flash",
        attn_param: dict = None,
        rope_dim_list: list = [16, 56, 56],
        rope_theta: int = 10000,
        omni_dim: int = 1280,
        is_reshape_temporal_channels: bool = False,
        use_duration_embedding: bool = False,
        is_audio_type: bool = False,
        use_omni_embedding: bool = False,
        use_omni_last_embedding: bool = False,
        projector_type: str = "linear",
        guidance_embed: bool = False,
        temporal_rope_scaling_factor: float = 1.0,
        repa_z_dim: int = None,
        repa_layer_num: int = None,
        audio_patch_type: str = "conv_mlp",
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.heads_num = heads_num
        self.is_audio_type = is_audio_type
        self.concat_condition = concat_condition
        self.out_channels = out_channels or in_channels
        self.patch_size = patch_size
        self.rope_dim_list = rope_dim_list
        self.rope_theta = rope_theta
        self.temporal_rope_scaling_factor = temporal_rope_scaling_factor
        self.mm_double_blocks_depth = mm_double_blocks_depth
        self.mm_single_blocks_depth = mm_single_blocks_depth
        self.total_blocks_depth = mm_double_blocks_depth + mm_single_blocks_depth
        self.guidance_embed = guidance_embed
        self.use_duration_embedding = use_duration_embedding
        self.use_omni_embedding = use_omni_embedding
        self.projector_type = projector_type
        self.repa_z_dim = repa_z_dim
        self.repa_layer_num = repa_layer_num
        self.use_omni_last_embedding = use_omni_last_embedding
        self.gradient_checkpointing = False

        self.img_in = UniversalPatchEmbed(
            patch_size=patch_size,
            in_chans=in_channels,
            embed_dim=hidden_size,
            is_audio=is_audio_type,
            is_reshape_temporal_channels=is_reshape_temporal_channels,
            audio_kernel_size=7,
            audio_padding=3,
            concat_condition=concat_condition,
            audio_patch_type=audio_patch_type,
        )

        self.time_in = TimestepEmbedder(hidden_size, get_activation_layer("silu"))
        self.guidance_in = TimestepEmbedder(hidden_size, get_activation_layer("silu")) if guidance_embed else None

        if use_duration_embedding:
            self.duration_embedder = DurationEmbedder(hidden_size, min_value=0, max_value=30)
        else:
            self.duration_embedder = None

        if use_omni_embedding:
            if self.projector_type == "linear":
                self.deep_fusion_projs = nn.ModuleList([
                    nn.Linear(omni_dim, hidden_size) for _ in range(self.total_blocks_depth)
                ])
            elif self.projector_type == "mlp":
                self.deep_fusion_projs = nn.ModuleList([
                    nn.Sequential(
                        nn.Linear(omni_dim, hidden_size),
                        nn.SiLU(),
                        nn.Linear(hidden_size, hidden_size)
                    ) for _ in range(self.total_blocks_depth)
                ])
            else:
                raise ValueError(f"Invalid projector type: {self.projector_type}")
        else:
            self.deep_fusion_projs = None

        if self.use_omni_last_embedding:
            if self.projector_type == "linear":
                self.omni_last_proj = nn.Linear(omni_dim, hidden_size)
            elif self.projector_type == "mlp":
                self.omni_last_proj = nn.Sequential(
                    nn.Linear(omni_dim, hidden_size),
                    nn.SiLU(),
                    nn.Linear(hidden_size, hidden_size)
                )
            else:
                raise ValueError(f"Invalid projector type: {self.projector_type}")
        else:
            self.omni_last_proj = None

        need_text_ffn = self.use_omni_last_embedding or (self.duration_embedder is not None)

        self.double_blocks = nn.ModuleList([
            MMDoubleStreamBlock(
                hidden_size=hidden_size,
                heads_num=heads_num,
                mlp_width_ratio=mlp_width_ratio,
                mlp_act_type=mlp_act_type,
                attn_mode="flash",
                qk_norm=qk_norm,
                qk_norm_type=qk_norm_type,
                qkv_bias=qkv_bias,
                has_text_ffn=need_text_ffn,
            ) for _ in range(mm_double_blocks_depth)
        ])
        self.single_blocks = nn.ModuleList([
            MMSingleStreamBlock(
                hidden_size=hidden_size,
                heads_num=heads_num,
                mlp_width_ratio=mlp_width_ratio,
                mlp_act_type=mlp_act_type,
                attn_mode="flash",
                qk_norm=qk_norm,
                qk_norm_type=qk_norm_type,
            ) for _ in range(mm_single_blocks_depth)
        ])

        if self.is_audio_type:
            final_patch_size = (1, 1, 1)
        else:
            final_patch_size = patch_size

        self.final_layer = FinalLayer(hidden_size, final_patch_size, self.out_channels, get_activation_layer("silu"))

        if self.repa_z_dim is not None and self.repa_layer_num is not None:
            self.repa_proj = nn.Linear(hidden_size, repa_z_dim)
        else:
            self.repa_proj = None

    def enable_gradient_checkpointing(self):
        self.gradient_checkpointing = True

    def disable_gradient_checkpointing(self):
        self.gradient_checkpointing = False

    def get_rotary_pos_embed(self, grid_sizes):
        if self.is_audio_type:
            return get_audio_rotary_pos_embed(
                rope_dim_list=self.rope_dim_list,
                length=grid_sizes[0],
                theta=self.rope_theta,
                freqs_scaling=self.temporal_rope_scaling_factor,
                use_real=True
            )
        else:
            return get_nd_rotary_pos_embed(
                self.rope_dim_list,
                grid_sizes,
                theta=self.rope_theta,
                use_real=True,
                theta_rescale_factor=1
            )

    def count_parameters(self, verbose=True):
        total_params = 0
        trainable_params = 0

        module_params = {
            "Patch Embed (img_in)": 0,
            "Timestep Embed (time_in)": 0,
            "Guidance Embed": 0,
            "Duration Embed": 0,
            "Omni Last Projection": 0,
            "Deep Fusion Projections": 0,
            "Double Blocks": 0,
            "Single Blocks": 0,
            "Final Layer": 0,
            "RepA Projection": 0,
            "Others": 0,
        }

        for name, param in self.named_parameters():
            num_params = param.numel()
            total_params += num_params
            if param.requires_grad:
                trainable_params += num_params

            if "img_in" in name:
                module_params["Patch Embed (img_in)"] += num_params
            elif "time_in" in name:
                module_params["Timestep Embed (time_in)"] += num_params
            elif "guidance_in" in name:
                module_params["Guidance Embed"] += num_params
            elif "duration_embedder" in name:
                module_params["Duration Embed"] += num_params
            elif "omni_last_proj" in name:
                module_params["Omni Last Projection"] += num_params
            elif "deep_fusion_projs" in name:
                module_params["Deep Fusion Projections"] += num_params
            elif "double_blocks" in name:
                module_params["Double Blocks"] += num_params
            elif "single_blocks" in name:
                module_params["Single Blocks"] += num_params
            elif "final_layer" in name:
                module_params["Final Layer"] += num_params
            elif "repa_proj" in name:
                module_params["RepA Projection"] += num_params
            else:
                module_params["Others"] += num_params

        if verbose:
            print(f"\n{'='*50}")
            print(f"  Model Parameter Statistics")
            print(f"{'='*50}")
            for category, count in module_params.items():
                if count > 0:
                    print(f"  {category:<30}: {count / 1e6:>8.2f} M")
            print(f"  {'-'*48}")
            print(f"  {'Total Parameters':<30}: {total_params / 1e6:>8.2f} M ({total_params / 1e9:.3f} B)")
            print(f"  {'Trainable Parameters':<30}: {trainable_params / 1e6:>8.2f} M ({trainable_params / 1e9:.3f} B)")
            print(f"{'='*50}\n")

        return total_params

    def prepare_packed_indices(self, mask_list):
        total_mask = torch.cat(mask_list, dim=1).bool()
        sorted_mask, sort_indices = torch.sort(
            total_mask.int(),
            dim=1,
            descending=True,
            stable=True
        )
        gather_indices = sort_indices.unsqueeze(-1).expand(-1, -1, self.hidden_size)
        return sorted_mask.bool(), gather_indices

    def apply_packing(self, embedding_list, gather_indices, sorted_mask):
        raw_concat = torch.cat(embedding_list, dim=1)
        packed_embedding = torch.gather(raw_concat, dim=1, index=gather_indices)
        mask_broadcaster = sorted_mask.unsqueeze(-1).to(dtype=packed_embedding.dtype)
        packed_embedding = packed_embedding * mask_broadcaster
        return packed_embedding

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        duration: torch.Tensor = None,
        omni_emb_list: List[torch.Tensor] = None,
        omni_last_emb: torch.Tensor = None,
        omni_mask: torch.Tensor = None,
        guidance: torch.Tensor = None,
    ):
        assert self.use_omni_embedding or self.use_omni_last_embedding, \
            "At least one of use_omni_embedding / use_omni_last_embedding must be True"
        assert omni_mask is not None, "omni_mask is required"
        if self.use_omni_embedding:
            assert omni_emb_list is not None and len(omni_emb_list) >= self.total_blocks_depth, \
                f"omni_emb_list required. Expected len >= {self.total_blocks_depth}, got {len(omni_emb_list) if omni_emb_list else 0}"
        if self.use_omni_last_embedding:
            if omni_last_emb is None:
                assert omni_emb_list is not None and len(omni_emb_list) > 0
                omni_last_emb = omni_emb_list[-1]
        if omni_mask.dim() == 3:
            omni_mask = omni_mask.squeeze(1)

        # --- 1. Patching & Embeddings ---
        img = self.img_in(x)
        vec = self.time_in(t)
        repa_feat = None
        if self.guidance_embed and guidance is not None:
            vec = vec + self.guidance_in(guidance)

        # --- 2. RoPE (always continuous, no offset) ---
        grid_sizes_target = None
        if self.is_audio_type:
            freqs_cos, freqs_sin = get_audio_rotary_pos_embed(
                rope_dim_list=self.rope_dim_list,
                length=img.shape[1],
                theta=self.rope_theta,
                freqs_scaling=self.temporal_rope_scaling_factor,
                use_real=True,
            )
            freqs_cis = (freqs_cos.to(img.device), freqs_sin.to(img.device))
        else:
            _, _, ot, oh, ow = x.shape
            pt, ph, pw = self.patch_size
            grid_sizes_target = (ot // pt, oh // ph, ow // pw)
            freqs_cos, freqs_sin = self.get_rotary_pos_embed(grid_sizes_target)
            freqs_cis = (freqs_cos.to(img.device), freqs_sin.to(img.device))

        # --- 3. Build base text tokens ---
        txt_base = None
        txt_mask = None

        if self.use_omni_last_embedding:
            txt_base = self.omni_last_proj(omni_last_emb)
            txt_mask = omni_mask

        if self.duration_embedder is not None and duration is not None:
            dur_emb = self.duration_embedder(duration)
            dur_mask = torch.ones(omni_mask.shape[0], 1, device=omni_mask.device, dtype=omni_mask.dtype)
            if txt_base is not None:
                txt_base = torch.cat([dur_emb, txt_base], dim=1)
                txt_mask = torch.cat([dur_mask, txt_mask], dim=1)
            else:
                txt_base = dur_emb
                txt_mask = dur_mask

        has_base = txt_base is not None
        txt_base_len = txt_base.shape[1] if has_base else 0

        if self.use_omni_embedding and has_base:
            txt_mask_sorted, gather_indices = self.prepare_packed_indices([txt_mask, omni_mask])
        elif self.use_omni_embedding and not has_base:
            txt_mask_sorted = omni_mask
            gather_indices = None
        else:
            txt_mask_sorted = txt_mask
            gather_indices = None

        # --- 4. Double Stream Loop ---
        for i, block in enumerate(self.double_blocks):
            if self.use_omni_embedding:
                omni_curr = self.deep_fusion_projs[i](omni_emb_list[i])
                if has_base:
                    txt_packed = self.apply_packing([txt_base, omni_curr], gather_indices, txt_mask_sorted)
                else:
                    txt_packed = omni_curr
            else:
                txt_packed = txt_base

            if self.training and self.gradient_checkpointing:
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, freqs_cis=freqs_cis, text_mask=txt_mask_sorted, is_flash=True)
                    return custom_forward

                img, txt_out_packed = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    img, txt_packed, vec,
                    use_reentrant=False
                )
            else:
                img, txt_out_packed = block(
                    img=img, txt=txt_packed, vec=vec,
                    freqs_cis=freqs_cis, text_mask=txt_mask_sorted,
                    is_flash=True
                )

            if self.repa_proj is not None and i == self.repa_layer_num:
                repa_feat = self.repa_proj(img)

            if self.use_omni_embedding and has_base:
                B, Total_Len, D = txt_packed.shape
                restored_txt = torch.zeros(B, Total_Len, D, device=img.device, dtype=img.dtype)
                restored_txt.scatter_(dim=1, index=gather_indices, src=txt_out_packed)
                txt_base = restored_txt[:, :txt_base_len, :]
            elif not self.use_omni_embedding:
                txt_base = txt_out_packed

        # --- 5. Single Stream Loop ---
        img_len = img.shape[1]
        block_idx_offset = self.mm_double_blocks_depth

        for i, block in enumerate(self.single_blocks):
            if self.use_omni_embedding:
                omni_curr = self.deep_fusion_projs[block_idx_offset + i](omni_emb_list[block_idx_offset + i])
                if has_base:
                    txt_packed = self.apply_packing([txt_base, omni_curr], gather_indices, txt_mask_sorted)
                else:
                    txt_packed = omni_curr
            else:
                txt_packed = txt_base

            x_in = torch.cat([img, txt_packed], dim=1)
            txt_len_packed = txt_packed.shape[1]

            if self.training and self.gradient_checkpointing:
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, txt_len=txt_len_packed, freqs_cis=freqs_cis, text_mask=txt_mask_sorted, is_flash=True)
                    return custom_forward

                x_out = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    x_in, vec,
                    use_reentrant=False
                )
            else:
                x_out = block(
                    x=x_in, vec=vec, txt_len=txt_len_packed,
                    freqs_cis=freqs_cis, text_mask=txt_mask_sorted,
                    is_flash=True
                )

            img = x_out[:, :img_len, :]

            if self.repa_proj is not None and i + block_idx_offset == self.repa_layer_num:
                repa_feat = self.repa_proj(img)

            txt_out_packed = x_out[:, img_len:, :]

            if self.use_omni_embedding and has_base:
                B, Total_Len, D = txt_packed.shape
                restored_txt = torch.zeros(B, Total_Len, D, device=img.device, dtype=img.dtype)
                restored_txt.scatter_(dim=1, index=gather_indices, src=txt_out_packed)
                txt_base = restored_txt[:, :txt_base_len, :]
            elif not self.use_omni_embedding:
                txt_base = txt_out_packed

        # --- 6. Final Layer ---
        img_out = self.final_layer(img, vec)
        out = self.unpatchify(img_out, shape_info=None if self.is_audio_type else grid_sizes_target)

        return out, 0, repa_feat

    def unpatchify(self, x, shape_info):
        if self.is_audio_type:
            return x.transpose(1, 2).contiguous()
        else:
            t, h, w = shape_info
            c = self.out_channels
            pt, ph, pw = self.patch_size

            x = x.reshape(x.shape[0], t, h, w, c, pt, ph, pw)
            x = torch.einsum("nthwcopq->nctohpwq", x)
            x = x.reshape(x.shape[0], c, t * pt, h * ph, w * pw)
            return x
