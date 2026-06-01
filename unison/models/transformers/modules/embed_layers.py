# Licensed under the TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5/blob/main/LICENSE
#
# Unless and only to the extent required by applicable law, the Tencent Hunyuan works and any
# output and results therefrom are provided "AS IS" without any express or implied warranties of
# any kind including any warranties of title, merchantability, noninfringement, course of dealing,
# usage of trade, or fitness for a particular purpose. You are solely responsible for determining the
# appropriateness of using, reproducing, modifying, performing, displaying or distributing any of
# the Tencent Hunyuan works or outputs and assume any and all risks associated with your or a
# third party's use or distribution of any of the Tencent Hunyuan works or outputs and your exercise
# of rights and permissions under this agreement.
# See the License for the specific language governing permissions and limitations under the License.

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Union
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models import ModelMixin
from math import pi
from unison.commons import to_2tuple, to_3tuple


class ChannelLastConv1d(nn.Module):
    """
    Conv1d that consumes [B, L, C] and returns [B, L, C].
    Used for Audio branch.
    """
    def __init__(self, in_channels, out_channels, kernel_size, padding=0, stride=1, bias=True):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size=kernel_size,
            padding=padding, stride=stride, bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input: [B, L, C]
        x = x.transpose(1, 2)  # [B, L, C] -> [B, C, L]
        x = self.conv(x)
        x = x.transpose(1, 2)  # [B, C, L] -> [B, L, C]
        return x

class ConvMLP(nn.Module):
    """
    SwiGLU style ConvMLP used by MMAudio/Ovi style embedding.
    """
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int = 256,
        kernel_size: int = 3,
        padding: int = 1,
    ):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = ChannelLastConv1d(dim, hidden_dim, kernel_size=kernel_size, padding=padding, bias=False)
        self.w2 = ChannelLastConv1d(hidden_dim, dim, kernel_size=kernel_size, padding=padding, bias=False)
        self.w3 = ChannelLastConv1d(dim, hidden_dim, kernel_size=kernel_size, padding=padding, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

class UniversalPatchEmbed(ModelMixin, ConfigMixin):
    """
    Universal Patch Embedding for HunyuanVideo 1.5 Architecture.
    Supports both Video (3D) and Audio (1D w/ Context).

    Audio patch embed types (audio_patch_type):
      "conv_mlp"  — Original: Conv1d + SwiGLU ConvMLP (~132M params, heavy)
      "mlp"       — Pointwise MLP: Linear-SiLU-Linear (~2.4M params)
      "conv_lite" — Small Conv + Linear: Conv1d(k=3)-SiLU-Linear (~2.5M params)
    """

    @register_to_config
    def __init__(
        self,
        patch_size=(1, 2, 2),
        in_chans=48,
        embed_dim=1280,
        is_reshape_temporal_channels=False,
        concat_condition=False,
        norm_layer=None,
        flatten=True,
        bias=True,
        is_audio=False,
        audio_kernel_size=7,
        audio_padding=3,
        audio_patch_type="conv_mlp",
        dtype=None,
        device=None,
    ):
        factory_kwargs = {"dtype": dtype, "device": device}
        super().__init__()

        self.is_audio = is_audio
        self.flatten = flatten
        self.audio_patch_type = audio_patch_type
        self.patch_size = to_3tuple(patch_size) if not is_audio else patch_size

        orig_in_chans = in_chans
        if concat_condition:
            if is_reshape_temporal_channels:
                in_chans = in_chans + in_chans//2 + 1
            else:
                in_chans = in_chans * 2 + 1

        self.in_chans = in_chans
        self.orig_in_chans = orig_in_chans

        if is_audio:
            if audio_patch_type == "conv_mlp":
                self.proj = nn.Sequential(
                    ChannelLastConv1d(in_chans, embed_dim, kernel_size=audio_kernel_size, padding=audio_padding, bias=bias),
                    nn.SiLU(),
                    ConvMLP(embed_dim, embed_dim * 4, kernel_size=audio_kernel_size, padding=audio_padding),
                )
                first_layer = self.proj[0].conv
            elif audio_patch_type == "mlp":
                self.proj = nn.Sequential(
                    nn.Linear(in_chans, embed_dim, bias=bias),
                    nn.SiLU(),
                    nn.Linear(embed_dim, embed_dim, bias=bias),
                )
                first_layer = self.proj[0]
            elif audio_patch_type == "conv_lite":
                self.proj = nn.Sequential(
                    ChannelLastConv1d(in_chans, embed_dim, kernel_size=3, padding=1, bias=bias),
                    nn.SiLU(),
                    nn.Linear(embed_dim, embed_dim, bias=bias),
                )
                first_layer = self.proj[0].conv
            else:
                raise ValueError(f"Unknown audio_patch_type: {audio_patch_type}")

            self._init_audio_proj(first_layer, orig_in_chans, concat_condition, bias)

        else:
            self.patch_size = to_3tuple(patch_size)
            self.proj = nn.Conv3d(
                in_chans,
                embed_dim,
                kernel_size=self.patch_size,
                stride=self.patch_size,
                bias=bias,
                **factory_kwargs,
            )

            nn.init.xavier_uniform_(self.proj.weight[:, :orig_in_chans].view(self.proj.weight[:, :orig_in_chans].size(0), -1))
            if concat_condition:
                nn.init.zeros_(self.proj.weight[:, orig_in_chans:].view(self.proj.weight[:, orig_in_chans:].size(0), -1))
            if bias:
                nn.init.zeros_(self.proj.bias)

        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    @staticmethod
    def _init_audio_proj(first_layer, orig_in_chans, concat_condition, bias):
        """Xavier init + zero-init concat channels for any audio proj first layer."""
        if isinstance(first_layer, nn.Conv1d):
            nn.init.xavier_uniform_(first_layer.weight)
            if bias and first_layer.bias is not None:
                nn.init.zeros_(first_layer.bias)
            if concat_condition:
                with torch.no_grad():
                    first_layer.weight[:, orig_in_chans:, :] = 0.0
        elif isinstance(first_layer, nn.Linear):
            nn.init.xavier_uniform_(first_layer.weight)
            if bias and first_layer.bias is not None:
                nn.init.zeros_(first_layer.bias)
            if concat_condition:
                with torch.no_grad():
                    first_layer.weight[:, orig_in_chans:] = 0.0

    def forward(self, x):
        if self.is_audio:
            if x.dim() == 5:
                x = x.squeeze(2).squeeze(2)
            elif x.dim() == 4:
                x = x.squeeze(2)

            # [B, C, L] -> [B, L, C]
            x = x.transpose(1, 2)
            x = self.proj(x)

        else:
            x = self.proj(x)
            if self.flatten:
                x = x.flatten(2).transpose(1, 2)

        x = self.norm(x)
        return x

class PatchEmbed(nn.Module):
    """2D Image to Patch Embedding

    Image to Patch Embedding using Conv2d

    A convolution based approach to patchifying a 2D image w/ embedding projection.

    Based on the impl in https://github.com/google-research/vision_transformer

    Hacked together by / Copyright 2020 Ross Wightman

    Remove the _assert function in forward function to be compatible with multi-resolution images.
    """

    def __init__(
        self,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        is_reshape_temporal_channels=False,
        concat_condition=True,
        norm_layer=None,
        flatten=True,
        bias=True,
        dtype=None,
        device=None,
    ):
        factory_kwargs = {"dtype": dtype, "device": device}
        super().__init__()
        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size
        self.flatten = flatten

        # Only support concat mode (multitask mask training)
        orig_in_chans = in_chans
        if concat_condition:
            if is_reshape_temporal_channels:
                in_chans = in_chans + in_chans//2 + 1
            else:
                in_chans = in_chans * 2 + 1

        self.proj = nn.Conv3d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            bias=bias,
            **factory_kwargs,
        )

        nn.init.xavier_uniform_(self.proj.weight[:, :orig_in_chans].view(self.proj.weight[:, :orig_in_chans].size(0), -1))
        # Special initialization for concat mode
        nn.init.zeros_(self.proj.weight[:, orig_in_chans:].view(self.proj.weight[:, orig_in_chans:].size(0), -1))

        if bias:
            nn.init.zeros_(self.proj.bias)

        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x


class TextProjection(nn.Module):
    """
    Projects text embeddings. Also handles dropout for classifier-free guidance.

    Adapted from https://github.com/PixArt-alpha/PixArt-alpha/blob/master/diffusion/model/nets/PixArt_blocks.py
    """

    def __init__(self, in_channels, hidden_size, act_layer, dtype=None, device=None):
        factory_kwargs = {"dtype": dtype, "device": device}
        super().__init__()
        self.linear_1 = nn.Linear(
            in_features=in_channels,
            out_features=hidden_size,
            bias=True,
            **factory_kwargs
        )
        self.act_1 = act_layer()
        self.linear_2 = nn.Linear(
            in_features=hidden_size,
            out_features=hidden_size,
            bias=True,
            **factory_kwargs
        )

    def forward(self, caption):
        hidden_states = self.linear_1(caption)
        hidden_states = self.act_1(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        return hidden_states



class VisionProjection(torch.nn.Module):

    def __init__(self, input_dim, output_dim):
        super().__init__()

        self.proj = torch.nn.Sequential(
            torch.nn.LayerNorm(input_dim), 
            torch.nn.Linear(input_dim, input_dim),
            torch.nn.GELU(), 
            torch.nn.Linear(input_dim, output_dim),
            torch.nn.LayerNorm(output_dim)
        )
        

    def forward(self, vision_embeds):
        return self.proj(vision_embeds)

class ClipVisionProjection(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Linear(in_channels, out_channels * 3)
        self.down = nn.Linear(out_channels * 3, out_channels)
        torch.nn.init.zeros_(self.down.weight)
        torch.nn.init.zeros_(self.down.bias)

    def forward(self, x):
        projected_x = self.down(nn.functional.silu(self.up(x)))
        return projected_x
    
def timestep_embedding(t, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.

    Args:
        t (torch.Tensor): a 1-D Tensor of N indices, one per batch element. These may be fractional.
        dim (int): the dimension of the output.
        max_period (int): controls the minimum frequency of the embeddings.

    Returns:
        embedding (torch.Tensor): An (N, D) Tensor of positional embeddings.

    .. ref_link: https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(start=0, end=half, dtype=torch.float32)
        / half
    ).to(device=t.device)
    args = t[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(
        self,
        hidden_size,
        act_layer,
        frequency_embedding_size=256,
        max_period=10000,
        out_size=None,
        dtype=None,
        device=None,
    ):
        factory_kwargs = {"dtype": dtype, "device": device}
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        self.max_period = max_period
        if out_size is None:
            out_size = hidden_size

        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True, **factory_kwargs),
            act_layer(),
            nn.Linear(hidden_size, out_size, bias=True, **factory_kwargs),
        )
        nn.init.normal_(self.mlp[0].weight, std=0.02)
        nn.init.normal_(self.mlp[2].weight, std=0.02)

    def forward(self, t):
        t_freq = timestep_embedding(
            t, self.frequency_embedding_size, self.max_period
        ).type(self.mlp[0].weight.dtype)
        t_emb = self.mlp(t_freq)
        return t_emb

class StableAudioPositionalEmbedding(nn.Module):
    """Used for continuous time
    Adapted from Stable Audio Open.
    """

    def __init__(self, dim: int):
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim))

    def forward(self, times: torch.Tensor) -> torch.Tensor:
        times = times[..., None]
        freqs = times * self.weights[None] * 2 * pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim=-1)
        fouriered = torch.cat((times, fouriered), dim=-1)
        return fouriered

class DurationEmbedder(nn.Module):
    """
    A simple linear projection model to map numbers to a latent space.

    Code is adapted from
    https://github.com/Stability-AI/stable-audio-tools

    Args:
        number_embedding_dim (`int`):
            Dimensionality of the number embeddings.
        min_value (`int`):
            The minimum value of the seconds number conditioning modules.
        max_value (`int`):
            The maximum value of the seconds number conditioning modules
        internal_dim (`int`):
            Dimensionality of the intermediate number hidden states.
    """

    def __init__(
        self,
        number_embedding_dim,
        min_value,
        max_value,
        internal_dim= 256,
    ):
        super().__init__()
        self.time_positional_embedding = nn.Sequential(
            StableAudioPositionalEmbedding(internal_dim),
            nn.Linear(in_features=internal_dim + 1, out_features=number_embedding_dim),
        )

        self.number_embedding_dim = number_embedding_dim
        self.min_value = min_value
        self.max_value = max_value
        self.dtype = torch.float32

    def forward(
        self,
        floats: torch.Tensor,
    ):
        floats = floats.clamp(self.min_value, self.max_value)

        normalized_floats = (floats - self.min_value) / (
            self.max_value - self.min_value
        )

        # Cast floats to same type as embedder
        embedder_dtype = next(self.time_positional_embedding.parameters()).dtype
        normalized_floats = normalized_floats.to(embedder_dtype)

        embedding = self.time_positional_embedding(normalized_floats)
        float_embeds = embedding.view(-1, 1, self.number_embedding_dim)

        return float_embeds