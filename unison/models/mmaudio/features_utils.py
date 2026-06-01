from typing import Literal, Optional

import torch
import torch.nn as nn

from .ext.autoencoder import AutoEncoderModule
from .ext.autoencoder.distributions import DiagonalGaussianDistribution
from .ext.mel_converter import get_mel_converter


class FeaturesUtils(nn.Module):

    def __init__(
        self,
        *,
        tod_vae_ckpt: str, 
        bigvgan_vocoder_ckpt: Optional[str] = None,
        mode=Literal['16k', '44k'],
        need_vae_encoder: bool = True,
    ):
        super().__init__()

        self.mel_converter = get_mel_converter(mode)
        self.tod = AutoEncoderModule(vae_ckpt_path=tod_vae_ckpt,
                                        vocoder_ckpt_path=bigvgan_vocoder_ckpt,
                                        mode=mode,
                                        need_vae_encoder=need_vae_encoder)

    def compile(self):

        self.decode = torch.compile(self.decode)
        self.vocode = torch.compile(self.vocode)

    def train(self, mode: bool) -> None:
        return super().train(False)

    @torch.inference_mode()
    def encode_audio(self, x) -> DiagonalGaussianDistribution:
        assert self.tod is not None, 'VAE is not loaded'
        # x: (B * L)
        mel = self.mel_converter(x)
        dist = self.tod.encode(mel)

        return dist

    @torch.inference_mode()
    def vocode(self, mel: torch.Tensor) -> torch.Tensor:
        assert self.tod is not None, 'VAE is not loaded'
        return self.tod.vocode(mel)

    @torch.inference_mode()
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        assert self.tod is not None, 'VAE is not loaded'
        return self.tod.decode(z)

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    @torch.no_grad()
    def wrapped_decode(self, z):
        with torch.amp.autocast('cuda', dtype=self.dtype):
            mel_decoded = self.decode(z)
            audio = self.vocode(mel_decoded)

            return audio 

    @torch.no_grad()
    def wrapped_encode(self, audio):
        with torch.amp.autocast('cuda', dtype=self.dtype):
            dist = self.encode_audio(audio)

            return dist.mean