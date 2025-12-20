# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved\n

import os
import math
from abc import ABCMeta, abstractmethod
from typing import Union

import dacvae
import torch
import sys

from sam_audio.model.config import DACVAEConfig


_DACVAE_WN_PATCHED = False


def _patch_dacvae_weight_norm() -> None:
    global _DACVAE_WN_PATCHED
    if _DACVAE_WN_PATCHED:
        return

    try:
        from torch.nn.utils.parametrizations import weight_norm as _weight_norm
    except Exception:
        _DACVAE_WN_PATCHED = True
        return

    try:
        import dacvae.nn.layers as _dac_layers

        _dac_layers.weight_norm = _weight_norm
    except Exception:
        pass

    _dac_discriminator = sys.modules.get("dacvae.model.discriminator")
    if _dac_discriminator is not None:
        try:
            _dac_discriminator.weight_norm = _weight_norm
        except Exception:
            pass

    try:
        from torch.nn.utils import parametrize as _parametrize
        from torch.nn.utils.weight_norm import (
            remove_weight_norm as _remove_weight_norm_legacy,
        )

        def _remove_weight_norm_compat(module, name: str = "weight"):
            if _parametrize.is_parametrized(module, name):
                _parametrize.remove_parametrizations(
                    module, name, leave_parametrized=True
                )
                return module
            return _remove_weight_norm_legacy(module, name=name)

        torch.nn.utils.remove_weight_norm = _remove_weight_norm_compat
    except Exception:
        pass

    _DACVAE_WN_PATCHED = True


class Encoder(torch.nn.Module, metaclass=ABCMeta):
    @abstractmethod
    def forward(self, waveform: torch.Tensor) -> torch.Tensor: ...


class Codec(Encoder):
    @abstractmethod
    def decode(self, encoded_frames: torch.Tensor) -> torch.Tensor: ...

    @abstractmethod
    def wav_idx_to_feature_idx(
        self, wav_idx: Union[torch.Tensor, int], sample_rate=None
    ) -> Union[torch.Tensor, int]: ...

    @abstractmethod
    def feature_idx_to_wav_idx(
        self, feature_idx: Union[torch.Tensor, int], sample_rate=None
    ) -> Union[torch.Tensor, int]: ...

    @staticmethod
    def cast_to_int(
        x: Union[int, torch.Tensor],
    ) -> Union[int, torch.Tensor]:
        if isinstance(x, torch.Tensor):
            return x.int()
        else:
            return int(x)


class DACVAEEncoder(Encoder):
    def __init__(self, config: DACVAEConfig) -> None:
        super().__init__()
        _patch_dacvae_weight_norm()
        model = dacvae.DACVAE(
            encoder_dim=config.encoder_dim,
            encoder_rates=config.encoder_rates,
            latent_dim=config.latent_dim,
            decoder_dim=config.decoder_dim,
            decoder_rates=config.decoder_rates,
            n_codebooks=config.n_codebooks,
            codebook_size=config.codebook_size,
            codebook_dim=config.codebook_dim,
            quantizer_dropout=config.quantizer_dropout,
            sample_rate=config.sample_rate,
        ).eval()
        self._setup_model(model)
        self.hop_length = config.hop_length
        self.sample_rate = config.sample_rate

    def _setup_model(self, model):
        self.encoder = model.encoder
        self.quantizer = model.quantizer

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        disable_cudnn = os.environ.get("SAM_AUDIO_DISABLE_CUDNN") == "1"
        with torch.no_grad(), torch.backends.cudnn.flags(enabled=not disable_cudnn):
            z = self.encoder(self._pad(waveform))
            mean, _ = self.quantizer.in_proj(z).chunk(2, dim=1)
            encoded_frames = mean
        return encoded_frames

    def _pad(self, wavs):
        length = wavs.size(-1)
        if length % self.hop_length:
            p1d = (0, self.hop_length - (length % self.hop_length))
            return torch.nn.functional.pad(wavs, p1d, "reflect")
        else:
            return wavs


class DACVAE(DACVAEEncoder, Codec):
    def _setup_model(self, model):
        super()._setup_model(model)
        self.decoder = model.decoder

    def decode(self, encoded_frames: torch.Tensor) -> torch.Tensor:
        disable_cudnn = os.environ.get("SAM_AUDIO_DISABLE_CUDNN") == "1"
        with torch.backends.cudnn.flags(enabled=not disable_cudnn):
            emb = self.quantizer.out_proj(encoded_frames)
            return self.decoder(emb)

    def feature_idx_to_wav_idx(self, feature_idx, sample_rate=None):
        if sample_rate is None:
            sample_rate = self.sample_rate
        orig_freq = sample_rate
        new_freq = self.sample_rate
        wav_chunklen = feature_idx * self.hop_length * (orig_freq / new_freq)
        return self.cast_to_int(wav_chunklen)

    def wav_idx_to_feature_idx(self, wav_idx, sample_rate=None):
        ceil = math.ceil
        if torch.is_tensor(wav_idx):
            ceil = torch.ceil
        if sample_rate is None:
            sample_rate = self.sample_rate
        orig_freq = sample_rate
        new_freq = self.sample_rate
        target_length = ceil(new_freq * wav_idx / orig_freq)
        res = ceil(target_length / self.hop_length)
        return self.cast_to_int(res)
