# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved\n

from typing import Optional

import torch
import torchaudio

from sam_audio.ranking.clap import get_model
from sam_audio.runtime import auto_tune


class CLAP(torch.nn.Module):
    def __init__(
        self,
        checkpoint: Optional[str] = None,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.device = auto_tune(device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        ))
        self.model = get_model(checkpoint_file=checkpoint, device=self.device)

    def __call__(
        self,
        target_wavs: list[torch.Tensor],
        descriptions: list[str],
        target_wavs_sample_rate: int = 48_000,
        **kwargs,
    ) -> list[dict[str, float]]:
        with torch.inference_mode():
            audio_data = []
            for wav in target_wavs:
                wav_cpu = wav
                if wav_cpu.is_cuda:
                    wav_cpu = wav_cpu.cpu()
                wav_cpu = wav_cpu.float()
                if wav_cpu.ndim > 1:
                    wav_cpu = wav_cpu.mean(dim=0)

                if target_wavs_sample_rate != 48_000:
                    wav_cpu = torchaudio.functional.resample(
                        wav_cpu.unsqueeze(0),
                        target_wavs_sample_rate,
                        48_000,
                    ).squeeze(0)

                wav_cpu = torch.clamp(wav_cpu, min=-1.0, max=1.0)
                wav_cpu = (wav_cpu * 32767.0).to(torch.int16)
                wav_cpu = wav_cpu.to(torch.float32) / 32767.0
                audio_data.append(wav_cpu)

            audio_embs = self.model.get_audio_embedding_from_data(
                audio_data, use_tensor=True
            )
            text_embs = self.model.get_text_embedding(
                descriptions, use_tensor=True)
            sims = audio_embs.unsqueeze(1) @ text_embs.unsqueeze(2)
            return {"CLAPSimilarity": sims.cpu()[:, 0, 0].tolist()}
