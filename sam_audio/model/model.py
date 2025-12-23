# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

import math
import re
from dataclasses import dataclass
from typing import Any, Dict, Optional, Literal

import torch
from torchdiffeq import odeint

from sam_audio.runtime import (
    async_init,
    autocast_context,
    clear_cuda_cache,
    compile_model,
    get_ode_options,
    should_compile,
    wrap_with_cuda_graph,
)
from sam_audio.model.align import AlignModalities
from sam_audio.model.base import BaseModel
from sam_audio.model.codec import DACVAE
from sam_audio.model.config import SAMAudioConfig
from sam_audio.model.text_encoder import T5TextEncoder
from sam_audio.model.transformer import DiT
from sam_audio.model.vision_encoder import PerceptionEncoder
from sam_audio.processor import Batch
from sam_audio.ranking import create_ranker
from sam_audio.metrics import measure_time, log_memory 

# ODE preset names
ODEPreset = Literal["fast", "balanced", "quality", "max_quality"]


class SinusoidalEmbedding(torch.nn.Module):
    def __init__(self, dim, theta=10000):
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        inv_freq = torch.exp(
            -math.log(theta) * torch.arange(half_dim).float() / half_dim
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x, pos=None):
        if pos is None:
            seq_len, device = x.shape[1], x.device
            pos = torch.arange(seq_len, device=device)

        emb = torch.einsum("i, j -> i j", pos, self.inv_freq)
        emb = torch.cat((emb.cos(), emb.sin()), dim=-1)
        return emb


class EmbedAnchors(torch.nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, out_dim: int):
        super().__init__()
        self.embed = torch.nn.Embedding(
            num_embeddings + 1, embedding_dim, padding_idx=num_embeddings
        )
        self.gate = torch.nn.Parameter(torch.tensor([0.0]))
        self.proj = torch.nn.Linear(embedding_dim, out_dim, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        anchor_ids: Optional[torch.Tensor] = None,
        anchor_alignment: Optional[torch.Tensor] = None,
    ):
        if anchor_ids is None:
            return x

        embs = self.embed(anchor_ids.gather(1, anchor_alignment))
        proj = self.proj(embs)
        return x + self.gate.tanh() * proj


@dataclass
class SeparationResult:
    target: torch.Tensor
    residual: torch.Tensor
    noise: torch.Tensor


class SAMAudio(BaseModel):
    config_cls = SAMAudioConfig
    revision = None

    def __init__(self, cfg: SAMAudioConfig):
        super().__init__()
        # Start initializing heavy components in background threads
        future_codec = async_init(DACVAE, cfg.audio_codec)
        future_text = async_init(T5TextEncoder, cfg.text_encoder)

        self._vision_encoder_cfg = cfg.vision_encoder
        self._vision_dim = cfg.vision_encoder.dim
        self.vision_encoder = None
        
        # Initialize large components on meta device to save time/memory
        with torch.device("meta"):
            self.transformer = DiT(cfg.transformer)
            self.proj = torch.nn.Linear(cfg.in_channels, cfg.transformer.dim)
            self.align_masked_video = AlignModalities(
                cfg.vision_encoder.dim, cfg.transformer.dim
            )
            self.embed_anchors = EmbedAnchors(
                cfg.num_anchors, cfg.anchor_embedding_dim, cfg.transformer.dim
            )
            self.memory_proj = torch.nn.Linear(
                cfg.text_encoder.dim, cfg.transformer.dim)

        # Manually fix RoPE buffers to be on CPU (since they are non-persistent and were on meta)
        if self.transformer.rope_embeddings is not None:
             self.transformer.rope_embeddings.reset_parameters()
             
        # Manually fix TimestepEmbedder buffer
        if self.transformer.t_embedder is not None:
             t_emb = self.transformer.t_embedder
             half = t_emb.frequency_embedding_size // 2
             freqs = torch.exp(
                -math.log(10000)
                * torch.arange(start=0, end=half, dtype=torch.float32)
                / half
             )
             t_emb.register_buffer("freqs", freqs, persistent=False)

        self.timestep_emb = SinusoidalEmbedding(cfg.transformer.dim)
        self._visual_ranker_cfg = cfg.visual_ranker
        self._text_ranker_cfg = cfg.text_ranker
        self.visual_ranker = None
        self.text_ranker = None
        self._span_predictor_cfg = cfg.span_predictor
        self.span_predictor = None
        self.span_predictor_transform = None
        
        # Wait for background initialization to complete
        self.audio_codec = future_codec.result()
        self.text_encoder = future_text.result()
        
        # Pre-allocated noise tensor pool for reduced allocation overhead
        # Keyed by (batch_size, seq_len, channels) tuple
        self._noise_pool: dict = {}

    def _get_vision_encoder(self):
        if self.vision_encoder is None:
            self.vision_encoder = PerceptionEncoder(self._vision_encoder_cfg)
        return self.vision_encoder

    def _get_span_predictor(self):
        if self._span_predictor_cfg is None:
            return None
        if self.span_predictor is None:
            # Lazy import to avoid OOM at module load time
            from core.audio_visual_encoder import PEAudioFrame, PEAudioFrameTransform
            self.span_predictor = PEAudioFrame.from_config(
                self._span_predictor_cfg, pretrained=True
            )
            self.span_predictor_transform = PEAudioFrameTransform.from_config(
                self._span_predictor_cfg
            )
            # Move span predictor to the same device as the model
            device = next(self.parameters()).device
            self.span_predictor = self.span_predictor.to(device)
        return self.span_predictor

    def _get_visual_ranker(self):
        if self.visual_ranker is None and self._visual_ranker_cfg is not None:
            self.visual_ranker = create_ranker(self._visual_ranker_cfg)
            # Move ranker to the same device as the model
            device = next(self.parameters()).device
            self.visual_ranker = self.visual_ranker.to(device)
        return self.visual_ranker

    def _get_text_ranker(self):
        if self.text_ranker is None and self._text_ranker_cfg is not None:
            self.text_ranker = create_ranker(self._text_ranker_cfg)
            # Move ranker to the same device as the model
            device = next(self.parameters()).device
            self.text_ranker = self.text_ranker.to(device)
        return self.text_ranker

    @property
    def sample_rate(self):
        return self.audio_codec.sample_rate

    def _get_noise(self, like: torch.Tensor, force_new: bool = False) -> torch.Tensor:
        """
        Get or create a noise tensor from the pool.
        
        This reduces memory allocation overhead by reusing pre-allocated tensors
        when possible. The tensor is re-randomized to ensure fresh noise.
        
        Args:
            like: Template tensor to match shape, dtype, and device.
            force_new: If True, always allocate a new tensor.
        
        Returns:
            Random noise tensor with same shape/dtype/device as `like`.
        """
        if force_new:
            return torch.randn_like(like)
        
        shape_key = (like.shape, like.dtype, like.device)
        
        if shape_key in self._noise_pool:
            # Reuse existing tensor, but re-randomize it
            noise = self._noise_pool[shape_key]
            noise.normal_()  # In-place randomization
            return noise
        
        # Create new tensor and cache it
        # Limit pool size to prevent memory bloat (keep last 4 shapes)
        if len(self._noise_pool) >= 4:
            oldest_key = next(iter(self._noise_pool))
            del self._noise_pool[oldest_key]
        
        noise = torch.randn_like(like)
        self._noise_pool[shape_key] = noise
        return noise

    def warmup(
        self,
        batch_size: int = 1,
        seq_len: int = 256,
        description: str = "warmup",
    ) -> None:
        """
        Warmup the model by running a dummy inference to compile CUDA kernels.
        
        This pre-compiles JIT code, torch.compile graphs, and triggers cuDNN
        autotuning, reducing latency on the first real inference.
        
        Args:
            batch_size: Batch size for warmup inference.
            seq_len: Sequence length (audio frames) for warmup.
            description: Dummy text description.
        """
        import logging
        logger = logging.getLogger(__name__)
        
        device = self.device()
        dtype = next(self.parameters()).dtype
        
        logger.info("Running ODE solver warmup to compile kernels...")
        
        try:
            # Create minimal dummy inputs
            # seq_len corresponds to audio codec frames, not raw samples
            in_channels = self.proj.in_features
            
            # proj expects [noisy_audio, zeros_like(audio_features), audio_features]
            # where noisy_audio and audio_features have the same shape
            # So feature_dim = in_channels // 3, and both have shape [B, T, feature_dim]
            feature_dim = in_channels // 3
            
            # Create dummy noisy_audio input (same shape as audio_features)
            dummy_noisy = torch.randn(
                batch_size, seq_len, feature_dim,
                device=device, dtype=dtype
            )
            
            # Create dummy audio features (same shape as noisy_audio, but in real code
            # audio_features is stacked [features, features] so with dim=feature_dim)
            dummy_features = torch.randn(
                batch_size, seq_len, feature_dim,
                device=device, dtype=dtype
            )
            
            # Create dummy text features
            text_dim = self.memory_proj.in_features
            dummy_text = torch.randn(batch_size, 16, text_dim, device=device, dtype=dtype)
            text_mask = torch.zeros(batch_size, 16, dtype=torch.bool, device=device)
            
            # Create dummy video features
            vision_dim = self._vision_dim
            dummy_video = torch.zeros(batch_size, vision_dim, seq_len, device=device, dtype=dtype)
            
            # Run a few forward passes to warmup
            dummy_time = torch.tensor([0.5], device=device, dtype=dtype)
            
            for _ in range(2):  # 2 warmup passes
                with autocast_context(device=device):
                    aligned = self.align_inputs(
                        dummy_noisy,
                        dummy_features,
                        masked_video_features=dummy_video,
                    )
                    _ = self.transformer(
                        aligned,
                        dummy_time.expand(batch_size),
                        padding_mask=None,
                        memory=self.memory_proj(dummy_text) + self.timestep_emb(
                            dummy_time, pos=dummy_time
                        ).unsqueeze(1),
                        memory_padding_mask=text_mask,
                    )
            
            # Synchronize to ensure all kernels are compiled
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            
            logger.info("ODE solver warmup complete")
            
        except Exception as e:
            logger.warning("Warmup failed (non-fatal): %s", e)

    def align_inputs(
        self,
        noisy_audio,
        audio_features: torch.Tensor,
        masked_video_features: Optional[torch.Tensor] = None,
        anchor_ids: Optional[torch.Tensor] = None,
        anchor_alignment: Optional[torch.Tensor] = None,
    ):
        x = torch.cat(
            [
                noisy_audio,
                torch.zeros_like(audio_features),
                audio_features,
            ],
            dim=2,
        )

        projected = self.proj(x)
        aligned = self.align_masked_video(projected, masked_video_features)
        aligned = self.embed_anchors(aligned, anchor_ids, anchor_alignment)
        return aligned

    def forward(
        self,
        noisy_audio: torch.Tensor,
        audio_features: torch.Tensor,
        text_features: torch.Tensor,
        time: torch.Tensor,
        masked_video_features: Optional[torch.Tensor] = None,
        text_mask: Optional[torch.Tensor] = None,
        anchor_ids: Optional[torch.Tensor] = None,
        anchor_alignment: Optional[torch.Tensor] = None,
        audio_pad_mask: Optional[torch.Tensor] = None,
    ):
        """
        Forward pass for the model.  Represents one function evaluation of the ODE.
        In the below descriptions, B is batch size, T is sequence length, C is channel size.
        Note that the size of C and T may vary across arguments (ex. text_features vs. audio_features),
        it is used only to designate a Channel or time/sequence-length dimension respectively.

        Args:
            noisy_audio (torch.Tensor): Noisy audio input tensor (being denoised).
            audio_features (torch.Tensor): Clean audio features [B x T x C].
            text_features (torch.Tensor): Encoded text features tensor [B x T x C].
            time (torch.Tensor): Timestep tensor for positional encoding [B].
            masked_video_features (Optional[torch.Tensor], optional): Masked video features tensor. [B x C x T].
            text_mask (Optional[torch.Tensor], optional): Padding mask for text features. [B x T].
            anchor_ids (Optional[torch.Tensor], optional): Anchor IDs tensor. Defaults to None [B x T].
            anchor_alignment (Optional[torch.Tensor], optional): Anchor alignment tensor. B x T.
            audio_pad_mask (Optional[torch.Tensor], optional): Padding mask for audio input. [B x T].

        Returns:
            torch.Tensor
        """
        aligned_inputs = self.align_inputs(
            noisy_audio,
            audio_features,
            masked_video_features=masked_video_features,
            anchor_ids=anchor_ids,
            anchor_alignment=anchor_alignment,
        )

        memory = timestep_emb = self.timestep_emb(time, pos=time).unsqueeze(1)
        if text_features is not None:
            memory = self.memory_proj(text_features) + timestep_emb

        return self.transformer(
            aligned_inputs,
            time,
            padding_mask=audio_pad_mask,
            memory=memory,
            memory_padding_mask=text_mask,
        )

    def _get_audio_features(self, audios: torch.Tensor):
        audio_features = self.audio_codec(audios).transpose(1, 2)
        return torch.cat([audio_features, audio_features], dim=2)

    def _get_video_features(self, video, audio_features):
        B, T, _ = audio_features.shape
        if video is None:
            return audio_features.new_zeros(B, self._vision_dim, T)
        else:
            return self._get_vision_encoder()(video).transpose(1, 2)

    def _repeat_for_reranking(self, tensor, candidates):
        if candidates > 1:
            B = tensor.size(0)
            rest = tensor.shape[1:]
            return (
                tensor.unsqueeze(1)
                .expand(B, candidates, *rest)
                .reshape(B * candidates, *rest)
            )
        else:
            return tensor

    def _unrepeat_from_reranking(self, tensor, candidates):
        return tensor[::candidates]

    def _get_forward_args(self, batch: Batch, candidates: int = 1):
        audio_features = self._get_audio_features(batch.audios)
        text_features, text_mask = self.text_encoder(batch.descriptions)
        masked_video_features = self._get_video_features(
            batch.masked_video, audio_features
        )

        return {
            "audio_features": self._repeat_for_reranking(audio_features, candidates),
            "text_features": self._repeat_for_reranking(text_features, candidates),
            "text_mask": self._repeat_for_reranking(text_mask, candidates),
            "masked_video_features": self._repeat_for_reranking(
                masked_video_features, candidates
            ),
            "anchor_ids": self._repeat_for_reranking(batch.anchor_ids, candidates),
            "anchor_alignment": self._repeat_for_reranking(
                batch.anchor_alignment, candidates
            ),
            "audio_pad_mask": self._repeat_for_reranking(
                batch.audio_pad_mask, candidates
            ),
        }

    def predict_spans(
        self, batch: Batch, audio_features: torch.Tensor, audio_pad_mask: torch.Tensor
    ) -> Batch:
        with measure_time("span_prediction_time"):
            input = self.span_predictor_transform(text=batch.descriptions).to(
                audio_features.device
            )
            output = self.span_predictor(
                input_features=audio_features[:, :, :128],
                padding_mask=audio_pad_mask,
                return_spans=True,
                **input,
            )
            anchors = [[["+"] + anchor for anchor in anchors]
                       for anchors in output.spans]
            batch.process_anchors(anchors)
        return batch

    @torch.inference_mode()
    def separate(
        self,
        batch: Batch,
        noise: Optional[torch.Tensor] = None,
        ode_opt: Optional[Dict[str, Any]] = None,
        ode_preset: ODEPreset = "quality",
        reranking_candidates: int = 1,
        predict_spans: bool = False,
    ) -> SeparationResult:
        """
        Separate target audio from mixture.

        Args:
            batch: Processed input batch.
            noise: Optional initial noise tensor.
            ode_opt: ODE solver options dict. If None, uses ode_preset.
            ode_preset: ODE solver preset ('fast', 'balanced', 'quality', 'max_quality').
            reranking_candidates: Number of candidates for reranking.
            predict_spans: Whether to predict time spans for anchoring.

        Returns:
            SeparationResult with target, residual, and noise tensors.
        """
        # Resolve ODE options
        if ode_opt is None:
            ode_opt = get_ode_options(ode_preset)

        device = batch.audios.device
        
        # Log VRAM before starting major operations
        log_memory("inference_start_vram", device)
        
        # Encode audio/text/video (keep state tensors float32; AMP is applied in compute-heavy ops)
        with measure_time("encoding_time"):
            with autocast_context(device=device):
                audio_features = self.audio_codec(batch.audios).transpose(1, 2)
            audio_features = audio_features.float()
            audio_features = torch.cat([audio_features, audio_features], dim=2)
    
            text_features, text_mask = self.text_encoder(batch.descriptions)
            with autocast_context(device=device):
                masked_video_features = self._get_video_features(
                    batch.masked_video, audio_features
                )
            masked_video_features = masked_video_features.float()

        forward_args = {
            "audio_features": self._repeat_for_reranking(audio_features, reranking_candidates),
            "text_features": self._repeat_for_reranking(text_features, reranking_candidates),
            "text_mask": self._repeat_for_reranking(text_mask, reranking_candidates),
            "masked_video_features": self._repeat_for_reranking(
                masked_video_features, reranking_candidates
            ),
            "anchor_ids": self._repeat_for_reranking(batch.anchor_ids, reranking_candidates),
            "anchor_alignment": self._repeat_for_reranking(
                batch.anchor_alignment, reranking_candidates
            ),
            "audio_pad_mask": self._repeat_for_reranking(
                batch.audio_pad_mask, reranking_candidates
            ),
        }

        # Aggressive memory clearing after encoding (task-12)
        # Free intermediate tensors before compute-intensive ODE loop
        if device.type == "cuda":
            del audio_features, text_features, text_mask, masked_video_features
            clear_cuda_cache(log=False)

        if predict_spans and batch.anchors is None and self._span_predictor_cfg is not None:
            self._get_span_predictor()
            batch = self.predict_spans(
                batch=batch,
                audio_features=self._unrepeat_from_reranking(
                    forward_args["audio_features"], reranking_candidates
                ),
                audio_pad_mask=self._unrepeat_from_reranking(
                    forward_args["audio_pad_mask"], reranking_candidates
                ),
            )

        audio_features = forward_args["audio_features"]
        B, T, C = audio_features.shape
        C = C // 2  # we stack audio_features, so the actual channels is half

        if noise is None:
            noise = self._get_noise(audio_features)

        def vector_field(t, noisy_audio):
            with autocast_context(device=noisy_audio.device):
                res = self.forward(
                    noisy_audio=noisy_audio,
                    time=t.expand(noisy_audio.size(0)),
                    **forward_args,
                )
            return res

        # Wrap with CUDA graph for reduced kernel launch overhead
        # (only active when SAM_AUDIO_CUDA_GRAPHS=1 and SAM_AUDIO_COMPILE=0)
        vector_field = wrap_with_cuda_graph(vector_field, warmup_runs=3)

        log_memory("pre_ode_vram", device)

        with measure_time("ode_solver_time"):
            states = odeint(
                vector_field,
                noise,
                torch.tensor([0.0, 1.0], device=noise.device),
                **ode_opt,
            )

        # Clear ODE intermediates before decoding (task-12)
        # Save noise reference before deleting for return value
        noise_out = noise
        if device.type == "cuda":
            del forward_args, noise
            clear_cuda_cache(log=False)
            
        generated_features = states[-1].transpose(1, 2)
        # generated_features has shape [B, 2C, T].  Reshape to stack along the batch dimension
        with measure_time("decoding_time"):
            with autocast_context(device=generated_features.device):
                wavs = self.audio_codec.decode(
                    generated_features.reshape(2 * B, C, T)
                ).view(B, 2, -1)
            wavs = wavs.float()

        bsz = wavs.size(0) // reranking_candidates
        sizes = self.audio_codec.feature_idx_to_wav_idx(batch.sizes)
        target_wavs = self.unbatch(
            wavs[:, 0].view(bsz, reranking_candidates, -1), sizes
        )
        residual_wavs = self.unbatch(
            wavs[:, 1].view(bsz, reranking_candidates, -1), sizes
        )

        with measure_time("reranking_time"):
            if reranking_candidates > 1 and batch.masked_video is not None:
                visual_ranker = self._get_visual_ranker()
            else:
                visual_ranker = None
    
            if reranking_candidates > 1 and visual_ranker is not None:
                scores = visual_ranker(
                    extracted_audio=target_wavs,
                    videos=batch.masked_video,
                    sample_rate=self.audio_codec.sample_rate,
                )
                idxs = scores.argmax(dim=1)
            elif reranking_candidates > 1 and self._get_text_ranker() is not None:
                input_audio = [
                    audio[:, :size].expand(reranking_candidates, -1)
                    for audio, size in zip(batch.audios, sizes, strict=False)
                ]
                scores = self._get_text_ranker()(
                    extracted_audio=target_wavs,
                    input_audio=input_audio,
                    descriptions=batch.descriptions,
                    sample_rate=self.audio_codec.sample_rate,
                )
                idxs = scores.argmax(dim=1)
            else:
                idxs = torch.zeros(bsz, dtype=torch.long, device=device)

        log_memory("post_inference_vram", device)

        return SeparationResult(
            target=[wav[idx]
                    for wav, idx in zip(target_wavs, idxs, strict=False)],
            residual=[
                wavs[idx] for wavs, idx in zip(residual_wavs, idxs, strict=False)
            ],
            noise=noise_out,
        )

    def unbatch(self, wavs: torch.Tensor, sizes: torch.Tensor, time_dim: int = -1):
        result = []
        for row, size in zip(wavs, sizes, strict=False):
            result.append(row.narrow(dim=time_dim, start=0, length=size))
        return result

    @torch.inference_mode()
    def separate_fast(
        self,
        batch: Batch,
        noise: Optional[torch.Tensor] = None,
        reranking_candidates: int = 1,
        predict_spans: bool = False,
    ) -> SeparationResult:
        """
        Fast audio separation using reduced ODE steps.

        This is a convenience wrapper around separate() with the 'fast' preset.
        Trades some quality for ~2-4x faster inference.

        Args:
            batch: Processed input batch.
            noise: Optional initial noise tensor.
            reranking_candidates: Number of candidates for reranking.
            predict_spans: Whether to predict time spans for anchoring.

        Returns:
            SeparationResult with target, residual, and noise tensors.
        """
        return self.separate(
            batch=batch,
            noise=noise,
            ode_preset="fast",
            reranking_candidates=reranking_candidates,
            predict_spans=predict_spans,
        )

    def load_state_dict(self, state_dict, strict=True, assign=False):
        if strict:
            missing_keys, unexpected_keys = super().load_state_dict(
                state_dict, strict=False, assign=assign
            )
            # We load this directly from HF, not in checkpoint
            skip_regex = re.compile(
                "(^text_encoder|^vision_encoder|^visual_ranker|^text_ranker|^span_predictor)"
            )
            missing_keys = [
                x for x in missing_keys if not re.search(skip_regex, x)]
            unexpected_keys = [
                x for x in unexpected_keys if not re.search(skip_regex, x)
            ]
            if len(missing_keys) > 0 or len(unexpected_keys) > 0:
                raise RuntimeError(
                    f"Missing keys: {missing_keys}, unexpected_keys: {unexpected_keys}"
                )


__all__ = ["SAMAudio"]
