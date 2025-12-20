# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

import os
from functools import lru_cache
from typing import Optional, Tuple

import torch
import transformers

from sam_audio.model.config import T5EncoderConfig


# Cache size for text embeddings (0 = disabled)
_TEXT_CACHE_SIZE = int(os.environ.get("SAM_AUDIO_TEXT_CACHE_SIZE", "128"))


class T5TextEncoder(torch.nn.Module):
    """T5 text encoder with optional caching for repeated descriptions."""

    def __init__(self, cfg: T5EncoderConfig):
        super().__init__()
        self.model = transformers.T5EncoderModel.from_pretrained(cfg.name)
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(cfg.name)
        self.pad_mode = cfg.pad_mode
        self.max_length = cfg.max_length
        self._cache_enabled = _TEXT_CACHE_SIZE > 0
        self._cache: dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
        self._cache_device: Optional[torch.device] = None

    def clear_cache(self) -> None:
        """Clear the text embedding cache."""
        self._cache.clear()
        self._cache_device = None

    def _encode_single(self, text: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode a single text string."""
        device = next(self.model.parameters()).device
        encoded = self.tokenizer(
            [text],
            truncation=True,
            max_length=self.max_length,
            padding="max_length",  # Use max_length for consistent shapes
            return_tensors="pt",
        )

        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)
        res = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )["last_hidden_state"]

        return res.squeeze(0), attention_mask.squeeze(0).bool()

    def forward(self, texts: list[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        device = next(self.model.parameters()).device

        # Check if device changed (invalidates cache)
        if self._cache_device is not None and self._cache_device != device:
            self.clear_cache()
        self._cache_device = device

        # Check for single text with caching
        if self._cache_enabled and len(texts) == 1:
            text = texts[0]
            if text in self._cache:
                cached_res, cached_mask = self._cache[text]
                # Move to current device if needed
                if cached_res.device != device:
                    cached_res = cached_res.to(device)
                    cached_mask = cached_mask.to(device)
                    self._cache[text] = (cached_res, cached_mask)
                return cached_res.unsqueeze(0), cached_mask.unsqueeze(0)

        # Standard encoding path
        encoded = self.tokenizer(
            texts,
            truncation=True,
            max_length=self.max_length,
            padding=self.pad_mode,
            return_tensors="pt",
        )

        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)
        res = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )["last_hidden_state"]

        # Cache single results
        if self._cache_enabled and len(texts) == 1:
            if len(self._cache) >= _TEXT_CACHE_SIZE:
                # Simple LRU: remove oldest entry
                oldest_key = next(iter(self._cache))
                del self._cache[oldest_key]
            self._cache[texts[0]] = (res.squeeze(0), attention_mask.squeeze(0).bool())

        return res, attention_mask.bool()
