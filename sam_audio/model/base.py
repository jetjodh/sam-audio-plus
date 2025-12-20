# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

import json
import os
import sys
import time
from typing import Callable, Dict, Optional, Union

import torch
from huggingface_hub import ModelHubMixin, snapshot_download

from sam_audio.logging_config import LogContext, flush_output, get_logger
from sam_audio.runtime import auto_tune, compile_model, get_default_device, should_compile
from transformers import BitsAndBytesConfig


logger = get_logger(__name__)


class BaseModel(torch.nn.Module, ModelHubMixin):
    config_cls: Callable

    def device(self):
        return next(self.parameters()).device

    @classmethod
    def _from_pretrained(
        cls,
        *,
        model_id: str,
        cache_dir: str,
        force_download: bool,
        proxies: Optional[Dict],
        resume_download: bool,
        local_files_only: bool,
        token: Union[str, bool, None],
        map_location: str = "cpu",
        strict: bool = True,
        revision: Optional[str] = None,
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
        **model_kwargs,
    ):
        quantization_config = None
        if load_in_8bit or load_in_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=load_in_8bit,
                load_in_4bit=load_in_4bit,
                bnb_4bit_compute_dtype=torch.float16,
            )

        auto_device = os.environ.get("SAM_AUDIO_AUTO_DEVICE", "1") != "0"
        requested_device = os.environ.get("SAM_AUDIO_DEVICE")
        if requested_device is not None:
            device = torch.device(requested_device)
        else:
            device = get_default_device()
        device = auto_tune(device)
        oom_fallback = False

        verbose = os.environ.get("SAM_AUDIO_LOAD_VERBOSE") == "1"

        # Suppress HuggingFace Hub progress bars to prevent output buffering issues
        os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

        logger.info(
            "Loading %s (cache_dir=%s, local_files_only=%s, force_download=%s, revision=%s)",
            model_id,
            cache_dir,
            local_files_only,
            force_download,
            cls.revision,
        )
        flush_output()

        if os.path.isdir(model_id):
            cached_model_dir = model_id
        else:
            # Try loading from local cache first to avoid network checks
            try:
                # First attempt: strictly local
                # This avoids "Fetching X files" if we already have it
                cached_model_dir = snapshot_download(
                    repo_id=model_id,
                    revision=cls.revision,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    proxies=proxies,
                    resume_download=resume_download,
                    token=token,
                    local_files_only=True,
                )
            except Exception:
                # Fallback: allow network
                # We catch Exception broadly because hf_hub raises various errors for missing files
                with LogContext(f"Downloading snapshot for {model_id}", logger):
                    cached_model_dir = snapshot_download(
                        repo_id=model_id,
                        revision=cls.revision,
                        cache_dir=cache_dir,
                        force_download=force_download,
                        proxies=proxies,
                        resume_download=resume_download,
                        token=token,
                        local_files_only=local_files_only,
                    )
            logger.info("Snapshot ready at %s", cached_model_dir)
            flush_output()

        logger.info("Reading config.json from %s", cached_model_dir)
        with open(os.path.join(cached_model_dir, "config.json")) as fin:
            config = json.load(fin)

        for key, value in model_kwargs.items():
            if key in config:
                config[key] = value

        if quantization_config:
            config["quantization_config"] = quantization_config


        logger.info("Instantiating model modules for %s", model_id)
        flush_output()
        config = cls.config_cls(**config)
        model = cls(config)

        load_on_device = os.environ.get("SAM_AUDIO_LOAD_ON_DEVICE", "1") != "0"
        checkpoint_device = None
        if isinstance(map_location, str):
            if map_location.startswith("cuda"):
                checkpoint_device = torch.device(map_location)
            elif map_location == "cpu":
                checkpoint_device = torch.device("cpu")
        elif isinstance(map_location, torch.device):
            checkpoint_device = map_location

        if auto_device and device.type == "cuda" and load_on_device:
            if checkpoint_device is None or checkpoint_device.type == "cpu":
                checkpoint_device = device

        # Check if model has parameters on meta device (from meta init optimization)
        has_meta = any(p.device.type == "meta" for p in model.parameters())

        if not has_meta and checkpoint_device is not None and checkpoint_device.type != "cpu":
            try:
                model = model.to(checkpoint_device)
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    logger.warning(
                        "CUDA OOM while moving initialized model to %s; loading checkpoint on CPU",
                        checkpoint_device,
                    )
                    torch.cuda.empty_cache()
                    checkpoint_device = torch.device("cpu")
                    model = model.to("cpu")
                    oom_fallback = True
                else:
                    raise

        checkpoint_path = os.path.join(cached_model_dir, "checkpoint.pt")
        try:
            checkpoint_bytes = os.path.getsize(checkpoint_path)
            logger.info(
                "Loading checkpoint from %s (%.1f MB)",
                checkpoint_path,
                checkpoint_bytes / (1024 * 1024),
            )
        except OSError:
            logger.info("Loading checkpoint from %s", checkpoint_path)
        flush_output()

        # Optimization: Always load to CPU with mmap=True first.
        # This makes the initial load instant (virtual memory) and avoids
        # reading the entire 6GB+ file into RAM at once.
        # The data is read on-demand during load_state_dict.
        t0 = time.perf_counter()
        try:
            state_dict = torch.load(
                checkpoint_path,
                weights_only=True,
                map_location="cpu",
                mmap=True,
            )
        except Exception as e:
            logger.error("Failed to load checkpoint: %s", e)
            raise
            
        logger.info("Checkpoint loaded (%.1fs) - mmap active", time.perf_counter() - t0)
        flush_output()

        logger.info("Applying state dict (strict=%s)", strict)
        t0 = time.perf_counter()
        
        # Optimization: use assign=True for faster weight loading (skips some checks/copies)
        # using try-except to support older torch versions if needed, though project requires >=2.5
        try:
            model.load_state_dict(state_dict, strict=strict, assign=True)
        except TypeError:
            # Fallback for older torch versions without assign arg
            model.load_state_dict(state_dict, strict=strict)
            
        del state_dict
        logger.info("State dict applied (%.1fs)", time.perf_counter() - t0)
        flush_output()

        if auto_device and device.type == "cuda" and model.device().type == "cpu":
            if oom_fallback:
                logger.warning(
                    "Staying on CPU after CUDA OOM during checkpoint load (set SAM_AUDIO_AUTO_DEVICE=0 to silence)",
                )
            else:
                try:
                    model = model.to(device)
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        logger.warning(
                            "CUDA OOM while moving model to %s; staying on CPU",
                            device,
                        )
                        torch.cuda.empty_cache()
                    else:
                        raise

        # Compile DiT transformer for optimized inference (P0 optimization)
        # Uses fullgraph=True and reduce-overhead mode for maximum performance
        if should_compile() and hasattr(model, 'transformer'):
            try:
                logger.info("Compiling DiT transformer with torch.compile...")
                t0 = time.perf_counter()
                model.transformer = compile_model(
                    model.transformer,

                    mode=None, # Use default from env var
                    fullgraph=True,
                    dynamic=False,  # Use static shapes for better optimization
                )
                logger.info("DiT transformer compiled (%.1fs)", time.perf_counter() - t0)
                flush_output()
            except Exception as e:
                logger.warning("Failed to compile DiT transformer: %s", e)

        # Optional warmup to pre-compile CUDA kernels (P0 optimization)
        # Reduces latency on first real inference by 10-20%
        if os.environ.get("SAM_AUDIO_WARMUP", "1") != "0" and hasattr(model, 'warmup'):
            try:
                t0 = time.perf_counter()
                model.warmup()
                logger.info("Model warmup complete (%.1fs)", time.perf_counter() - t0)
                flush_output()
            except Exception as e:
                logger.warning("Model warmup failed (non-fatal): %s", e)

        return model
