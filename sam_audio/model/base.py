# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved\n

import json
import logging
import os
import time
from typing import Callable, Dict, Optional, Union

import torch
from huggingface_hub import ModelHubMixin, snapshot_download

from sam_audio.runtime import auto_tune, get_default_device

logger = logging.getLogger(__name__)


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
        **model_kwargs,
    ):
        auto_device = os.environ.get("SAM_AUDIO_AUTO_DEVICE", "1") != "0"
        requested_device = os.environ.get("SAM_AUDIO_DEVICE")
        if requested_device is not None:
            device = torch.device(requested_device)
        else:
            device = get_default_device()
        device = auto_tune(device)

        verbose = os.environ.get("SAM_AUDIO_LOAD_VERBOSE") == "1"

        def emit(message: str, *args):
            logger.info(message, *args)
            if verbose:
                try:
                    text = message % args if args else message
                except Exception:
                    text = f"{message} {args}"
                print(f"[{__name__}] {text}", flush=True)

        emit(
            "Loading %s (cache_dir=%s, local_files_only=%s, force_download=%s, revision=%s)",
            model_id,
            cache_dir,
            local_files_only,
            force_download,
            cls.revision,
        )
        if os.path.isdir(model_id):
            cached_model_dir = model_id
        else:
            t0 = time.perf_counter()
            emit("Downloading snapshot for %s...", model_id)
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
            emit(
                "Snapshot ready for %s at %s (%.1fs)",
                model_id,
                cached_model_dir,
                time.perf_counter() - t0,
            )

        emit("Reading config.json from %s", cached_model_dir)
        with open(os.path.join(cached_model_dir, "config.json")) as fin:
            config = json.load(fin)

        for key, value in model_kwargs.items():
            if key in config:
                config[key] = value

        emit("Instantiating model modules for %s", model_id)
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

        if checkpoint_device is not None and checkpoint_device.type != "cpu":
            try:
                model = model.to(checkpoint_device)
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    emit(
                        "CUDA OOM while moving initialized model to %s; loading checkpoint on CPU",
                        checkpoint_device,
                    )
                    torch.cuda.empty_cache()
                    checkpoint_device = torch.device("cpu")
                    model = model.to("cpu")
                else:
                    raise

        checkpoint_path = os.path.join(cached_model_dir, "checkpoint.pt")
        try:
            checkpoint_bytes = os.path.getsize(checkpoint_path)
            emit(
                "Loading checkpoint from %s (%.1f MB)",
                checkpoint_path,
                checkpoint_bytes / (1024 * 1024),
            )
        except OSError:
            emit("Loading checkpoint from %s", checkpoint_path)

        t0 = time.perf_counter()
        oom_fallback = False
        mmap = None
        if checkpoint_device is None or checkpoint_device.type == "cpu":
            mmap = True
        else:
            mmap = False

        try:
            state_dict = torch.load(
                checkpoint_path,
                weights_only=True,
                map_location=checkpoint_device or map_location,
                mmap=mmap,
            )
        except RuntimeError as e:
            if (
                checkpoint_device is not None
                and checkpoint_device.type == "cuda"
                and "out of memory" in str(e).lower()
            ):
                emit(
                    "CUDA OOM while loading checkpoint on %s; falling back to CPU load",
                    checkpoint_device,
                )
                oom_fallback = True
                model = model.to("cpu")
                torch.cuda.empty_cache()
                state_dict = torch.load(
                    checkpoint_path,
                    weights_only=True,
                    map_location="cpu",
                    mmap=True,
                )
            else:
                raise
        emit("Checkpoint loaded (%.1fs)", time.perf_counter() - t0)

        emit("Applying state dict (strict=%s)", strict)
        t0 = time.perf_counter()
        model.load_state_dict(state_dict, strict=strict)
        del state_dict
        emit("State dict applied (%.1fs)", time.perf_counter() - t0)

        if auto_device and device.type == "cuda" and model.device().type == "cpu":
            if oom_fallback:
                emit(
                    "Staying on CPU after CUDA OOM during checkpoint load (set SAM_AUDIO_AUTO_DEVICE=0 to silence)",
                )
            else:
                try:
                    model = model.to(device)
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        emit(
                            "CUDA OOM while moving model to %s; staying on CPU",
                            device,
                        )
                        torch.cuda.empty_cache()
                    else:
                        raise
        return model
