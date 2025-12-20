import os
from contextlib import nullcontext

import torch


def _env_truthy(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value not in {"0", "false", "False", "no", "NO"}


def get_default_device() -> torch.device:
    if torch.cuda.is_available():
        local_rank = os.environ.get("LOCAL_RANK")
        if local_rank is None:
            local_rank = os.environ.get("RANK")
        if local_rank is not None:
            try:
                return torch.device(f"cuda:{int(local_rank)}")
            except Exception:
                return torch.device("cuda")
        return torch.device("cuda")
    return torch.device("cpu")


def _maybe_call(obj, name: str, *args, **kwargs) -> None:
    fn = getattr(obj, name, None)
    if callable(fn):
        fn(*args, **kwargs)


def auto_tune(device: torch.device | None = None) -> torch.device:
    device = device or get_default_device()
    if device.type != "cuda":
        return device

    if _env_truthy("SAM_AUDIO_ENABLE_TF32", True):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision(
            os.environ.get("SAM_AUDIO_MATMUL_PRECISION", "high")
        )

    if _env_truthy("SAM_AUDIO_CUDNN_BENCHMARK", True):
        torch.backends.cudnn.benchmark = True

    if _env_truthy("SAM_AUDIO_ENABLE_SDP", True):
        _maybe_call(torch.backends.cuda, "enable_flash_sdp", True)
        _maybe_call(torch.backends.cuda, "enable_mem_efficient_sdp", True)
        _maybe_call(torch.backends.cuda, "enable_math_sdp", True)

    return device


def get_autocast_dtype(device: torch.device | None = None) -> torch.dtype | None:
    device = device or get_default_device()
    if device.type != "cuda":
        return None

    requested = os.environ.get("SAM_AUDIO_AMP_DTYPE")
    if requested is not None:
        requested = requested.lower().strip()
        if requested in {"bf16", "bfloat16"}:
            return torch.bfloat16
        if requested in {"fp16", "float16", "half"}:
            return torch.float16

    if torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def autocast_context(
    device: torch.device | None = None,
    *,
    enabled: bool | None = None,
    dtype: torch.dtype | None = None,
):
    device = device or get_default_device()
    if enabled is None:
        enabled = device.type == "cuda" and _env_truthy("SAM_AUDIO_AMP", True)
    if not enabled or device.type != "cuda":
        return nullcontext()

    dtype = dtype or get_autocast_dtype(device) or torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def optimize_cuda_allocator(device: torch.device | None = None) -> None:
    """
    Optimize CUDA memory allocator for better fragmentation handling.

    Args:
        device: Target CUDA device. Uses default if None.
    """
    device = device or get_default_device()
    if device.type != "cuda":
        return

    # Use expandable segments for better large allocation handling
    if hasattr(torch.cuda.memory, "set_per_process_memory_fraction"):
        # Leave some memory for the system (95% max usage)
        try:
            torch.cuda.memory.set_per_process_memory_fraction(0.95, device)
        except RuntimeError:
            pass  # Already set or not supported

    # Enable memory-efficient allocator settings
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def clear_cuda_cache(log: bool = True) -> None:
    """
    Clear CUDA memory cache and run garbage collection.

    Args:
        log: Whether to log memory usage before/after clearing.
    """
    if not torch.cuda.is_available():
        return

    import gc

    if log:
        try:
            allocated_before = torch.cuda.memory_allocated() / (1024**3)
            reserved_before = torch.cuda.memory_reserved() / (1024**3)
        except Exception:
            log = False

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    if log:
        try:
            allocated_after = torch.cuda.memory_allocated() / (1024**3)
            reserved_after = torch.cuda.memory_reserved() / (1024**3)
            from sam_audio.logging_config import get_logger
            logger = get_logger(__name__)
            logger.debug(
                "CUDA cache cleared: allocated %.2f->%.2f GB, reserved %.2f->%.2f GB",
                allocated_before, allocated_after,
                reserved_before, reserved_after,
            )
        except Exception:
            pass

