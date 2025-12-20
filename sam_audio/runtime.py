import gc
import os
from contextlib import nullcontext
from functools import lru_cache
from typing import Optional

import torch
from sam_audio.metrics import log_metric


def _env_truthy(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value not in {"0", "false", "False", "no", "NO"}


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


@lru_cache(maxsize=1)
def _supports_compile() -> bool:
    """Check if torch.compile is available and functional."""
    if not hasattr(torch, "compile"):
        return False
    # Require PyTorch 2.0+
    major = int(torch.__version__.split(".")[0])
    return major >= 2


def should_compile() -> bool:
    """Check if model compilation is enabled and supported."""
    if not _supports_compile():
        return False
    return _env_truthy("SAM_AUDIO_COMPILE", default=True)


def compile_model(
    model: torch.nn.Module,
    *,
    mode: str = "reduce-overhead",
    fullgraph: bool = False,
    dynamic: bool = True,
) -> torch.nn.Module:
    """
    Compile a model with torch.compile for optimized inference.

    Args:
        model: PyTorch model to compile.
        mode: Compilation mode ('default', 'reduce-overhead', 'max-autotune').
        fullgraph: If True, require full graph compilation.
        dynamic: If True, enable dynamic shape support.

    Returns:
        Compiled model, or original model if compilation unavailable.
    """
    if not should_compile():
        return model

    try:
        return torch.compile(
            model,
            mode=mode,
            fullgraph=fullgraph,
            dynamic=dynamic,
        )
    except Exception:
        # Fallback to uncompiled model on any compilation error
        return model


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
            logger = get_logger(__name__)
            logger.debug(
                "CUDA cache cleared: allocated %.2f->%.2f GB, reserved %.2f->%.2f GB",
                allocated_before, allocated_after,
                reserved_before, reserved_after,
            )
            log_metric("cuda_cache_freed", reserved_before - reserved_after, "GB")
        except Exception:
            pass


# ODE solver presets for speed vs quality tradeoffs
ODE_PRESETS = {
    "fast": {"method": "euler", "options": {"step_size": 1 / 8}},  # 8 steps
    "balanced": {"method": "midpoint", "options": {"step_size": 1 / 16}},  # 16 steps
    "quality": {"method": "midpoint", "options": {"step_size": 2 / 32}},  # 16 steps (original)
    "max_quality": {"method": "dopri5", "rtol": 1e-3, "atol": 1e-5},  # adaptive
}


def get_ode_options(preset: str = "quality") -> dict:
    """
    Get ODE solver options for the specified preset.

    Args:
        preset: One of 'fast', 'balanced', 'quality', 'max_quality'.
                Can also be overridden via SAM_AUDIO_ODE_PRESET env var.

    Returns:
        Dictionary of ODE solver options for torchdiffeq.odeint.
    """
    env_preset = os.environ.get("SAM_AUDIO_ODE_PRESET")
    if env_preset is not None:
        preset = env_preset.lower().strip()

    if preset not in ODE_PRESETS:
        preset = "quality"

    # Allow step override via environment
    env_steps = _env_int("SAM_AUDIO_ODE_STEPS", 0)
    if env_steps > 0:
        options = ODE_PRESETS[preset].copy()
        if "options" in options:
            options["options"] = options["options"].copy()
            options["options"]["step_size"] = 1 / env_steps
        return options

    return ODE_PRESETS[preset]


def prefetch_cuda_info(device: torch.device | None = None) -> None:
    """
    Pre-warm CUDA device info queries to avoid first-call latency.

    Args:
        device: Target CUDA device. Uses default if None.
    """
    device = device or get_default_device()
    if device.type != "cuda":
        return

    # Warm up device properties cache
    try:
        _ = torch.cuda.get_device_properties(device)
        _ = torch.cuda.get_device_capability(device)
        _ = torch.cuda.memory_allocated(device)
        _ = torch.cuda.memory_reserved(device)
    except Exception:
        pass


def synchronize_cuda(device: torch.device | None = None) -> None:
    """
    Synchronize CUDA stream if on GPU.

    Args:
        device: Target CUDA device. Uses default if None.
    """
    device = device or get_default_device()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
