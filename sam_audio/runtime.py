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


def should_use_cuda_graphs() -> bool:
    """
    Check if CUDA graphs should be used for kernel launch overhead reduction.
    
    CUDA graphs are an alternative to torch.compile's reduce-overhead mode.
    They are most beneficial when torch.compile is disabled.
    
    Control via SAM_AUDIO_CUDA_GRAPHS environment variable.
    Defaults to False (torch.compile is preferred).
    """
    if not torch.cuda.is_available():
        return False
    # Only use CUDA graphs if explicitly enabled AND torch.compile is disabled
    # (since torch.compile with reduce-overhead already uses CUDA graphs internally)
    return _env_truthy("SAM_AUDIO_CUDA_GRAPHS", default=False) and not should_compile()


class CUDAGraphWrapper:
    """
    Wrapper for capturing and replaying CUDA graphs.
    
    This reduces kernel launch overhead by capturing a sequence of CUDA operations
    and replaying them as a single graph execution.
    
    Note: CUDA graphs require fixed input shapes. The wrapper handles warmup
    and graph capture automatically on first call with a new input shape.
    """
    
    def __init__(self, func, warmup_runs: int = 3):
        """
        Initialize CUDA graph wrapper.
        
        Args:
            func: The function to wrap. Must accept and return tensors.
            warmup_runs: Number of warmup runs before graph capture.
        """
        self.func = func
        self.warmup_runs = warmup_runs
        self._graphs = {}  # Cache graphs by input shape
        self._static_inputs = {}  # Static input tensors for graph replay
        self._static_outputs = {}  # Static output tensors from graph replay
        self._warmup_counts = {}  # Track warmup progress per shape
    
    def __call__(self, *args, **kwargs):
        """Execute the function, using CUDA graph if available."""
        if not torch.cuda.is_available() or not args:
            return self.func(*args, **kwargs)
        
        # Get input shape signature for caching
        shape_key = self._get_shape_key(args)
        
        # Check if we have a cached graph
        if shape_key in self._graphs:
            return self._replay_graph(shape_key, args)
        
        # Track warmup runs
        if shape_key not in self._warmup_counts:
            self._warmup_counts[shape_key] = 0
        
        # Run warmup
        if self._warmup_counts[shape_key] < self.warmup_runs:
            self._warmup_counts[shape_key] += 1
            return self.func(*args, **kwargs)
        
        # Capture graph after warmup
        try:
            return self._capture_and_run(shape_key, args, kwargs)
        except Exception:
            # Fall back to non-graph execution on any error
            return self.func(*args, **kwargs)
    
    def _get_shape_key(self, args):
        """Generate a hashable key from tensor shapes."""
        shapes = []
        for arg in args:
            if isinstance(arg, torch.Tensor):
                shapes.append(tuple(arg.shape))
            else:
                shapes.append(type(arg).__name__)
        return tuple(shapes)
    
    def _capture_and_run(self, shape_key, args, kwargs):
        """Capture CUDA graph and execute."""
        # Create static copies of inputs
        static_inputs = tuple(
            arg.clone() if isinstance(arg, torch.Tensor) else arg
            for arg in args
        )
        self._static_inputs[shape_key] = static_inputs
        
        # Warmup run before capture
        torch.cuda.synchronize()
        _ = self.func(*static_inputs, **kwargs)
        torch.cuda.synchronize()
        
        # Capture graph
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            static_output = self.func(*static_inputs, **kwargs)
        
        self._graphs[shape_key] = graph
        self._static_outputs[shape_key] = static_output
        
        # Replay with actual inputs
        return self._replay_graph(shape_key, args)
    
    def _replay_graph(self, shape_key, args):
        """Replay cached CUDA graph with new inputs."""
        # Copy inputs to static buffers
        static_inputs = self._static_inputs[shape_key]
        for src, dst in zip(args, static_inputs):
            if isinstance(src, torch.Tensor) and isinstance(dst, torch.Tensor):
                dst.copy_(src)
        
        # Replay graph
        self._graphs[shape_key].replay()
        
        # Return output (copy to avoid returning static buffer)
        output = self._static_outputs[shape_key]
        if isinstance(output, torch.Tensor):
            return output.clone()
        return output


def wrap_with_cuda_graph(func, warmup_runs: int = 3):
    """
    Wrap a function with CUDA graph capture for reduced kernel launch overhead.
    
    Args:
        func: Function to wrap.
        warmup_runs: Number of warmup runs before graph capture.
    
    Returns:
        Wrapped function that uses CUDA graphs when enabled.
    """
    if not should_use_cuda_graphs():
        return func
    return CUDAGraphWrapper(func, warmup_runs=warmup_runs)


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


# Attention kernel selection based on sequence length
# FlashAttention-2 is more efficient for longer sequences
# Memory-efficient attention is better for shorter sequences with irregular shapes
_FLASH_ATTENTION_THRESHOLD = _env_int("SAM_AUDIO_FLASH_THRESHOLD", 1024)


def get_preferred_sdp_backend(seq_len: int) -> str:
    """
    Get the preferred SDPA backend based on sequence length.
    
    Args:
        seq_len: Sequence length for the attention operation.
    
    Returns:
        Backend name: 'flash', 'mem_efficient', or 'math'
    """
    if seq_len >= _FLASH_ATTENTION_THRESHOLD:
        return "flash"
    return "mem_efficient"


from contextlib import contextmanager


@contextmanager
def sdp_kernel_context(seq_len: Optional[int] = None, backend: Optional[str] = None):
    """
    Context manager for selecting SDPA kernel based on sequence length.
    
    This optimizes attention performance by selecting the best kernel:
    - 'flash': FlashAttention-2, best for long sequences (>=1024)
    - 'mem_efficient': Memory-efficient attention, best for shorter sequences
    - 'math': Standard math implementation, fallback
    
    Args:
        seq_len: Sequence length to auto-select backend for.
        backend: Explicit backend override ('flash', 'mem_efficient', 'math').
    
    Example:
        with sdp_kernel_context(seq_len=2048):
            # Will use FlashAttention-2
            output = F.scaled_dot_product_attention(q, k, v)
    """
    if not torch.cuda.is_available():
        yield
        return
    
    # Determine which backend to use
    if backend is None and seq_len is not None:
        backend = get_preferred_sdp_backend(seq_len)
    
    if backend is None:
        yield
        return
    
    # Check if new SDPA kernel API is available (PyTorch 2.2+)
    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel
        
        backends_list = []
        if backend == "flash":
            backends_list.append(SDPBackend.FLASH_ATTENTION)
        elif backend == "mem_efficient":
            backends_list.append(SDPBackend.EFFICIENT_ATTENTION)
        elif backend == "math":
            backends_list.append(SDPBackend.MATH)
            
        if not backends_list:
            yield
            return
            
        try:
            with sdpa_kernel(backends_list):
                yield
        except Exception:
            yield
        return

    except ImportError:
        # Fallback for older PyTorch versions
        if not hasattr(torch.backends.cuda, "sdp_kernel"):
            yield
            return
        
        try:
            if backend == "flash":
                with torch.backends.cuda.sdp_kernel(
                    enable_flash=True,
                    enable_mem_efficient=False,
                    enable_math=False,
                ):
                    yield
            elif backend == "mem_efficient":
                with torch.backends.cuda.sdp_kernel(
                    enable_flash=False,
                    enable_mem_efficient=True,
                    enable_math=False,
                ):
                    yield
            elif backend == "math":
                with torch.backends.cuda.sdp_kernel(
                    enable_flash=False,
                    enable_mem_efficient=False,
                    enable_math=True,
                ):
                    yield
            else:
                yield
        except Exception:
            yield


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

    # Only enable BF16 on Ampere+ (Compute Capability >= 8.0) to avoid "No available kernel" errors
    # Turing (7.5) and older do not support BF16 native instructions
    try:
        if torch.cuda.get_device_capability(device)[0] >= 8:
            if torch.cuda.is_bf16_supported():
                return torch.bfloat16
    except Exception:
        pass
        
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

from concurrent.futures import ThreadPoolExecutor

class AsyncInit:
    """Helper for initializing objects in a background thread."""
    def __init__(self, cls, *args, **kwargs):
        self.cls = cls
        self.args = args
        self.kwargs = kwargs
        self._future = ThreadPoolExecutor(max_workers=1).submit(self._init)

    def _init(self):
        return self.cls(*self.args, **self.kwargs)

    def result(self):
        return self._future.result()

def async_init(cls, *args, **kwargs) -> AsyncInit:
    return AsyncInit(cls, *args, **kwargs)
