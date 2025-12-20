#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

"""
SAM-Audio separation CLI with automatic GPU detection and model selection.

This script provides a command-line interface for audio separation using SAM-Audio,
with automatic GPU detection and model selection based on available VRAM.

Usage:
    uv run python scripts/separate.py --input audio.wav --description "guitar solo"
    uv run python scripts/separate.py --input audio.wav --description "speech" --predict-spans
"""

from __future__ import annotations

import argparse
import logging
import os
import signal
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
import torchaudio


@dataclass
class GPUInfo:
    """Information about an available GPU."""

    name: str
    vram_gb: float
    available_vram_gb: float = 0.0
    index: int = 0


@dataclass
class GPUProcess:
    """Information about a process using GPU VRAM."""

    pid: int
    name: str
    used_memory_mb: float


def get_gpu_info(device_index: int = 0) -> GPUInfo | None:
    """
    Get GPU information using pynvml.

    Args:
        device_index: CUDA device index to query.

    Returns:
        GPUInfo if a GPU is available, None otherwise.
    """
    if not torch.cuda.is_available():
        return None

    try:
        import pynvml

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
        name = pynvml.nvmlDeviceGetName(handle)
        if isinstance(name, bytes):
            name = name.decode("utf-8")
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        vram_gb = mem_info.total / (1024**3)
        available_vram_gb = mem_info.free / (1024**3)
        pynvml.nvmlShutdown()
        return GPUInfo(
            name=name,
            vram_gb=vram_gb,
            available_vram_gb=available_vram_gb,
            index=device_index,
        )
    except Exception:
        # Fall back to torch properties (no available VRAM info)
        try:
            props = torch.cuda.get_device_properties(device_index)
            total_gb = props.total_memory / (1024**3)
            return GPUInfo(
                name=props.name,
                vram_gb=total_gb,
                available_vram_gb=total_gb,  # Assume all available as fallback
                index=device_index,
            )
        except Exception:
            return None


def estimate_model_vram(model_size: str, predict_spans: bool = True, verbose: bool = False) -> float:
    """
    Estimate VRAM requirements for a given model size.

    These estimates include:
    - Model parameters (fp16/bf16)
    - Activation memory during inference
    - Audio codec (DAC) overhead
    - Text encoder (T5-base) overhead
    - Span predictor (PE-AV Frame Large + SIGLIP) when enabled
    - CLAP ranker (HTSAT-tiny + RoBERTa-base) overhead
    - Safety margin for PyTorch allocations and memory fragmentation

    Args:
        model_size: One of 'small', 'base', 'large'.
        predict_spans: Whether span prediction is enabled (loads additional model).
        verbose: If True, print detailed VRAM breakdown.

    Returns:
        Estimated VRAM requirement in GB.
    """
    # Verified parameter counts from model architectures
    # SAM-Audio: DiT transformer with varying dims/layers
    model_params = {
        "small": 300e6,   # ~300M parameters (dim=1024, layers=12)
        "base": 600e6,    # ~600M parameters (dim=1536, layers=16)
        "large": 1200e6,  # ~1.2B parameters (dim=2048, layers=24)
    }

    # Bytes per parameter: fp16 weights = 2 bytes
    # Plus activation memory during forward pass (~1.5x for ODE solver)
    bytes_per_param = 2.0
    activation_multiplier = 1.5

    # Fixed overhead components (in GB):
    # DAC audio codec: ~75M params → ~0.15GB weights + buffers
    dac_codec_gb = 0.3

    # T5-base text encoder: ~220M params → ~0.44GB weights + tokenizer/cache
    t5_encoder_gb = 0.8

    # Span predictor (PE-AV Frame Large): ~600M params (ViT-L + audio encoder)
    # Plus SIGLIP text encoder loaded by perception_models
    span_predictor_gb = 2.5 if predict_spans else 0.0

    # CLAP ranker (loaded for text-based reranking):
    # HTSAT-tiny audio encoder (~30M) + RoBERTa-base text (~125M) + fusion
    # Note: CLAP is only loaded when reranking_candidates > 1, but we budget for it
    clap_ranker_gb = 0.8

    # PyTorch CUDA memory fragmentation and allocation overhead
    fragmentation_gb = 1.0

    # Calculate model VRAM
    params = model_params.get(model_size, model_params["small"])
    model_weights_gb = (params * bytes_per_param) / (1024**3)
    model_activations_gb = model_weights_gb * (activation_multiplier - 1)
    model_total_gb = model_weights_gb + model_activations_gb

    # Sum all components
    total_gb = (
        model_total_gb +
        dac_codec_gb +
        t5_encoder_gb +
        span_predictor_gb +
        clap_ranker_gb +
        fragmentation_gb
    )

    if verbose:
        print(f"\n{'─' * 50}")
        print(f"VRAM Estimation for SAM-Audio {model_size}")
        print(f"{'─' * 50}")
        print(f"  SAM-Audio model ({model_size}):  {model_total_gb:>6.2f} GB")
        print(f"    - Weights ({params/1e6:.0f}M params):    {model_weights_gb:>6.2f} GB")
        print(f"    - Activations:              {model_activations_gb:>6.2f} GB")
        print(f"  DAC audio codec:              {dac_codec_gb:>6.2f} GB")
        print(f"  T5-base text encoder:         {t5_encoder_gb:>6.2f} GB")
        if predict_spans:
            print(f"  Span predictor (PE-AV):       {span_predictor_gb:>6.2f} GB")
        else:
            print(f"  Span predictor:               (disabled)")
        print(f"  CLAP ranker (budgeted):       {clap_ranker_gb:>6.2f} GB")
        print(f"  Fragmentation overhead:       {fragmentation_gb:>6.2f} GB")
        print(f"{'─' * 50}")
        print(f"  TOTAL ESTIMATED:              {total_gb:>6.2f} GB")
        print(f"{'─' * 50}\n")

    return total_gb


def get_gpu_processes(device_index: int = 0) -> list[GPUProcess]:
    """
    Get list of processes consuming GPU VRAM.

    Args:
        device_index: CUDA device index to query.

    Returns:
        List of GPUProcess objects, empty if unavailable.
    """
    processes: list[GPUProcess] = []

    try:
        import pynvml

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)

        # Get both compute and graphics processes
        try:
            compute_procs = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
        except Exception:
            compute_procs = []

        try:
            graphics_procs = pynvml.nvmlDeviceGetGraphicsRunningProcesses(handle)
        except Exception:
            graphics_procs = []

        seen_pids: set[int] = set()
        for proc in compute_procs + graphics_procs:
            if proc.pid in seen_pids:
                continue
            seen_pids.add(proc.pid)

            # Get process name
            try:
                proc_name = pynvml.nvmlSystemGetProcessName(proc.pid)
                if isinstance(proc_name, bytes):
                    proc_name = proc_name.decode("utf-8")
                proc_name = os.path.basename(proc_name)
            except Exception:
                proc_name = f"PID {proc.pid}"

            used_mb = (proc.usedGpuMemory or 0) / (1024**2)
            processes.append(GPUProcess(pid=proc.pid, name=proc_name, used_memory_mb=used_mb))

        pynvml.nvmlShutdown()
    except Exception:
        pass

    return sorted(processes, key=lambda p: p.used_memory_mb, reverse=True)


def display_gpu_info(gpu_info: GPUInfo, processes: list[GPUProcess]) -> None:
    """
    Display GPU VRAM information and process list.

    Args:
        gpu_info: GPU information object.
        processes: List of processes using GPU VRAM.
    """
    used_gb = gpu_info.vram_gb - gpu_info.available_vram_gb
    usage_pct = (used_gb / gpu_info.vram_gb) * 100 if gpu_info.vram_gb > 0 else 0

    print(f"\n{'=' * 60}")
    print(f"GPU: {gpu_info.name}")
    print(f"{'=' * 60}")
    print(f"  Total VRAM:     {gpu_info.vram_gb:6.2f} GB")
    print(f"  Available VRAM: {gpu_info.available_vram_gb:6.2f} GB")
    print(f"  Used VRAM:      {used_gb:6.2f} GB ({usage_pct:.1f}%)")

    if processes:
        print(f"\n  {'PID':<10} {'Process':<30} {'Memory (MB)':>12}")
        print(f"  {'-' * 10} {'-' * 30} {'-' * 12}")
        for proc in processes:
            print(f"  {proc.pid:<10} {proc.name:<30} {proc.used_memory_mb:>12.1f}")
    else:
        print("\n  No GPU processes detected.")
    print(f"{'=' * 60}\n")


def prompt_process_termination(processes: list[GPUProcess]) -> bool:
    """
    Prompt user to optionally terminate VRAM-consuming processes.

    Args:
        processes: List of processes that could be terminated.

    Returns:
        True if any process was terminated, False otherwise.
    """
    if not processes:
        return False

    # Filter out current process
    current_pid = os.getpid()
    killable = [p for p in processes if p.pid != current_pid]

    if not killable:
        return False

    print("Would you like to terminate any processes to free up VRAM?")
    print("Enter PID(s) separated by commas, or press Enter to skip: ", end="")

    try:
        user_input = input().strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return False

    if not user_input:
        return False

    terminated = False
    pids_to_kill = [p.strip() for p in user_input.split(",") if p.strip()]

    for pid_str in pids_to_kill:
        try:
            pid = int(pid_str)
            if pid == current_pid:
                print(f"  Cannot terminate current process (PID {pid})")
                continue

            # Check if PID is in our known processes
            if not any(p.pid == pid for p in killable):
                print(f"  PID {pid} not found in GPU process list")
                continue

            os.kill(pid, signal.SIGTERM)
            print(f"  Sent SIGTERM to PID {pid}")
            terminated = True
        except ValueError:
            print(f"  Invalid PID: {pid_str}")
        except ProcessLookupError:
            print(f"  Process {pid_str} not found")
        except PermissionError:
            print(f"  Permission denied to terminate PID {pid_str}")
        except Exception as e:
            print(f"  Error terminating PID {pid_str}: {e}")

    return terminated


def select_model(
    gpu_info: GPUInfo | None,
    predict_spans: bool = True,
    vram_reserve_gb: float = 1.0,
    verbose: bool = False,
) -> str:
    """
    Select the appropriate SAM-Audio model based on GPU VRAM.

    Dynamically calculates VRAM requirements based on model parameter counts
    and selects the largest model that fits within available memory.

    Args:
        gpu_info: GPU information or None for CPU.
        predict_spans: Whether span prediction is enabled (affects VRAM requirements).
        vram_reserve_gb: Amount of VRAM (GB) to leave free for desktop responsiveness.
        verbose: If True, print detailed VRAM estimation breakdown.

    Returns:
        HuggingFace model path string.
    """
    if gpu_info is None:
        if verbose:
            print("No GPU detected, selecting smallest model for CPU inference.")
        return "facebook/sam-audio-small"

    # Reserve VRAM for desktop responsiveness (absolute reserve instead of percentage)
    available_vram = max(0, gpu_info.available_vram_gb - vram_reserve_gb)
    if verbose:
        print(f"\nAvailable VRAM: {gpu_info.available_vram_gb:.2f} GB (reserving {vram_reserve_gb:.1f} GB = {available_vram:.2f} GB usable)")

    # Try models from largest to smallest
    for size in ["large", "base", "small"]:
        required = estimate_model_vram(size, predict_spans=predict_spans, verbose=verbose)
        if available_vram >= required:
            if verbose:
                print(f"✓ Selected: facebook/sam-audio-{size} (requires {required:.2f} GB)")
            return f"facebook/sam-audio-{size}"
        elif verbose:
            print(f"✗ Skipping: facebook/sam-audio-{size} (requires {required:.2f} GB, exceeds available)")

    # Fall back to small if nothing fits (will likely OOM, but user can try)
    if verbose:
        print("⚠ Warning: No model fits comfortably. Using 'small' but may OOM.")
    return "facebook/sam-audio-small"


def select_reranking_candidates(gpu_info: GPUInfo | None, default: int = 8) -> int:
    """
    Adjust reranking candidates based on available VRAM.

    Args:
        gpu_info: GPU information or None for CPU.
        default: Default number of candidates.

    Returns:
        Adjusted number of reranking candidates.
    """
    if gpu_info is None:
        return 1  # CPU: minimal candidates

    if gpu_info.vram_gb < 10:
        return min(2, default)
    elif gpu_info.vram_gb < 16:
        return min(4, default)
    else:
        return default


def main() -> int:
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(
        description="Separate audio using SAM-Audio with automatic GPU detection.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic text prompting
    uv run python scripts/separate.py --input audio.wav --description "guitar solo"

    # With span prediction (better for non-ambient sounds)
    uv run python scripts/separate.py --input audio.wav --description "dog barking" --predict-spans

    # Force a specific model
    uv run python scripts/separate.py --input audio.wav --description "speech" --model facebook/sam-audio-large
        """,
    )

    parser.add_argument(
        "-i", "--input",
        type=Path,
        required=True,
        help="Path to input audio file.",
    )
    parser.add_argument(
        "-d", "--description",
        type=str,
        required=True,
        help="Text description of the sound to isolate.",
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=None,
        help="Path for output target audio. Defaults to <input>_target.wav",
    )
    parser.add_argument(
        "--residual",
        type=Path,
        default=None,
        help="Path for residual audio. Defaults to <input>_residual.wav",
    )
    parser.add_argument(
        "-m", "--model",
        type=str,
        default=None,
        help="HuggingFace model path. Auto-selected based on GPU if not specified.",
    )
    parser.add_argument(
        "--predict-spans",
        action="store_true",
        default=True,
        help="Enable span prediction (default: enabled).",
    )
    parser.add_argument(
        "--no-predict-spans",
        action="store_false",
        dest="predict_spans",
        help="Disable span prediction.",
    )
    parser.add_argument(
        "-c", "--candidates",
        type=int,
        default=None,
        help="Number of reranking candidates. Auto-adjusted based on GPU if not specified.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (e.g., 'cuda:0', 'cpu'). Auto-detected if not specified.",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Print detailed information.",
    )
    parser.add_argument(
        "-n", "--no-interactive",
        action="store_true",
        help="Skip interactive prompts (e.g., process termination).",
    )
    parser.add_argument(
        "--log-vram",
        action="store_true",
        help="Print detailed VRAM estimation breakdown during model selection.",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=None,
        help="Directory for log files. If specified, logs will be written to rotating files.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO).",
    )
    parser.add_argument(
        "--vram-reserve-gb",
        type=float,
        default=1.0,
        help="Amount of VRAM (GB) to leave free for desktop responsiveness (default: 1.0).",
    )

    args = parser.parse_args()

    # Set up logging early (before any model imports)
    from sam_audio.logging_config import setup_logging, get_logger, flush_output
    log_level = getattr(logging, args.log_level.upper(), logging.INFO)
    setup_logging(
        log_dir=args.log_dir,
        log_level=log_level,
        enable_console=args.verbose,
        enable_file=args.log_dir is not None,
    )
    logger = get_logger(__name__)

    # Validate input file
    if not args.input.exists():
        print(f"Error: Input file not found: {args.input}", file=sys.stderr)
        return 1

    # Set up output paths
    stem = args.input.stem
    parent = args.input.parent
    output_target = args.output or parent / f"{stem}_target.wav"
    output_residual = args.residual or parent / f"{stem}_residual.wav"

    # Detect GPU and display info
    gpu_info = get_gpu_info()
    if gpu_info:
        processes = get_gpu_processes(gpu_info.index)
        display_gpu_info(gpu_info, processes)

        # Offer to terminate processes if VRAM is low and interactive mode
        if not args.no_interactive and processes:
            used_gb = gpu_info.vram_gb - gpu_info.available_vram_gb
            if used_gb > 1.0:  # More than 1GB used
                if prompt_process_termination(processes):
                    # Re-query GPU info after termination
                    import time
                    time.sleep(1)  # Brief pause for cleanup
                    gpu_info = get_gpu_info()
                    if gpu_info:
                        print(f"Updated available VRAM: {gpu_info.available_vram_gb:.2f} GB")
    else:
        if args.verbose:
            print("No GPU detected, using CPU")

    # Select model
    log_vram = args.log_vram or args.verbose
    model_path = args.model or select_model(
        gpu_info,
        predict_spans=args.predict_spans,
        vram_reserve_gb=args.vram_reserve_gb,
        verbose=log_vram
    )
    if args.verbose and not log_vram:
        print(f"Using model: {model_path}")

    # Select reranking candidates
    candidates = args.candidates or select_reranking_candidates(gpu_info)
    if args.verbose:
        print(f"Reranking candidates: {candidates}")
        print(f"Span prediction: {args.predict_spans}")

    # Determine device
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    if args.verbose:
        print(f"Device: {device}")

    # Import SAM-Audio (deferred to allow --help without loading models)
    from sam_audio import SAMAudio, SAMAudioProcessor
    from sam_audio.runtime import auto_tune

    # Apply runtime optimizations
    device = auto_tune(device)

    # Load model and processor
    if args.verbose:
        print("Loading model...")
    logger.info("Loading model from %s", model_path)
    flush_output()
    model = SAMAudio.from_pretrained(model_path)
    model = model.eval()
    if device.type == "cuda":
        model = model.cuda()

    processor = SAMAudioProcessor.from_pretrained(model_path)

    # Prepare batch
    if args.verbose:
        print(f"Processing: {args.input}")
        print(f"Description: {args.description}")

    logger.info("Preparing batch for: %s", args.input)
    batch = processor(
        audios=[str(args.input)],
        descriptions=[args.description],
    ).to(device)
    logger.info("Batch prepared, starting separation")
    flush_output()

    # Run separation
    logger.info("Running audio separation (predict_spans=%s, candidates=%d)", args.predict_spans, candidates)
    flush_output()
    with torch.inference_mode():
        result = model.separate(
            batch,
            predict_spans=args.predict_spans,
            reranking_candidates=candidates,
        )
    logger.info("Separation complete")
    flush_output()

    # Save outputs (result.target/residual are lists for batch processing)
    sample_rate = processor.audio_sampling_rate
    target_audio = result.target[0] if isinstance(result.target, list) else result.target
    residual_audio = result.residual[0] if isinstance(result.residual, list) else result.residual
    
    logger.info("Saving output files")
    torchaudio.save(str(output_target), target_audio.cpu().unsqueeze(0), sample_rate)
    torchaudio.save(str(output_residual), residual_audio.cpu().unsqueeze(0), sample_rate)

    logger.info("Target saved to: %s", output_target)
    logger.info("Residual saved to: %s", output_residual)
    print(f"Target saved to: {output_target}")
    print(f"Residual saved to: {output_residual}")
    flush_output()

    return 0


if __name__ == "__main__":
    sys.exit(main())
