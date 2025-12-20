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
    index: int = 0


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
        pynvml.nvmlShutdown()
        return GPUInfo(name=name, vram_gb=vram_gb, index=device_index)
    except Exception:
        # Fall back to torch properties
        try:
            props = torch.cuda.get_device_properties(device_index)
            return GPUInfo(
                name=props.name,
                vram_gb=props.total_memory / (1024**3),
                index=device_index,
            )
        except Exception:
            return None


def estimate_model_vram(model_size: str) -> float:
    """
    Estimate VRAM requirements for a given model size.

    These estimates include:
    - Model parameters (fp16/bf16)
    - Activation memory during inference
    - Audio codec and text encoder overhead
    - Safety margin for PyTorch allocations

    Args:
        model_size: One of 'small', 'base', 'large'.

    Returns:
        Estimated VRAM requirement in GB.
    """
    # Parameter counts estimated from model configs (transformer dims/layers)
    # Plus auxiliary models (codec, text encoder, rankers)
    model_params = {
        "small": 0.3e9,   # ~300M parameters
        "base": 0.6e9,    # ~600M parameters
        "large": 1.2e9,   # ~1.2B parameters
    }

    # Bytes per parameter (fp16 = 2, plus optimizer states during loading)
    bytes_per_param = 2.5

    # Activation memory multiplier (sequence length dependent)
    # For 30s audio at 16kHz with codec compression
    activation_multiplier = 1.8

    # Fixed overhead: codec, text encoder, auxiliary models
    overhead_gb = 2.0

    params = model_params.get(model_size, model_params["small"])
    model_gb = (params * bytes_per_param) / (1024**3)
    total_gb = (model_gb * activation_multiplier) + overhead_gb

    return total_gb


def select_model(gpu_info: GPUInfo | None, safety_margin: float = 0.9) -> str:
    """
    Select the appropriate SAM-Audio model based on GPU VRAM.

    Dynamically calculates VRAM requirements based on model parameter counts
    and selects the largest model that fits within available memory.

    Args:
        gpu_info: GPU information or None for CPU.
        safety_margin: Use only this fraction of available VRAM (default 90%).

    Returns:
        HuggingFace model path string.
    """
    if gpu_info is None:
        return "facebook/sam-audio-small"

    available_vram = gpu_info.vram_gb * safety_margin

    # Try models from largest to smallest
    for size in ["large", "base", "small"]:
        required = estimate_model_vram(size)
        if available_vram >= required:
            return f"facebook/sam-audio-{size}"

    # Fall back to small if nothing fits (will likely OOM, but user can try)
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

    args = parser.parse_args()

    # Validate input file
    if not args.input.exists():
        print(f"Error: Input file not found: {args.input}", file=sys.stderr)
        return 1

    # Set up output paths
    stem = args.input.stem
    parent = args.input.parent
    output_target = args.output or parent / f"{stem}_target.wav"
    output_residual = args.residual or parent / f"{stem}_residual.wav"

    # Detect GPU
    gpu_info = get_gpu_info()
    if args.verbose:
        if gpu_info:
            print(f"Detected GPU: {gpu_info.name} ({gpu_info.vram_gb:.1f}GB VRAM)")
        else:
            print("No GPU detected, using CPU")

    # Select model
    model_path = args.model or select_model(gpu_info)
    if args.verbose:
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
    model = SAMAudio.from_pretrained(model_path)
    model = model.eval()
    if device.type == "cuda":
        model = model.cuda()

    processor = SAMAudioProcessor.from_pretrained(model_path)

    # Prepare batch
    if args.verbose:
        print(f"Processing: {args.input}")
        print(f"Description: {args.description}")

    batch = processor(
        audios=[str(args.input)],
        descriptions=[args.description],
    ).to(device)

    # Run separation
    with torch.inference_mode():
        result = model.separate(
            batch,
            predict_spans=args.predict_spans,
            reranking_candidates=candidates,
        )

    # Save outputs
    sample_rate = processor.audio_sampling_rate
    torchaudio.save(str(output_target), result.target.cpu(), sample_rate)
    torchaudio.save(str(output_residual), result.residual.cpu(), sample_rate)

    print(f"Target saved to: {output_target}")
    print(f"Residual saved to: {output_residual}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
