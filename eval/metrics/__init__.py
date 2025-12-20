# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved\n

from metrics.aes import Aesthetic
from metrics.clap import CLAP
from metrics.judge import Judge

__all__ = [
    "Aesthetic",
    "CLAP",
    "Judge",
]


def __getattr__(name: str):
    if name == "ImageBind":
        from metrics.imagebind import ImageBind

        return ImageBind
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
