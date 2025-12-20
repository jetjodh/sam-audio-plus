# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved\n
import warnings
# Suppress pynvml deprecation warning globally
warnings.filterwarnings("ignore", category=FutureWarning, message="The pynvml package is deprecated")

from .model import *  # noqa
from .processor import *  # noqa
