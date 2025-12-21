# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved\n
import warnings
# Suppress pynvml deprecation warning globally
warnings.filterwarnings("ignore", category=FutureWarning, message="The pynvml package is deprecated")
# Suppress torch.compile/Inductor online softmax warning (informational, not an error)
# Note: PyTorch's warning message starts with a newline, hence the regex
warnings.filterwarnings("ignore", category=UserWarning, message=r"^\s*Online softmax is disabled")

from .model import *  # noqa
from .processor import *  # noqa
