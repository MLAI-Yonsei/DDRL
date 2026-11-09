"""Small helpers: seeding, CUDA timing, attribute-style config."""
import os
import random
import time
from contextlib import contextmanager

import numpy as np
import torch


def seed_everything(seed):
    """Seed Python, NumPy and PyTorch (CPU and CUDA)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def enable_determinism(cudnn_deterministic=True):
    """Deterministic cuBLAS/cuDNN kernels. Call before the first CUDA operation."""
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    if cudnn_deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


def cuda_sync():
    """CUDA kernels run asynchronously; synchronize before reading a timer."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()


class Stopwatch:
    """Accumulates wall-clock time per named phase (synchronized with the GPU)."""

    def __init__(self, *names):
        self.total = {name: 0.0 for name in names}

    @contextmanager
    def __call__(self, name):
        t0 = time.time()
        yield
        cuda_sync()
        self.total[name] += time.time() - t0


class Config(dict):
    """dict with attribute access (config.key)."""

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as e:
            raise AttributeError(key) from e

    def __setattr__(self, key, value):
        self[key] = value
