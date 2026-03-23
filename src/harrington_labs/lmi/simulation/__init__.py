"""LMI simulation engines — beam propagation, nonlinear optics, thermal.

All engines use harrington_common.compute for automatic acceleration:
    CUDA GPU → Numba JIT → NumPy (portable fallback)
"""
from .beam_propagation import *  # noqa: F401,F403
from .nonlinear import *  # noqa: F401,F403
from .thermal import *  # noqa: F401,F403
