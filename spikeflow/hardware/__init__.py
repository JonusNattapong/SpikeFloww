"""
SpikeFlow Hardware Backend Module
Support for neuromorphic processors and optimization
"""

from .base import HardwareBackend
from .edge_optimizer import EdgeOptimizer
from .quantization import NetworkQuantizer

# Optional hardware backends
try:
    from .loihi import LoihiBackend
    LOIHI_AVAILABLE = True
except ImportError:
    LOIHI_AVAILABLE = False

try:
    from .spinnaker import SpiNNakerBackend
    SPINNAKER_AVAILABLE = True
except ImportError:
    SPINNAKER_AVAILABLE = False

__all__ = [
    'HardwareBackend', 'EdgeOptimizer', 'NetworkQuantizer'
]

if LOIHI_AVAILABLE:
    __all__.append('LoihiBackend')
if SPINNAKER_AVAILABLE:
    __all__.append('SpiNNakerBackend')
