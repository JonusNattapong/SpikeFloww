"""
SpikeFlow: Enhanced Spiking Neural Network Library
"""

__version__ = "0.1.0"
__author__ = "JonusNattapong"

# Core imports
from .core.neurons import *
from .core.synapses import *
from .core.networks import *

# High-level API
from .api import SpikeFlow, create_snn_classifier, Sequential

# Functional utilities
from . import functional
from . import optim
from . import datasets
from . import visualization

# Hardware backends
try:
    from . import hardware
except ImportError:
    # Hardware backends are optional
    pass

# Set default backend
import torch
_default_device = "cuda" if torch.cuda.is_available() else "cpu"

def set_backend(backend: str):
    """Set default backend for SpikeFlow operations"""
    global _default_device
    _default_device = backend

def get_backend() -> str:
    """Get current backend"""
    return _default_device

# Convenience functions
LIF = lambda *args, **kwargs: AdaptiveLeakyIntegrateAndFire(*args, **kwargs)
Izhikevich = lambda *args, **kwargs: IzhikevichNeuron(*args, **kwargs)
STDPLinear = lambda input_size=None, output_size=None, **kwargs: STDPSynapse(input_size, output_size, **kwargs)

__all__ = [
    # Core classes
    'AdaptiveLeakyIntegrateAndFire', 'IzhikevichNeuron', 'STDPSynapse',
    'SpikingSequential', 'RateEncoder', 'TemporalEncoder',
    
    # High-level API
    'SpikeFlow', 'create_snn_classifier', 'Sequential',
    'LIF', 'Izhikevich', 'STDPLinear',
    
    # Modules
    'functional', 'optim', 'datasets', 'visualization',
    
    # Utilities
    'set_backend', 'get_backend',
]
