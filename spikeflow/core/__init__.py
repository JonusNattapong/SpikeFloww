"""
SpikeFlow Core Module
🧠 Advanced Spiking Neural Network Components

This module contains the fundamental building blocks for SNN:
- Neuron models (LIF, Izhikevich, etc.)
- Synaptic models (STDP, metaplasticity)
- Network architectures
- Encoding schemes
"""

# Neuron models
from .neurons import (
    BaseSpikingNeuron,
    AdaptiveLeakyIntegrateAndFire,
    IzhikevichNeuron,
    ExpIF,
    QuadraticIF
)

# Synaptic models
from .synapses import (
    STDPSynapse,
    MetaplasticSynapse,
    ShortTermPlasticitySynapse,
    CurrentBasedSynapse
)

# Network architectures
from .networks import (
    SpikingSequential,
    SpikingLayer,
    RecurrentSpikingNetwork
)

# Encoding schemes
from .encoding import (
    RateEncoder,
    TemporalEncoder,
    PopulationEncoder,
    LatencyEncoder
)

# Utility functions
from .utils import (
    spike_count,
    membrane_potential_plot,
    reset_network_state
)

__all__ = [
    # Neurons
    'BaseSpikingNeuron',
    'AdaptiveLeakyIntegrateAndFire',
    'IzhikevichNeuron', 
    'ExpIF',
    'QuadraticIF',
    
    # Synapses
    'STDPSynapse',
    'MetaplasticSynapse',
    'ShortTermPlasticitySynapse',
    'CurrentBasedSynapse',
    
    # Networks
    'SpikingSequential',
    'SpikingLayer',
    'RecurrentSpikingNetwork',
    
    # Encoding
    'RateEncoder',
    'TemporalEncoder', 
    'PopulationEncoder',
    'LatencyEncoder',
    
    # Utils
    'spike_count',
    'membrane_potential_plot',
    'reset_network_state'
]
