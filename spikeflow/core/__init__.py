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
    IzhikevichNeuron
)

# Synaptic models
from .synapses import (
    STDPSynapse,
    MetaplasticSynapse,
    CurrentBasedSynapse
)

# Network architectures
from .networks import (
    SpikingSequential,
    SpikingLayer,
    RecurrentSpikingNetwork
)

__all__ = [
    # Neurons
    'BaseSpikingNeuron',
    'AdaptiveLeakyIntegrateAndFire',
    'IzhikevichNeuron', 
    
    # Synapses
    'STDPSynapse',
    'MetaplasticSynapse',
    'CurrentBasedSynapse',
    
    # Networks
    'SpikingSequential',
    'SpikingLayer',
    'RecurrentSpikingNetwork',
]
