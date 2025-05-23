"""
SpikeFlow Main API
🚀 PyTorch-like interface for easy SNN development

This module provides:
- Simple model creation functions
- PyTorch-compatible interfaces
- Quick prototyping tools
- Hardware backend management
"""

import torch
import torch.nn as nn
from typing import Optional, List, Union, Tuple
from .core.neurons import AdaptiveLeakyIntegrateAndFire, IzhikevichNeuron, PoissonNeuron
from .core.synapses import STDPSynapse, LinearSynapse, MetaplasticSynapse

class SpikeFlow:
    """Main SpikeFlow API - PyTorch-like interface"""
    
    def __init__(self, backend: str = 'cpu'):
        self.backend = backend
        self.device = torch.device(backend if backend in ['cpu', 'cuda'] else 'cpu')
        
        # Initialize hardware backend if available
        self.hw_backend = None
        if backend == 'loihi':
            try:
                from .hardware.loihi import LoihiBackend
                self.hw_backend = LoihiBackend()
            except ImportError:
                print("Warning: Loihi backend not available. Using CPU instead.")
                self.device = torch.device('cpu')
        elif backend == 'spinnaker':
            try:
                from .hardware.spinnaker import SpiNNakerBackend
                self.hw_backend = SpiNNakerBackend()
            except ImportError:
                print("Warning: SpiNNaker backend not available. Using CPU instead.")
                self.device = torch.device('cpu')
    
    def LIF(self, shape: Union[int, Tuple[int, ...]], **kwargs):
        """Create Leaky Integrate-and-Fire neuron"""
        return AdaptiveLeakyIntegrateAndFire(shape, device=str(self.device), **kwargs)
    
    def Izhikevich(self, shape: Union[int, Tuple[int, ...]], **kwargs):
        """Create Izhikevich neuron"""
        return IzhikevichNeuron(shape, device=str(self.device), **kwargs)
    
    def Poisson(self, shape: Union[int, Tuple[int, ...]], **kwargs):
        """Create Poisson neuron"""
        return PoissonNeuron(shape, device=str(self.device), **kwargs)
    
    def STDPLinear(self, input_size: int, output_size: int, **kwargs):
        """Create STDP-enabled linear layer"""
        return STDPSynapse(input_size, output_size, **kwargs).to(self.device)
    
    def Linear(self, input_size: int, output_size: int, **kwargs):
        """Create standard linear layer"""
        return LinearSynapse(input_size, output_size, **kwargs).to(self.device)
    
    def MetaplasticLinear(self, input_size: int, output_size: int, **kwargs):
        """Create metaplastic synapse layer"""
        return MetaplasticSynapse(input_size, output_size, **kwargs).to(self.device)
    
    def Sequential(self, *layers):
        """Create sequential SNN model"""
        return SpikingSequential(*layers)


class SpikingSequential(nn.Module):
    """Sequential container for spiking layers"""
    
    def __init__(self, *layers):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        
    def forward(self, x: torch.Tensor, time_steps: int = 100) -> torch.Tensor:
        """Forward pass through time"""
        
        # Initialize outputs
        outputs = []
        layer_outputs = [None] * len(self.layers)
        
        for t in range(time_steps):
            # Get input at time t
            if x.dim() == 3:  # (time, batch, features)
                input_t = x[t] if t < x.size(0) else torch.zeros_like(x[0])
            elif x.dim() == 2:  # (batch, features) - repeat for all time steps
                input_t = x
            else:  # (features,) - single sample
                input_t = x.unsqueeze(0)
            
            # Forward through layers
            layer_input = input_t
            
            for i, layer in enumerate(self.layers):
                if hasattr(layer, 'forward'):
                    # Check if it's a synapse (needs both pre and post spikes)
                    if hasattr(layer, 'x_trace'):  # STDP synapse
                        if i + 1 < len(self.layers) and hasattr(self.layers[i + 1], 'membrane_potential'):
                            # Next layer is a neuron, we'll get post spikes from it
                            layer_input = layer(layer_input)
                        else:
                            layer_input = layer(layer_input)
                    else:
                        layer_input = layer(layer_input)
                
                layer_outputs[i] = layer_input
            
            outputs.append(layer_input)
        
        # Stack outputs (time, batch, features)
        return torch.stack(outputs, dim=0)
    
    def reset_state(self):
        """Reset all layer states"""
        for layer in self.layers:
            if hasattr(layer, 'reset_state'):
                layer.reset_state()
    
    def set_learning(self, enabled: bool):
        """Enable/disable learning in all layers"""
        for layer in self.layers:
            if hasattr(layer, 'set_learning'):
                layer.set_learning(enabled)


def create_snn_classifier(input_size: int, 
                         hidden_sizes: List[int], 
                         output_size: int,
                         backend: str = 'cpu',
                         neuron_type: str = 'LIF',
                         synapse_type: str = 'STDP') -> SpikingSequential:
    """Create SNN classifier with simple API"""
    
    sf = SpikeFlow(backend=backend)
    
    layers = []
    current_size = input_size
    
    # Choose neuron and synapse types
    if neuron_type.upper() == 'LIF':
        neuron_class = sf.LIF
    elif neuron_type.upper() == 'IZHIKEVICH':
        neuron_class = sf.Izhikevich
    else:
        raise ValueError(f"Unknown neuron type: {neuron_type}")
    
    if synapse_type.upper() == 'STDP':
        synapse_class = sf.STDPLinear
    elif synapse_type.upper() == 'LINEAR':
        synapse_class = sf.Linear
    elif synapse_type.upper() == 'METAPLASTIC':
        synapse_class = sf.MetaplasticLinear
    else:
        raise ValueError(f"Unknown synapse type: {synapse_type}")
    
    # Hidden layers
    for hidden_size in hidden_sizes:
        layers.extend([
            synapse_class(current_size, hidden_size),
            neuron_class(hidden_size)
        ])
        current_size = hidden_size
    
    # Output layer
    layers.extend([
        synapse_class(current_size, output_size),
        neuron_class(output_size)
    ])
    
    return sf.Sequential(*layers)


# Convenience aliases
Sequential = SpikingSequential
LIF = lambda shape, **kwargs: AdaptiveLeakyIntegrateAndFire(shape, **kwargs)
Izhikevich = lambda shape, **kwargs: IzhikevichNeuron(shape, **kwargs)
STDPLinear = lambda in_size, out_size, **kwargs: STDPSynapse(in_size, out_size, **kwargs)
