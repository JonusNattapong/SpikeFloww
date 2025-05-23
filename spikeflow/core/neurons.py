"""
Advanced Spiking Neuron Models for SpikeFlow
🧠 Biologically-inspired yet computationally efficient neuron implementations

This module provides various spiking neuron models:
- Leaky Integrate-and-Fire (LIF) with adaptation
- Izhikevich neurons (rich dynamics)
- Exponential Integrate-and-Fire (ExpIF)
- Quadratic Integrate-and-Fire (QIF)
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict, Union
from abc import ABC, abstractmethod

class BaseSpikingNeuron(nn.Module, ABC):
    """Base class for all spiking neuron models"""
    
    def __init__(self, 
                 shape: Union[int, Tuple[int, ...]],
                 dt: float = 1.0,
                 device: str = 'cpu'):
        super().__init__()
        
        # Handle shape input
        if isinstance(shape, int):
            shape = (shape,)
        
        self.shape = shape
        self.dt = dt
        self.device = device
        
        # State variables
        self.register_buffer('membrane_potential', torch.zeros(shape, device=device))
        self.register_buffer('spike_history', torch.zeros(shape, device=device))
        
    @abstractmethod
    def forward(self, input_current: torch.Tensor) -> torch.Tensor:
        """Forward pass returning spike tensor"""
        pass
    
    def reset_state(self):
        """Reset neuron states"""
        self.membrane_potential.zero_()
        self.spike_history.zero_()
    
    def get_state(self) -> Dict[str, torch.Tensor]:
        """Get current neuron state"""
        return {
            'membrane_potential': self.membrane_potential.clone(),
            'spike_history': self.spike_history.clone()
        }


class AdaptiveLeakyIntegrateAndFire(BaseSpikingNeuron):
    """Advanced LIF with adaptation and multiple timescales"""
    
    def __init__(self, 
                 shape: Union[int, Tuple[int, ...]],
                 tau_mem: float = 20.0,      # Membrane time constant
                 tau_adapt: float = 200.0,   # Adaptation time constant
                 threshold: float = 1.0,     # Spike threshold
                 reset_potential: float = 0.0,
                 adaptation_strength: float = 0.1,
                 refractory_period: int = 2,
                 **kwargs):
        
        super().__init__(shape, **kwargs)
        
        # Parameters
        self.tau_mem = tau_mem
        self.tau_adapt = tau_adapt
        self.threshold = threshold
        self.reset_potential = reset_potential
        self.adaptation_strength = adaptation_strength
        self.refractory_period = refractory_period
        
        # Decay factors
        self.alpha_mem = torch.exp(torch.tensor(-self.dt / tau_mem))
        self.alpha_adapt = torch.exp(torch.tensor(-self.dt / tau_adapt))
        
        # Additional state variables
        self.register_buffer('adaptation_current', torch.zeros(self.shape, device=self.device))
        self.register_buffer('refractory_counter', torch.zeros(self.shape, dtype=torch.int, device=self.device))
        
    def forward(self, input_current: torch.Tensor) -> torch.Tensor:
        """Enhanced LIF dynamics with adaptation"""
        
        # Ensure input has correct shape
        if input_current.shape != self.shape:
            input_current = input_current.view(self.shape)
        
        # Update refractory counter
        self.refractory_counter = torch.clamp(self.refractory_counter - 1, 0)
        
        # Membrane potential dynamics (only for non-refractory neurons)
        non_refractory = (self.refractory_counter == 0)
        
        # Leak + input + adaptation
        self.membrane_potential = (
            self.alpha_mem * self.membrane_potential + 
            input_current - 
            self.adaptation_current
        ) * non_refractory.float()
        
        # Check for spikes
        spikes = (self.membrane_potential >= self.threshold).float()
        
        # Reset membrane potential for spiking neurons
        self.membrane_potential = torch.where(
            spikes.bool(),
            torch.tensor(self.reset_potential, device=self.device),
            self.membrane_potential
        )
        
        # Update adaptation current
        self.adaptation_current = (
            self.alpha_adapt * self.adaptation_current + 
            self.adaptation_strength * spikes
        )
        
        # Set refractory period for spiking neurons
        self.refractory_counter = torch.where(
            spikes.bool(),
            torch.tensor(self.refractory_period, device=self.device),
            self.refractory_counter
        )
        
        # Update spike history
        self.spike_history = spikes
        
        return spikes


class IzhikevichNeuron(BaseSpikingNeuron):
    """Izhikevich neuron model - computationally efficient yet biologically realistic"""
    
    def __init__(self, 
                 shape: Union[int, Tuple[int, ...]],
                 a: float = 0.02,  # Recovery variable decay
                 b: float = 0.2,   # Coupling strength
                 c: float = -65.0, # Reset potential
                 d: float = 8.0,   # Recovery increment
                 **kwargs):
        
        super().__init__(shape, **kwargs)
        
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        
        # Recovery variable
        self.register_buffer('recovery', torch.zeros(self.shape, device=self.device))
        
    def forward(self, input_current: torch.Tensor) -> torch.Tensor:
        """Izhikevich dynamics"""
        
        # Ensure input has correct shape
        if input_current.shape != self.shape:
            input_current = input_current.view(self.shape)
        
        # Membrane potential update
        v = self.membrane_potential
        u = self.recovery
        
        # Izhikevich equations
        dv = 0.04 * v**2 + 5 * v + 140 - u + input_current
        du = self.a * (self.b * v - u)
        
        self.membrane_potential += dv * self.dt
        self.recovery += du * self.dt
        
        # Check for spikes (v >= 30mV)
        spikes = (self.membrane_potential >= 30.0).float()
        
        # Reset for spiking neurons
        self.membrane_potential = torch.where(
            spikes.bool(),
            torch.tensor(self.c, device=self.device),
            self.membrane_potential
        )
        
        self.recovery = torch.where(
            spikes.bool(),
            self.recovery + self.d,
            self.recovery
        )
        
        # Update spike history
        self.spike_history = spikes
        
        return spikes


class PoissonNeuron(BaseSpikingNeuron):
    """Poisson neuron for rate-based encoding"""
    
    def __init__(self, shape: Union[int, Tuple[int, ...]], **kwargs):
        super().__init__(shape, **kwargs)
        
    def forward(self, rate: torch.Tensor) -> torch.Tensor:
        """Generate Poisson spikes based on input rate"""
        
        # Ensure input has correct shape
        if rate.shape != self.shape:
            rate = rate.view(self.shape)
        
        # Generate Poisson spikes
        spike_prob = rate * self.dt
        spikes = (torch.rand_like(spike_prob) < spike_prob).float()
        
        # Update spike history
        self.spike_history = spikes
        
        return spikes
