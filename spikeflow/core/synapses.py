"""
Advanced Synaptic Models for SpikeFlow
⚡🔗 Sophisticated plasticity mechanisms for learning

This module provides various synaptic models:
- STDP (Spike-Timing Dependent Plasticity)
- Metaplasticity (plasticity of plasticity)
- Short-term plasticity (depression/facilitation)
- Current-based synapses
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Union, Callable
import math
import numpy as np

class CurrentBasedSynapse(nn.Module):
    """
    Basic current-based synapse without plasticity
    
    Features:
    - Fixed weights
    - Optional synaptic delays
    - Linear synaptic transmission
    """
    
    def __init__(self,
                 input_size: int,
                 output_size: int,
                 weight_init: str = 'xavier',
                 bias: bool = False,
                 delay_max: int = 1):
        super().__init__()
        
        self.input_size = input_size
        self.output_size = output_size
        self.delay_max = delay_max
        
        # Synaptic weights
        self.weight = nn.Parameter(torch.empty(output_size, input_size))
        
        # Optional bias
        if bias:
            self.bias = nn.Parameter(torch.zeros(output_size))
        else:
            self.register_parameter('bias', None)
        
        # Initialize weights
        self._init_weights(weight_init)
        
        # Delay buffer for spike transmission
        if delay_max > 1:
            self.register_buffer('spike_buffer', torch.zeros(delay_max, input_size))
            
    def _init_weights(self, init_type: str):
        """Initialize synaptic weights"""
        if init_type == 'xavier':
            nn.init.xavier_uniform_(self.weight)
        elif init_type == 'kaiming':
            nn.init.kaiming_uniform_(self.weight)
        elif init_type == 'normal':
            nn.init.normal_(self.weight, 0, 0.1)
        elif init_type == 'uniform':
            nn.init.uniform_(self.weight, -0.1, 0.1)
        else:
            raise ValueError(f"Unknown weight initialization: {init_type}")
    
    def forward(self, pre_spikes: torch.Tensor) -> torch.Tensor:
        """Forward synaptic transmission"""
        
        if self.delay_max > 1:
            # Update delay buffer
            self.spike_buffer = torch.roll(self.spike_buffer, -1, dims=0)
            self.spike_buffer[-1] = pre_spikes
            
            # Use delayed spikes
            delayed_spikes = self.spike_buffer[0]
        else:
            delayed_spikes = pre_spikes
        
        # Linear synaptic transmission
        output = F.linear(delayed_spikes, self.weight, self.bias)
        
        return output

class STDPSynapse(nn.Module):
    """Spike-Timing Dependent Plasticity Synapse"""
    
    def __init__(self,
                 input_size: int,
                 output_size: int,
                 tau_plus: float = 20.0,     # LTP time constant
                 tau_minus: float = 20.0,    # LTD time constant
                 A_plus: float = 0.01,       # LTP amplitude
                 A_minus: float = 0.01,      # LTD amplitude
                 w_max: float = 1.0,         # Maximum weight
                 w_min: float = 0.0,         # Minimum weight
                 delay_max: int = 10,        # Maximum synaptic delay
                 learning_rate: float = 1.0):
        
        super().__init__()
        
        self.input_size = input_size
        self.output_size = output_size
        self.tau_plus = tau_plus
        self.tau_minus = tau_minus
        self.A_plus = A_plus
        self.A_minus = A_minus
        self.w_max = w_max
        self.w_min = w_min
        self.delay_max = delay_max
        self.learning_rate = learning_rate
        
        # Synaptic weights (learnable)
        self.weight = nn.Parameter(
            torch.rand(output_size, input_size) * 0.5
        )
        
        # Synaptic delays (learnable)
        self.delay = nn.Parameter(
            torch.randint(1, delay_max + 1, (output_size, input_size)).float()
        )
        
        # STDP traces
        self.register_buffer('x_trace', torch.zeros(input_size))   # Pre-synaptic trace
        self.register_buffer('y_trace', torch.zeros(output_size))  # Post-synaptic trace
        
        # Spike buffers for delays
        self.register_buffer('spike_buffer', 
                           torch.zeros(delay_max, input_size))
        
        # Decay factors
        self.alpha_plus = torch.exp(torch.tensor(-1.0 / tau_plus))
        self.alpha_minus = torch.exp(torch.tensor(-1.0 / tau_minus))
        
        # Training flag
        self.stdp_learning = True
    
    def forward(self, 
                pre_spikes: torch.Tensor, 
                post_spikes: Optional[torch.Tensor] = None,
                learning: bool = True) -> torch.Tensor:
        """Forward pass with optional STDP learning"""
        
        # Ensure correct input shape
        if pre_spikes.dim() == 1:
            pre_spikes = pre_spikes.unsqueeze(0)  # Add batch dimension
        
        batch_size = pre_spikes.size(0)
        
        # Update spike buffer (circular buffer for delays)
        self.spike_buffer = torch.roll(self.spike_buffer, -1, dims=0)
        if batch_size == 1:
            self.spike_buffer[-1] = pre_spikes.squeeze(0)
        else:
            # For batch processing, use last spike in batch
            self.spike_buffer[-1] = pre_spikes[-1]
        
        # Apply synaptic delays and compute output
        delayed_spikes = self._apply_delays()
        
        # Compute synaptic current
        if batch_size == 1:
            synaptic_current = F.linear(delayed_spikes.unsqueeze(0), self.weight)
        else:
            synaptic_current = F.linear(pre_spikes, self.weight)
        
        # STDP learning
        if learning and self.stdp_learning and post_spikes is not None:
            if post_spikes.dim() == 1:
                post_spikes = post_spikes.unsqueeze(0)
            
            self._update_stdp_traces(pre_spikes[-1] if batch_size > 1 else pre_spikes.squeeze(0), 
                                   post_spikes[-1] if post_spikes.size(0) > 1 else post_spikes.squeeze(0))
            self._update_weights(pre_spikes[-1] if batch_size > 1 else pre_spikes.squeeze(0), 
                               post_spikes[-1] if post_spikes.size(0) > 1 else post_spikes.squeeze(0))
        
        return synaptic_current
    
    def _apply_delays(self) -> torch.Tensor:
        """Apply learnable synaptic delays"""
        delayed_spikes = torch.zeros(self.input_size, device=self.spike_buffer.device)
        
        for i in range(self.input_size):
            # Average delay across all output neurons for this input
            avg_delay = torch.round(self.delay[:, i].mean()).int().item()
            delay_idx = max(0, min(self.delay_max - 1, avg_delay - 1))
            delayed_spikes[i] = self.spike_buffer[delay_idx, i]
        
        return torch.clamp(delayed_spikes, 0, 1)
    
    def _update_stdp_traces(self, pre_spikes: torch.Tensor, post_spikes: torch.Tensor):
        """Update STDP eligibility traces"""
        self.x_trace = self.alpha_plus * self.x_trace + pre_spikes
        self.y_trace = self.alpha_minus * self.y_trace + post_spikes
    
    def _update_weights(self, pre_spikes: torch.Tensor, post_spikes: torch.Tensor):
        """Update synaptic weights using STDP rule"""
        
        # LTP: post spike causes potentiation of recently active pre-synapses
        ltp_update = torch.outer(post_spikes, self.x_trace) * self.A_plus
        
        # LTD: pre spike causes depression of recently active post-synapses  
        ltd_update = torch.outer(self.y_trace, pre_spikes) * self.A_minus
        
        # Apply weight updates
        weight_update = (ltp_update - ltd_update) * self.learning_rate
        
        # Update weights with bounds
        self.weight.data = torch.clamp(
            self.weight.data + weight_update,
            self.w_min, self.w_max
        )
    
    def set_learning(self, enabled: bool):
        """Enable/disable STDP learning"""
        self.stdp_learning = enabled


class MetaplasticSynapse(STDPSynapse):
    """Synapse with metaplasticity - plasticity of plasticity"""
    
    def __init__(self, *args, 
                 meta_rate: float = 0.001,
                 meta_tau: float = 1000.0,
                 **kwargs):
        super().__init__(*args, **kwargs)
        
        self.meta_rate = meta_rate
        self.meta_tau = meta_tau
        
        # Metaplastic variables
        self.register_buffer('theta', torch.ones_like(self.weight))  # Modification threshold
        self.register_buffer('activity_history', torch.zeros_like(self.weight))
        
        self.alpha_meta = torch.exp(torch.tensor(-1.0 / meta_tau))
    
    def _update_weights(self, pre_spikes: torch.Tensor, post_spikes: torch.Tensor):
        """STDP with metaplasticity"""
        
        # Standard STDP
        ltp_update = torch.outer(post_spikes, self.x_trace) * self.A_plus
        ltd_update = torch.outer(self.y_trace, pre_spikes) * self.A_minus
        
        # Metaplastic modification
        activity = torch.outer(post_spikes, pre_spikes)
        
        # Update activity history
        self.activity_history = (self.alpha_meta * self.activity_history + 
                               activity)
        
        # Modify plasticity based on recent activity
        meta_factor = torch.sigmoid(self.theta - self.activity_history)
        
        # Apply metaplastic modulation
        weight_update = (ltp_update - ltd_update) * meta_factor * self.learning_rate
        
        # Update weights
        self.weight.data = torch.clamp(
            self.weight.data + weight_update,
            self.w_min, self.w_max
        )
        
        # Update metaplastic threshold
        self.theta += self.meta_rate * (self.activity_history - self.theta)


class LinearSynapse(nn.Module):
    """Simple linear synapse without plasticity"""
    
    def __init__(self, input_size: int, output_size: int, bias: bool = True):
        super().__init__()
        self.linear = nn.Linear(input_size, output_size, bias=bias)
    
    def forward(self, pre_spikes: torch.Tensor, post_spikes: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.linear(pre_spikes)
