"""
Network Architectures for SpikeFlow
🏗️ Building blocks for complex spiking neural networks

This module provides:
- Sequential SNN models
- Recurrent SNN architectures
- Modular layer components
- Network utilities
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Dict, Union, Tuple, Any
from .neurons import BaseSpikingNeuron
from .synapses import CurrentBasedSynapse, STDPSynapse

class SpikingLayer(nn.Module):
    """
    A complete spiking layer with synapses and neurons
    
    Features:
    - Flexible synapse-neuron combinations
    - Batch processing support
    - State management
    """
    
    def __init__(self,
                 synapse: nn.Module,
                 neuron: BaseSpikingNeuron,
                 name: Optional[str] = None):
        super().__init__()
        
        self.synapse = synapse
        self.neuron = neuron
        self.name = name or f"SpikingLayer_{id(self)}"
        
        # Validate compatibility
        if hasattr(synapse, 'output_size') and hasattr(neuron, 'shape'):
            neuron_size = neuron.neuron_count
            synapse_output = synapse.output_size
            
            if neuron_size != synapse_output:
                raise ValueError(
                    f"Synapse output size ({synapse_output}) doesn't match "
                    f"neuron input size ({neuron_size})"
                )
    
    def forward(self, input_spikes: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through synapse and neuron
        
        Args:
            input_spikes: Input spike tensor
            
        Returns:
            output_spikes: Output spike tensor
        """
        
        # Pass through synapse
        if hasattr(self.synapse, 'forward'):
            if isinstance(self.synapse, (STDPSynapse,)):
                # STDP synapses need both pre and post spikes
                # For feedforward, we'll use a dummy post-spike initially
                synaptic_current = self.synapse(input_spikes, torch.zeros_like(input_spikes), learning=False)
            else:
                synaptic_current = self.synapse(input_spikes)
        else:
            # Simple linear transformation
            synaptic_current = F.linear(input_spikes, self.synapse.weight)
        
        # Pass through neuron
        output_spikes = self.neuron(synaptic_current)
        
        # Update STDP if applicable
        if isinstance(self.synapse, (STDPSynapse,)) and self.training:
            # Now update with actual post-synaptic spikes
            self.synapse(input_spikes, output_spikes, learning=True)
        
        return output_spikes
    
    def reset_state(self):
        """Reset layer states"""
        if hasattr(self.neuron, 'reset_state'):
            self.neuron.reset_state()
        if hasattr(self.synapse, 'reset_state'):
            self.synapse.reset_state()

class SpikingSequential(nn.Module):
    """
    Sequential container for spiking layers
    
    Features:
    - Time-based simulation
    - Flexible layer stacking
    - State management across layers
    - Recording capabilities
    """
    
    def __init__(self, *layers, record_states: bool = False):
        super().__init__()
        
        self.layers = nn.ModuleList()
        self.record_states = record_states
        
        # Process layers
        for i, layer in enumerate(layers):
            if isinstance(layer, (nn.Module,)):
                self.layers.append(layer)
            else:
                raise TypeError(f"Layer {i} must be a nn.Module, got {type(layer)}")
        
        # Recording storage
        if record_states:
            self.recordings = {
                'spikes': [],
                'membrane_potentials': [],
                'layer_outputs': []
            }
    
    def forward(self, 
                x: torch.Tensor, 
                time_steps: Optional[int] = None,
                record_layer: Optional[int] = None) -> torch.Tensor:
        """
        Forward pass through time
        
        Args:
            x: Input tensor (time, batch, features) or (batch, features)
            time_steps: Number of simulation time steps
            record_layer: Layer index to record (for analysis)
            
        Returns:
            output: Output spikes (time, batch, features)
        """
        
        # Determine time steps
        if x.dim() == 3:  # (time, batch, features)
            actual_time_steps = x.size(0)
            batch_size = x.size(1)
        elif x.dim() == 2:  # (batch, features)
            if time_steps is None:
                time_steps = 100  # Default simulation time
            actual_time_steps = time_steps
            batch_size = x.size(0)
        else:
            raise ValueError(f"Input must be 2D or 3D, got {x.dim()}D")
        
        # Initialize output storage
        output_spikes = []
        
        if self.record_states:
            self.recordings = {
                'spikes': [],
                'membrane_potentials': [],
                'layer_outputs': []
            }
        
        # Simulate through time
        for t in range(actual_time_steps):
            # Get input at time t
            if x.dim() == 3:
                input_t = x[t]
            else:
                input_t = x  # Repeat same input
            
            # Forward through layers
            layer_input = input_t
            layer_outputs = []
            
            for layer_idx, layer in enumerate(self.layers):
                if isinstance(layer, SpikingLayer):
                    layer_output = layer(layer_input)
                elif hasattr(layer, 'forward'):
                    layer_output = layer(layer_input)
                else:
                    raise TypeError(f"Unsupported layer type: {type(layer)}")
                
                layer_outputs.append(layer_output)
                layer_input = layer_output
                
                # Record specific layer if requested
                if record_layer == layer_idx and self.record_states:
                    if hasattr(layer, 'neuron') and hasattr(layer.neuron, 'membrane_potential'):
                        self.recordings['membrane_potentials'].append(
                            layer.neuron.membrane_potential.clone()
                        )
            
            # Store final output
            output_spikes.append(layer_input)
            
            # Record layer outputs
            if self.record_states:
                self.recordings['layer_outputs'].append(layer_outputs)
                self.recordings['spikes'].append(layer_input.clone())
        
        # Stack outputs (time, batch, features)
        return torch.stack(output_spikes, dim=0)
    
    def reset_state(self):
        """Reset all layer states"""
        for layer in self.layers:
            if hasattr(layer, 'reset_state'):
                layer.reset_state()
        
        if self.record_states:
            self.recordings = {
                'spikes': [],
                'membrane_potentials': [],
                'layer_outputs': []
            }
    
    def get_spike_rates(self, time_window: int = 100) -> List[torch.Tensor]:
        """Get spike rates for each layer"""
        spike_rates = []
        
        for layer in self.layers:
            if isinstance(layer, SpikingLayer) and hasattr(layer.neuron, 'get_spike_rate'):
                spike_rates.append(layer.neuron.get_spike_rate(time_window))
            else:
                spike_rates.append(None)
        
        return spike_rates
    
    def energy_consumption(self) -> float:
        """Estimate total energy consumption"""
        total_energy = 0.0
        
        for layer in self.layers:
            if isinstance(layer, SpikingLayer) and hasattr(layer.neuron, 'energy_consumption'):
                total_energy += layer.neuron.energy_consumption()
        
        return total_energy

class RecurrentSpikingNetwork(nn.Module):
    """
    Recurrent spiking neural network with flexible connectivity
    
    Features:
    - Arbitrary recurrent connections
    - Multiple neuron populations
    - Complex dynamics
    """
    
    def __init__(self,
                 neuron_populations: Dict[str, BaseSpikingNeuron],
                 connections: Dict[str, Dict[str, nn.Module]],
                 external_inputs: Optional[List[str]] = None):
        super().__init__()
        
        self.neuron_populations = nn.ModuleDict(neuron_populations)
        self.connections = nn.ModuleDict()
        self.external_inputs = external_inputs or []
        
        # Flatten nested connections dict
        for source, targets in connections.items():
            for target, synapse in targets.items():
                connection_name = f"{source}_to_{target}"
                self.connections[connection_name] = synapse
        
        # Store connection topology
        self.connectivity = connections
        
        # Current state
        self.current_spikes = {
            name: torch.zeros_like(neuron.spike_output)
            for name, neuron in self.neuron_populations.items()
        }
    
    def forward(self,
                external_inputs: Dict[str, torch.Tensor],
                time_steps: int = 100) -> Dict[str, torch.Tensor]:
        """
        Simulate recurrent network dynamics
        
        Args:
            external_inputs: Dict of external input currents for each population
            time_steps: Number of simulation steps
            
        Returns:
            spike_history: Dict of spike histories for each population
        """
        
        # Initialize spike history storage
        spike_history = {
            name: []
            for name in self.neuron_populations.keys()
        }
        
        # Simulate through time
        for t in range(time_steps):
            # Compute input currents for each population
            population_currents = {}
            
            for pop_name, neuron_pop in self.neuron_populations.items():
                total_current = torch.zeros_like(neuron_pop.membrane_potential)
                
                # Add external inputs
                if pop_name in external_inputs:
                    if external_inputs[pop_name].dim() == 2:  # (time, features)
                        total_current += external_inputs[pop_name][t]
                    else:  # (features,) - constant input
                        total_current += external_inputs[pop_name]
                
                # Add recurrent inputs
                for source_pop in self.neuron_populations.keys():
                    connection_name = f"{source_pop}_to_{pop_name}"
                    
                    if connection_name in self.connections:
                        synapse = self.connections[connection_name]
                        source_spikes = self.current_spikes[source_pop]
                        
                        if isinstance(synapse, STDPSynapse):
                            # STDP needs both pre and post spikes
                            recurrent_current = synapse(
                                source_spikes, 
                                self.current_spikes[pop_name],
                                learning=self.training
                            )
                        else:
                            recurrent_current = synapse(source_spikes)
                        
                        total_current += recurrent_current
                
                population_currents[pop_name] = total_current
            
            # Update all populations simultaneously
            new_spikes = {}
            for pop_name, neuron_pop in self.neuron_populations.items():
                spikes = neuron_pop(population_currents[pop_name])
                new_spikes[pop_name] = spikes
                spike_history[pop_name].append(spikes.clone())
            
            # Update current spikes
            self.current_spikes = new_spikes
        
        # Convert to tensors
        for pop_name in spike_history:
            spike_history[pop_name] = torch.stack(spike_history[pop_name], dim=0)
        
        return spike_history
    
    def reset_state(self):
        """Reset all network states"""
        for neuron_pop in self.neuron_populations.values():
            neuron_pop.reset_state()
        
        for synapse in self.connections.values():
            if hasattr(synapse, 'reset_state'):
                synapse.reset_state()
        
        # Reset current spikes
        for name, neuron in self.neuron_populations.items():
            self.current_spikes[name] = torch.zeros_like(neuron.spike_output)

class ReservoirComputing(nn.Module):
    """
    Liquid State Machine / Echo State Network for SNN
    
    Features:
    - Random recurrent reservoir
    - Trainable readout
    - Temporal processing capabilities
    """
    
    def __init__(self,
                 input_size: int,
                 reservoir_size: int,
                 output_size: int,
                 neuron_type: str = 'LIF',
                 connectivity: float = 0.1,
                 spectral_radius: float = 0.9):
        super().__init__()
        
        self.input_size = input_size
        self.reservoir_size = reservoir_size
        self.output_size = output_size
        self.connectivity = connectivity
        self.spectral_radius = spectral_radius
        
        # Create reservoir neurons
        if neuron_type == 'LIF':
            from .neurons import AdaptiveLeakyIntegrateAndFire
            self.reservoir = AdaptiveLeakyIntegrateAndFire(
                shape=(reservoir_size,),
                track_history=True
            )
        else:
            raise ValueError(f"Unsupported neuron type: {neuron_type}")
        
        # Input connections (random, fixed)
        self.input_weights = nn.Parameter(
            torch.randn(reservoir_size, input_size) * 0.1,
            requires_grad=False
        )
        
        # Recurrent connections (random, scaled)
        recurrent_weights = torch.randn(reservoir_size, reservoir_size)
        
        # Apply connectivity sparsity
        mask = torch.rand(reservoir_size, reservoir_size) < connectivity
        recurrent_weights *= mask.float()
        
        # Scale by spectral radius
        eigenvalues = torch.linalg.eigvals(recurrent_weights)
        current_spectral_radius = torch.max(torch.abs(eigenvalues)).real
        recurrent_weights *= spectral_radius / current_spectral_radius
        
        self.recurrent_weights = nn.Parameter(recurrent_weights, requires_grad=False)
        
        # Trainable readout (linear regression on reservoir states)
        self.readout = nn.Linear(reservoir_size, output_size)
        
        # State storage for readout training
        self.reservoir_states = []
    
    def forward(self, 
                input_sequence: torch.Tensor,
                collect_states: bool = False) -> torch.Tensor:
        """
        Process input sequence through reservoir
        
        Args:
            input_sequence: Input tensor (time, batch, features)
            collect_states: Whether to collect reservoir states for training
            
        Returns:
            output: Readout predictions (time, batch, output_size)
        """
        
        time_steps = input_sequence.size(0)
        batch_size = input_sequence.size(1) if input_sequence.dim() > 2 else 1
        
        outputs = []
        if collect_states:
            self.reservoir_states = []
        
        # Reset reservoir
        self.reservoir.reset_state()
        
        for t in range(time_steps):
            # Get input at time t
            if input_sequence.dim() == 3:
                input_t = input_sequence[t]
            else:
                input_t = input_sequence[t].unsqueeze(0)
            
            # Input current
            input_current = F.linear(input_t, self.input_weights)
            
            # Recurrent current
            reservoir_spikes = self.reservoir.spike_output
            recurrent_current = F.linear(reservoir_spikes, self.recurrent_weights)
            
            # Total current
            total_current = input_current + recurrent_current
            
            # Update reservoir
            reservoir_output = self.reservoir(total_current.squeeze())
            
            # Collect states if needed
            if collect_states:
                self.reservoir_states.append(self.reservoir.membrane_potential.clone())
            
            # Readout
            readout_input = self.reservoir.membrane_potential
            if batch_size > 1:
                readout_input = readout_input.unsqueeze(0).expand(batch_size, -1)
            
            output = self.readout(readout_input)
            outputs.append(output)
        
        return torch.stack(outputs, dim=0)
    
    def train_readout(self, 
                     input_sequences: torch.Tensor,
                     target_sequences: torch.Tensor,
                     washout_time: int = 50):
        """
        Train readout using ridge regression
        
        Args:
            input_sequences: Training input sequences
            target_sequences: Training target sequences  
            washout_time: Initial time steps to ignore (reservoir settling)
        """
        
        # Collect reservoir states
        with torch.no_grad():
            _ = self.forward(input_sequences, collect_states=True)
        
        # Prepare training data (ignore washout period)
        reservoir_states = torch.stack(self.reservoir_states[washout_time:], dim=0)
        targets = target_sequences[washout_time:]
        
        # Flatten time and batch dimensions
        X = reservoir_states.view(-1, self.reservoir_size)
        Y = targets.view(-1, self.output_size)
        
        # Ridge regression
        ridge_param = 1e-8
        I = torch.eye(self.reservoir_size, device=X.device) * ridge_param
        
        # Solve: W = (X^T X + λI)^(-1) X^T Y
        XTX = torch.mm(X.T, X)
        XTY = torch.mm(X.T, Y)
        
        W = torch.linalg.solve(XTX + I, XTY)
        
        # Update readout weights
        self.readout.weight.data = W.T
        self.readout.bias.data.zero_()
