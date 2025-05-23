import torch
import torch.nn.functional as F
from typing import Optional, Tuple

def spike_loss(output: torch.Tensor, 
               target: torch.Tensor,
               loss_type: str = 'rate') -> torch.Tensor:
    """
    Compute spike-based loss functions
    
    Args:
        output: Spike tensor (time, batch, neurons) or (batch, neurons)
        target: Target tensor (batch,) for classification
        loss_type: 'rate', 'temporal', 'first_spike'
    """
    
    if output.dim() == 3:  # (time, batch, neurons)
        time_steps = output.shape[0]
        
        if loss_type == 'rate':
            # Rate-based loss: use spike count as logits
            spike_counts = output.sum(dim=0)  # (batch, neurons)
            return F.cross_entropy(spike_counts, target)
        
        elif loss_type == 'temporal':
            # Temporal loss: weight later spikes less
            time_weights = torch.linspace(1.0, 0.1, time_steps, device=output.device)
            weighted_spikes = (output * time_weights.view(-1, 1, 1)).sum(dim=0)
            return F.cross_entropy(weighted_spikes, target)
        
        elif loss_type == 'first_spike':
            # First-to-spike loss
            first_spike_times = torch.zeros(output.shape[1], output.shape[2], device=output.device)
            
            for t in range(time_steps):
                mask = (output[t] > 0.5) & (first_spike_times == 0)
                first_spike_times[mask] = t + 1
            
            # Convert to scores (earlier spikes = higher scores)
            scores = time_steps - first_spike_times
            scores[first_spike_times == 0] = 0  # No spike = 0 score
            
            return F.cross_entropy(scores, target)
    
    else:  # (batch, neurons)
        return F.cross_entropy(output, target)


def poisson_encoding(data: torch.Tensor, 
                    time_steps: int,
                    max_rate: float = 100.0) -> torch.Tensor:
    """
    Convert static data to Poisson spike trains
    
    Args:
        data: Input data (batch, features)
        time_steps: Number of time steps
        max_rate: Maximum firing rate
    
    Returns:
        Spike trains (time_steps, batch, features)
    """
    
    batch_size, features = data.shape
    
    # Normalize data to [0, 1]
    normalized_data = torch.clamp(data, 0, 1)
    
    # Generate Poisson spikes
    spike_prob = normalized_data * max_rate / 1000.0  # Convert to probability per ms
    
    spikes = torch.zeros(time_steps, batch_size, features, device=data.device)
    
    for t in range(time_steps):
        spikes[t] = (torch.rand_like(normalized_data) < spike_prob).float()
    
    return spikes


def rate_encoding(data: torch.Tensor,
                 time_steps: int,
                 max_rate: float = 100.0) -> torch.Tensor:
    """
    Convert data to rate-based encoding
    """
    return poisson_encoding(data, time_steps, max_rate)


def temporal_encoding(data: torch.Tensor,
                     time_steps: int,
                     encoding_type: str = 'linear') -> torch.Tensor:
    """
    Convert data to temporal spike encoding
    
    Args:
        data: Input data (batch, features)
        time_steps: Number of time steps
        encoding_type: 'linear', 'exponential', 'gaussian'
    """
    
    batch_size, features = data.shape
    spikes = torch.zeros(time_steps, batch_size, features, device=data.device)
    
    # Normalize data to [0, 1]
    normalized_data = torch.clamp(data, 0, 1)
    
    if encoding_type == 'linear':
        # Linear temporal encoding: higher values spike earlier
        spike_times = ((1 - normalized_data) * (time_steps - 1)).long()
        
        for b in range(batch_size):
            for f in range(features):
                t = spike_times[b, f].item()
                if t < time_steps:
                    spikes[t, b, f] = 1.0
    
    elif encoding_type == 'gaussian':
        # Gaussian temporal encoding
        for b in range(batch_size):
            for f in range(features):
                if normalized_data[b, f] > 0:
                    center = normalized_data[b, f] * time_steps
                    sigma = time_steps * 0.1
                    
                    for t in range(time_steps):
                        prob = torch.exp(-0.5 * ((t - center) / sigma) ** 2)
                        if torch.rand(1) < prob * 0.1:  # Scale probability
                            spikes[t, b, f] = 1.0
    
    return spikes


def spike_regularization(spikes: torch.Tensor, 
                        reg_type: str = 'l1',
                        strength: float = 0.01) -> torch.Tensor:
    """
    Regularization for spike patterns
    
    Args:
        spikes: Spike tensor
        reg_type: 'l1', 'l2', 'entropy'
        strength: Regularization strength
    """
    
    if reg_type == 'l1':
        return strength * torch.sum(torch.abs(spikes))
    
    elif reg_type == 'l2':
        return strength * torch.sum(spikes ** 2)
    
    elif reg_type == 'entropy':
        # Encourage diverse spike patterns
        spike_probs = torch.mean(spikes, dim=0)  # Average over time
        entropy = -torch.sum(spike_probs * torch.log(spike_probs + 1e-8))
        return -strength * entropy  # Negative because we want to maximize entropy
    
    else:
        raise ValueError(f"Unknown regularization type: {reg_type}")


def compute_firing_rate(spikes: torch.Tensor, 
                       time_window: Optional[int] = None) -> torch.Tensor:
    """Compute firing rate from spike trains"""
    
    if time_window is None:
        if spikes.dim() == 3:  # (time, batch, neurons)
            return spikes.mean(dim=0)  # Average over time
        else:
            return spikes.mean(dim=-1)  # Average over last dimension
    else:
        # Compute rate in sliding windows
        if spikes.dim() == 3:
            time_steps = spikes.shape[0]
            rates = []
            
            for t in range(0, time_steps - time_window + 1, time_window):
                window_spikes = spikes[t:t+time_window]
                rates.append(window_spikes.mean(dim=0))
            
            return torch.stack(rates, dim=0)
        else:
            return spikes.mean(dim=-1)


def spike_distance(spikes1: torch.Tensor, 
                  spikes2: torch.Tensor,
                  metric: str = 'victor_purpura') -> torch.Tensor:
    """
    Compute distance between spike trains
    
    Args:
        spikes1, spikes2: Spike trains to compare
        metric: 'victor_purpura', 'van_rossum', 'isi'
    """
    
    if metric == 'victor_purpura':
        # Simplified Victor-Purpura distance
        spike_times1 = torch.where(spikes1 > 0.5)[0].float()
        spike_times2 = torch.where(spikes2 > 0.5)[0].float()
        
        if len(spike_times1) == 0 and len(spike_times2) == 0:
            return torch.tensor(0.0)
        elif len(spike_times1) == 0:
            return torch.tensor(float(len(spike_times2)))
        elif len(spike_times2) == 0:
            return torch.tensor(float(len(spike_times1)))
        
        # Simple implementation: count differences
        dist = torch.abs(len(spike_times1) - len(spike_times2))
        return torch.tensor(float(dist))
    
    elif metric == 'van_rossum':
        # Van Rossum distance using convolution
        tau = 10.0  # Time constant
        kernel_size = int(5 * tau)
        kernel = torch.exp(-torch.arange(kernel_size, dtype=torch.float) / tau)
        kernel = kernel / kernel.sum()
        
        # Convolve spikes with exponential kernel
        conv1 = F.conv1d(spikes1.unsqueeze(0).unsqueeze(0), 
                        kernel.unsqueeze(0).unsqueeze(0), padding=kernel_size//2)
        conv2 = F.conv1d(spikes2.unsqueeze(0).unsqueeze(0), 
                        kernel.unsqueeze(0).unsqueeze(0), padding=kernel_size//2)
        
        # Compute L2 distance
        return torch.sqrt(torch.sum((conv1 - conv2) ** 2))
    
    else:
        raise ValueError(f"Unknown spike distance metric: {metric}")
