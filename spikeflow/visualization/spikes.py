import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import ListedColormap
from typing import Optional, Tuple, List, Union
import seaborn as sns

class SpikeVisualizer:
    """Comprehensive spike visualization toolkit"""
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8)):
        self.figsize = figsize
        plt.style.use('seaborn-v0_8')
        
    def plot_raster(self, 
                    spikes: torch.Tensor,
                    title: str = "Spike Raster Plot",
                    neuron_labels: Optional[List[str]] = None,
                    time_unit: str = "ms",
                    save_path: Optional[str] = None) -> plt.Figure:
        """Create raster plot of spike trains"""
        
        if spikes.dim() == 3:  # (time, batch, neurons)
            spikes = spikes[:, 0, :]  # Take first batch
        elif spikes.dim() == 1:  # (neurons,)
            spikes = spikes.unsqueeze(0)  # Add time dimension
        
        time_steps, n_neurons = spikes.shape
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Find spike times and neuron indices
        spike_times, neuron_indices = torch.where(spikes > 0.5)
        
        # Create raster plot
        ax.scatter(spike_times.numpy(), neuron_indices.numpy(), 
                  s=20, c='black', marker='|', alpha=0.7)
        
        ax.set_xlabel(f'Time ({time_unit})')
        ax.set_ylabel('Neuron Index')
        ax.set_title(title)
        ax.set_xlim(0, time_steps-1)
        ax.set_ylim(-0.5, n_neurons-0.5)
        
        if neuron_labels:
            ax.set_yticks(range(n_neurons))
            ax.set_yticklabels(neuron_labels)
        
        ax.grid(True, alpha=0.3)
        
        # Add statistics
        total_spikes = spikes.sum().item()
        firing_rate = total_spikes / (time_steps * n_neurons)
        ax.text(0.02, 0.98, f'Total spikes: {total_spikes:.0f}\nFiring rate: {firing_rate:.3f}',
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_spike_heatmap(self,
                          spikes: torch.Tensor,
                          window_size: int = 10,
                          title: str = "Spike Activity Heatmap") -> plt.Figure:
        """Create heatmap of spike activity over time windows"""
        
        if spikes.dim() == 3:
            spikes = spikes[:, 0, :]
        
        time_steps, n_neurons = spikes.shape
        
        # Bin spikes into windows
        n_windows = time_steps // window_size
        binned_spikes = spikes[:n_windows*window_size].view(n_windows, window_size, n_neurons)
        activity = binned_spikes.sum(dim=1)  # Sum over time within each window
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Create heatmap
        im = ax.imshow(activity.T.numpy(), aspect='auto', cmap='hot', interpolation='nearest')
        
        ax.set_xlabel('Time Window')
        ax.set_ylabel('Neuron Index')
        ax.set_title(title)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Spike Count')
        
        plt.tight_layout()
        return fig
    
    def plot_spike_histogram(self,
                           spikes: torch.Tensor,
                           bins: int = 50,
                           title: str = "Spike Count Distribution") -> plt.Figure:
        """Plot histogram of spike counts per neuron"""
        
        if spikes.dim() == 3:
            spikes = spikes[:, 0, :]
        
        # Count spikes per neuron
        spike_counts = spikes.sum(dim=0).numpy()
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.figsize)
        
        # Histogram
        ax1.hist(spike_counts, bins=bins, alpha=0.7, edgecolor='black')
        ax1.set_xlabel('Spike Count')
        ax1.set_ylabel('Number of Neurons')
        ax1.set_title('Distribution of Spike Counts')
        ax1.grid(True, alpha=0.3)
        
        # Box plot
        ax2.boxplot(spike_counts, vert=True)
        ax2.set_ylabel('Spike Count')
        ax2.set_title('Spike Count Statistics')
        ax2.grid(True, alpha=0.3)
        
        # Add statistics
        mean_spikes = np.mean(spike_counts)
        std_spikes = np.std(spike_counts)
        ax1.axvline(mean_spikes, color='red', linestyle='--', 
                   label=f'Mean: {mean_spikes:.1f}')
        ax1.legend()
        
        plt.suptitle(title)
        plt.tight_layout()
        return fig
    
    def animate_spikes(self,
                      spikes: torch.Tensor,
                      interval: int = 100,
                      save_path: Optional[str] = None) -> animation.FuncAnimation:
        """Create animated visualization of spike propagation"""
        
        if spikes.dim() != 3:  # Need (time, height, width) or (time, batch, neurons)
            raise ValueError("Animation requires 3D spike tensor")
        
        if spikes.shape[1] == 1:  # (time, 1, neurons) -> reshape to 2D grid
            n_neurons = spikes.shape[2]
            grid_size = int(np.sqrt(n_neurons))
            spikes = spikes.squeeze(1).view(spikes.shape[0], grid_size, grid_size)
        
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Initialize plot
        im = ax.imshow(spikes[0].numpy(), cmap='hot', vmin=0, vmax=1)
        ax.set_title('Spike Activity Animation')
        plt.colorbar(im, ax=ax, label='Spike Activity')
        
        def animate(frame):
            im.set_array(spikes[frame].numpy())
            ax.set_title(f'Spike Activity - Time Step {frame}')
            return [im]
        
        anim = animation.FuncAnimation(fig, animate, frames=spikes.shape[0],
                                     interval=interval, blit=True, repeat=True)
        
        if save_path:
            anim.save(save_path, writer='pillow', fps=10)
        
        return anim


def plot_spikes(spikes: torch.Tensor, **kwargs) -> plt.Figure:
    """Convenience function for spike plotting"""
    visualizer = SpikeVisualizer()
    return visualizer.plot_raster(spikes, **kwargs)


def plot_raster(spike_trains: List[torch.Tensor], 
                labels: Optional[List[str]] = None,
                colors: Optional[List[str]] = None) -> plt.Figure:
    """Plot multiple spike trains as raster plot"""
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    if colors is None:
        colors = plt.cm.tab10(np.linspace(0, 1, len(spike_trains)))
    
    y_offset = 0
    for i, spikes in enumerate(spike_trains):
        if spikes.dim() > 1:
            spikes = spikes.flatten()
        
        spike_times = torch.where(spikes > 0.5)[0].numpy()
        
        ax.scatter(spike_times, np.ones_like(spike_times) * y_offset,
                  s=20, c=[colors[i]], marker='|', alpha=0.7,
                  label=labels[i] if labels else f'Train {i}')
        
        y_offset += 1
    
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Spike Train')
    ax.set_title('Multiple Spike Trains')
    
    if labels:
        ax.legend()
    
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    return fig
