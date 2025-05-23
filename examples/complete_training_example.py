"""
Complete SpikeFlow Training Example
Demonstrates end-to-end SNN training with all features
"""

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import time
import os
import sys

# Add SpikeFlow to path
sys.path.append('..')
import spikeflow as sf
from spikeflow.datasets import NMNIST
from spikeflow.visualization import SpikeVisualizer
from spikeflow.hardware import EdgeOptimizer

class SNNTrainer:
    """Complete SNN training pipeline"""
    
    def __init__(self, 
                 model: sf.SpikingSequential,
                 device: str = 'cpu',
                 time_steps: int = 100):
        
        self.model = model.to(device)
        self.device = device
        self.time_steps = time_steps
        self.visualizer = SpikeVisualizer()
        
        # Training history
        self.train_losses = []
        self.train_accuracies = []
        self.val_losses = []
        self.val_accuracies = []
        
        # Performance metrics
        self.inference_times = []
        self.energy_estimates = []
    
    def train_epoch(self, 
                   train_loader: DataLoader,
                   optimizer: optim.Optimizer,
                   criterion: callable,
                   epoch: int) -> Tuple[float, float]:
        """Train for one epoch"""
        
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        # Enable STDP learning
        self.model.set_learning(True)
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            # Reset neuron states for each batch
            self.model.reset_state()
            
            # Convert static data to spike trains
            if data.dim() == 4:  # Image data (batch, channels, height, width)
                data = data.view(data.size(0), -1)  # Flatten to (batch, features)
            
            spike_data = sf.functional.poisson_encoding(data, self.time_steps)
            
            # Forward pass
            optimizer.zero_grad()
            output = self.model(spike_data, time_steps=self.time_steps)
            
            # Compute loss using spike-based loss
            loss = criterion(output, target)
            
            # Add spike regularization
            spike_reg = sf.functional.spike_regularization(output, 'l1', 0.001)
            total_loss_with_reg = loss + spike_reg
            
            # Backward pass
            total_loss_with_reg.backward()
            optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            
            # Compute accuracy from spike counts
            spike_counts = output.sum(dim=0)  # Sum over time
            predicted = spike_counts.argmax(dim=1)
            correct += (predicted == target).sum().item()
            total += target.size(0)
            
            if batch_idx % 50 == 0:
                print(f'Epoch {epoch}, Batch {batch_idx}/{len(train_loader)}, '
                      f'Loss: {loss.item():.4f}, '
                      f'Acc: {100. * correct / total:.2f}%')
        
        epoch_loss = total_loss / len(train_loader)
        epoch_accuracy = 100. * correct / total
        
        return epoch_loss, epoch_accuracy
    
    def validate(self, 
                val_loader: DataLoader,
                criterion: callable) -> Tuple[float, float]:
        """Validate model performance"""
        
        self.model.eval()
        # Disable STDP learning during validation
        self.model.set_learning(False)
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                self.model.reset_state()
                
                # Convert to spikes
                if data.dim() == 4:
                    data = data.view(data.size(0), -1)
                
                # Measure inference time
                start_time = time.time()
                spike_data = sf.functional.poisson_encoding(data, self.time_steps)
                output = self.model(spike_data, time_steps=self.time_steps)
                inference_time = time.time() - start_time
                
                self.inference_times.append(inference_time)
                
                # Compute loss and accuracy
                loss = criterion(output, target)
                total_loss += loss.item()
                
                spike_counts = output.sum(dim=0)
                predicted = spike_counts.argmax(dim=1)
                correct += (predicted == target).sum().item()
                total += target.size(0)
        
        val_loss = total_loss / len(val_loader)
        val_accuracy = 100. * correct / total
        
        return val_loss, val_accuracy
    
    def train_complete(self,
                      train_loader: DataLoader,
                      val_loader: DataLoader,
                      epochs: int = 10,
                      learning_rate: float = 0.001) -> Dict[str, List[float]]:
        """Complete training pipeline"""
        
        print(f"🧠 Starting SNN training for {epochs} epochs...")
        print(f"Model: {sum(p.numel() for p in self.model.parameters()):,} parameters")
        print(f"Device: {self.device}")
        print(f"Time steps: {self.time_steps}")
        
        # Setup optimizer and loss
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        criterion = lambda output, target: sf.functional.spike_loss(output, target, 'rate')
        
        # Learning rate scheduler
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
        
        best_val_accuracy = 0.0
        best_model_state = None
        
        for epoch in range(epochs):
            print(f"\n--- Epoch {epoch+1}/{epochs} ---")
            
            # Training
            train_loss, train_acc = self.train_epoch(train_loader, optimizer, criterion, epoch)
            self.train_losses.append(train_loss)
            self.train_accuracies.append(train_acc)
            
            # Validation
            val_loss, val_acc = self.validate(val_loader, criterion)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_acc)
            
            # Update learning rate
            scheduler.step()
            
            print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
            print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
            
            # Save best model
            if val_acc > best_val_accuracy:
                best_val_accuracy = val_acc
                best_model_state = self.model.state_dict().copy()
                print(f'🎯 New best validation accuracy: {val_acc:.2f}%')
        
        # Load best model
        if best_model_state:
            self.model.load_state_dict(best_model_state)
        
        print(f"\n✅ Training completed!")
        print(f"Best validation accuracy: {best_val_accuracy:.2f}%")
        
        return {
            'train_losses': self.train_losses,
            'train_accuracies': self.train_accuracies,
            'val_losses': self.val_losses,
            'val_accuracies': self.val_accuracies
        }
    
    def visualize_training(self, save_path: str = None):
        """Visualize training progress"""
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        epochs = range(1, len(self.train_losses) + 1)
        
        # Loss curves
        ax1.plot(epochs, self.train_losses, 'b-', label='Training Loss', linewidth=2)
        ax1.plot(epochs, self.val_losses, 'r-', label='Validation Loss', linewidth=2)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Accuracy curves
        ax2.plot(epochs, self.train_accuracies, 'b-', label='Training Accuracy', linewidth=2)
        ax2.plot(epochs, self.val_accuracies, 'r-', label='Validation Accuracy', linewidth=2)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.set_title('Training and Validation Accuracy')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Inference time distribution
        if self.inference_times:
            ax3.hist(self.inference_times, bins=30, alpha=0.7, edgecolor='black')
            ax3.set_xlabel('Inference Time (s)')
            ax3.set_ylabel('Frequency')
            ax3.set_title('Inference Time Distribution')
            ax3.axvline(np.mean(self.inference_times), color='red', linestyle='--',
                       label=f'Mean: {np.mean(self.inference_times):.4f}s')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        # Model complexity metrics
        layer_sizes = []
        layer_names = []
        for name, module in self.model.named_modules():
            if hasattr(module, 'weight'):
                layer_sizes.append(module.weight.numel())
                layer_names.append(name.split('.')[-1][:10])  # Truncate name
        
        if layer_sizes:
            ax4.bar(range(len(layer_sizes)), layer_sizes)
            ax4.set_xlabel('Layer')
            ax4.set_ylabel('Parameters')
            ax4.set_title('Parameters per Layer')
            ax4.set_xticks(range(len(layer_names)))
            ax4.set_xticklabels(layer_names, rotation=45)
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def analyze_spikes(self, data_loader: DataLoader, n_samples: int = 5):
        """Analyze spike patterns from trained model"""
        
        self.model.eval()
        self.model.set_learning(False)
        
        with torch.no_grad():
            for i, (data, target) in enumerate(data_loader):
                if i >= n_samples:
                    break
                
                data = data.to(self.device)
                if data.dim() == 4:
                    data = data.view(data.size(0), -1)
                
                self.model.reset_state()
                spike_data = sf.functional.poisson_encoding(data[:1], self.time_steps)
                output = self.model(spike_data, time_steps=self.time_steps)
                
                # Visualize spike patterns
                fig = self.visualizer.plot_raster(
                    output[:, 0, :],  # First sample, all neurons
                    title=f'Output Spikes - Sample {i+1} (Class {target[0].item()})'
                )
                plt.show()


def main():
    """Main training script"""
    
    # Configuration
    config = {
        'batch_size': 32,
        'epochs': 15,
        'learning_rate': 0.001,
        'time_steps': 50,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    print("🧠⚡ SpikeFlow Complete Training Example")
    print(f"Configuration: {config}")
    
    # Create datasets
    print("\n📊 Loading datasets...")
    train_dataset = NMNIST(root='./data', train=True, download=True, time_window=config['time_steps'])
    val_dataset = NMNIST(root='./data', train=False, download=True, time_window=config['time_steps'])
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=0)
    
    # Create model
    print("\n🏗️ Creating SNN model...")
    model = sf.create_snn_classifier(
        input_size=34*34*2,  # N-MNIST dimensions
        hidden_sizes=[256, 128],
        output_size=10,
        backend=config['device'],
        neuron_type='LIF',
        synapse_type='STDP'
    )
    
    # Create trainer
    trainer = SNNTrainer(model, config['device'], config['time_steps'])
    
    # Train model
    print("\n🚀 Starting training...")
    training_history = trainer.train_complete(
        train_loader, val_loader, 
        epochs=config['epochs'],
        learning_rate=config['learning_rate']
    )
    
    # Visualize results
    print("\n📈 Generating visualizations...")
    trainer.visualize_training(save_path='training_results.png')
    
    # Analyze spike patterns
    print("\n🔍 Analyzing spike patterns...")
    trainer.analyze_spikes(val_loader, n_samples=3)
    
    # Edge optimization
    print("\n⚡ Optimizing for edge deployment...")
    edge_optimizer = EdgeOptimizer()
    optimized_model = edge_optimizer.optimize_for_edge(
        model,
        target_latency=5.0,   # 5ms
        target_memory=0.5,    # 0.5MB
        target_energy=50.0    # 50mJ
    )
    
    # Save models
    print("\n💾 Saving models...")
    torch.save(model.state_dict(), 'snn_model.pth')
    torch.save(optimized_model.state_dict(), 'snn_model_optimized.pth')
    
    print("\n✅ Training complete! Check 'training_results.png' for visualizations.")


if __name__ == "__main__":
    main()
