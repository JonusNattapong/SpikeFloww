"""
Unit tests for SpikeFlow core functionality
"""

import pytest
import torch
import sys
from pathlib import Path

# Add spikeflow to path
sys.path.append(str(Path(__file__).parent.parent))
import spikeflow as sf

class TestNeurons:
    """Test neuron models"""
    
    def test_lif_neuron_creation(self):
        """Test LIF neuron creation"""
        neuron = sf.LIF(shape=10, tau_mem=20.0, threshold=1.0)
        
        assert neuron.shape == (10,)
        assert neuron.tau_mem == 20.0
        assert neuron.threshold == 1.0
        assert neuron.membrane_potential.shape == (10,)
    
    def test_lif_neuron_forward(self):
        """Test LIF neuron forward pass"""
        neuron = sf.LIF(shape=5)
        input_current = torch.randn(5)
        
        spikes = neuron(input_current)
        
        assert spikes.shape == (5,)
        assert torch.all((spikes == 0) | (spikes == 1))  # Binary spikes
    
    def test_izhikevich_neuron(self):
        """Test Izhikevich neuron"""
        neuron = sf.Izhikevich(shape=3, a=0.02, b=0.2, c=-65.0, d=8.0)
        input_current = torch.ones(3) * 50.0  # Strong input
        
        spikes = neuron(input_current)
        
        assert spikes.shape == (3,)
        assert torch.all((spikes == 0) | (spikes == 1))
    
    def test_neuron_reset(self):
        """Test neuron state reset"""
        neuron = sf.LIF(shape=5)
        
        # Activate neuron
        neuron(torch.ones(5) * 2.0)  # Suprathreshold input
        
        # Check state is non-zero
        assert torch.any(neuron.membrane_potential != 0) or torch.any(neuron.spike_history != 0)
        
        # Reset
        neuron.reset_state()
        
        # Check state is reset
        assert torch.all(neuron.membrane_potential == 0)
        assert torch.all(neuron.spike_history == 0)

class TestSynapses:
    """Test synapse models"""
    
    def test_stdp_synapse_creation(self):
        """Test STDP synapse creation"""
        synapse = sf.STDPLinear(input_size=5, output_size=3)
        
        assert synapse.input_size == 5
        assert synapse.output_size == 3
        assert synapse.weight.shape == (3, 5)
    
    def test_stdp_forward(self):
        """Test STDP synapse forward pass"""
        synapse = sf.STDPLinear(input_size=4, output_size=2)
        pre_spikes = torch.rand(4)
        
        output = synapse(pre_spikes)
        
        assert output.shape == (1, 2)  # Batch dimension added
    
    def test_stdp_learning(self):
        """Test STDP learning"""
        synapse = sf.STDPLinear(input_size=3, output_size=2, A_plus=0.1, A_minus=0.1)
        
        # Store initial weights
        initial_weights = synapse.weight.data.clone()
        
        # Apply correlated pre/post spikes
        pre_spikes = torch.ones(3)
        post_spikes = torch.ones(2)
        
        # Forward with learning
        _ = synapse(pre_spikes, post_spikes, learning=True)
        
        # Weights should change
        assert not torch.allclose(initial_weights, synapse.weight.data)

class TestNetworks:
    """Test network construction"""
    
    def test_sequential_creation(self):
        """Test sequential network creation"""
        model = sf.Sequential(
            sf.STDPLinear(10, 5),
            sf.LIF(5),
            sf.STDPLinear(5, 2),
            sf.LIF(2)
        )
        
        assert len(model.layers) == 4
    
    def test_snn_classifier_creation(self):
        """Test SNN classifier creation"""
        model = sf.create_snn_classifier(
            input_size=784,
            hidden_sizes=[128, 64], 
            output_size=10
        )
        
        # Check model structure
        assert len(model.layers) == 6  # 3 synapses + 3 neurons
        
        # Check parameter count
        total_params = sum(p.numel() for p in model.parameters())
        expected_params = (784 * 128) + (128 * 64) + (64 * 10)
        assert total_params >= expected_params
    
    def test_network_forward(self):
        """Test network forward pass"""
        model = sf.create_snn_classifier(100, [50], 10)
        input_data = torch.randn(4, 100)  # Batch of 4
        
        output = model(input_data, time_steps=20)
        
        assert output.shape == (20, 4, 10)  # (time, batch, features)

class TestFunctional:
    """Test functional utilities"""
    
    def test_poisson_encoding(self):
        """Test Poisson encoding"""
        data = torch.rand(2, 10)  # 2 samples, 10 features
        spike_data = sf.functional.poisson_encoding(data, time_steps=50)
        
        assert spike_data.shape == (50, 2, 10)
        assert torch.all((spike_data == 0) | (spike_data == 1))
    
    def test_spike_loss(self):
        """Test spike loss functions"""
        output = torch.rand(30, 4, 10)  # (time, batch, classes)
        target = torch.randint(0, 10, (4,))
        
        # Test different loss types
        loss_rate = sf.functional.spike_loss(output, target, 'rate')
        loss_temporal = sf.functional.spike_loss(output, target, 'temporal')
        
        assert loss_rate.item() >= 0
        assert loss_temporal.item() >= 0
    
    def test_spike_regularization(self):
        """Test spike regularization"""
        spikes = torch.rand(20, 5, 10)
        
        reg_l1 = sf.functional.spike_regularization(spikes, 'l1', 0.01)
        reg_l2 = sf.functional.spike_regularization(spikes, 'l2', 0.01)
        
        assert reg_l1.item() >= 0
        assert reg_l2.item() >= 0

class TestDatasets:
    """Test dataset functionality"""
    
    def test_nmnist_dataset(self):
        """Test N-MNIST dataset"""
        dataset = sf.datasets.NMNIST(
            root='./test_data',
            train=True,
            download=True,
            time_window=50
        )
        
        assert len(dataset) > 0
        
        # Test data loading
        spike_data, label = dataset[0]
        assert spike_data.shape[0] == 50  # time_window
        assert isinstance(label, int)
        assert 0 <= label <= 9

# Test configuration
@pytest.fixture
def device():
    """Test device fixture"""
    return 'cuda' if torch.cuda.is_available() else 'cpu'

@pytest.fixture
def sample_model():
    """Sample SNN model fixture"""
    return sf.create_snn_classifier(784, [128], 10)

def test_device_compatibility(device):
    """Test device compatibility"""
    model = sf.create_snn_classifier(100, [50], 10)
    input_data = torch.randn(1, 100)
    
    if device == 'cuda':
        model = model.cuda()
        input_data = input_data.cuda()
    
    output = model(input_data, time_steps=10)
    assert output.device.type == device

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
