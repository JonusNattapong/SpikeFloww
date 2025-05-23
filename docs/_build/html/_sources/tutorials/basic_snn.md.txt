# Building Your First Spiking Neural Network

This tutorial will guide you through creating and training your first SNN using SpikeFlow.

## What You'll Learn

- Basic concepts of spiking neural networks
- How to create neurons and synapses
- Training with STDP (Spike-Timing Dependent Plasticity)
- Visualizing spike patterns

## Step 1: Understanding Spiking Neurons

Unlike traditional artificial neurons that output continuous values, spiking neurons communicate through discrete events called spikes.

```python
import spikeflow as sf
import torch
import matplotlib.pyplot as plt

# Create a Leaky Integrate-and-Fire (LIF) neuron
neuron = sf.LIF(
    shape=1,           # Single neuron
    tau_mem=20.0,      # Membrane time constant (ms)
    threshold=1.0,     # Spike threshold
    adaptation_strength=0.1  # Adaptation parameter
)

# Simulate neuron response
time_steps = 200
input_current = torch.ones(1) * 1.2  # Constant input

membrane_potentials = []
spikes = []

for t in range(time_steps):
    spike = neuron(input_current)
    membrane_potentials.append(neuron.membrane_potential.item())
    spikes.append(spike.item())
```

## Step 2: Creating Synaptic Connections

Synapses connect neurons and can learn through STDP:

```python
# Create STDP synapse
synapse = sf.STDPLinear(
    input_size=5,
    output_size=3,
    tau_plus=20.0,     # LTP time constant
    tau_minus=20.0,    # LTD time constant
    A_plus=0.01,       # LTP strength
    A_minus=0.01       # LTD strength
)

# Simulate synaptic learning
for step in range(100):
    pre_spikes = torch.rand(5) > 0.8
    output = synapse(pre_spikes.float())
    
    # Simulate post-synaptic response
    post_spikes = torch.rand(3) > 0.7
    
    # STDP learning occurs automatically
    _ = synapse(pre_spikes.float(), post_spikes, learning=True)
```

## Step 3: Building a Complete Network

```python
# Create a simple SNN classifier
model = sf.create_snn_classifier(
    input_size=784,      # MNIST input size
    hidden_sizes=[128, 64],
    output_size=10,
    neuron_type='LIF',
    synapse_type='STDP'
)

print(f"Created SNN with {sum(p.numel() for p in model.parameters()):,} parameters")
```

## Step 4: Data Encoding

Convert static data to spike trains:

```python
# Load sample data
data = torch.randn(32, 784)  # Batch of 32 samples

# Convert to Poisson spike trains
spike_data = sf.functional.poisson_encoding(
    data, 
    time_steps=100,
    max_rate=100.0
)

print(f"Spike data shape: {spike_data.shape}")  # (100, 32, 784)
```

## Step 5: Training the Network

```python
import torch.optim as optim

# Setup training
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = lambda output, target: sf.functional.spike_loss(output, target, 'rate')

# Training loop
model.train()
for epoch in range(5):
    model.reset_state()
    
    # Forward pass
    output = model(spike_data, time_steps=100)
    
    # Dummy targets for demonstration
    targets = torch.randint(0, 10, (32,))
    
    # Compute loss
    loss = criterion(output, targets)
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
```

## Step 6: Visualization

```python
from spikeflow.visualization import SpikeVisualizer

# Create visualizer
viz = SpikeVisualizer()

# Plot spike raster
fig = viz.plot_raster(
    output[:, 0, :],  # First sample output
    title="SNN Output Spikes"
)
plt.show()

# Plot spike histogram
fig = viz.plot_spike_histogram(output)
plt.show()
```

## Key Concepts Learned

1. **Temporal Dynamics**: SNNs process information over time
2. **Spike Encoding**: Converting data to spike patterns
3. **STDP Learning**: Biologically-inspired plasticity
4. **Sparse Computation**: Most neurons are silent most of the time

## Next Steps

- Try the [MNIST Classification Tutorial](mnist_tutorial.md)
- Learn about [Advanced Neuron Models](../user_guide/neurons.md)
- Explore [Hardware Deployment](../user_guide/deployment.md)

## Complete Code

```python
import spikeflow as sf
import torch
import matplotlib.pyplot as plt

# Create and train a simple SNN
def main():
    # 1. Create model
    model = sf.create_snn_classifier(784, [128, 64], 10)
    
    # 2. Prepare data
    data = torch.randn(32, 784)
    spike_data = sf.functional.poisson_encoding(data, 100)
    targets = torch.randint(0, 10, (32,))
    
    # 3. Train
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(10):
        model.reset_state()
        output = model(spike_data, time_steps=100)
        loss = sf.functional.spike_loss(output, targets, 'rate')
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if epoch % 2 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
    
    # 4. Visualize
    viz = sf.visualization.SpikeVisualizer()
    viz.plot_raster(output[:, 0, :])
    plt.show()

if __name__ == "__main__":
    main()
```
