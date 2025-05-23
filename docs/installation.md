# Installation Guide

## Requirements

- Python 3.8 or higher
- PyTorch 2.0 or higher
- CUDA 11.0+ (optional, for GPU acceleration)

## Quick Installation

### Via pip (Recommended)

```bash
pip install spikeflow
```

### Via conda

```bash
conda install -c conda-forge spikeflow
```

### From Source (Development)

```bash
git clone https://github.com/JonusNattapong/SpikeFlow.git
cd SpikeFlow
pip install -e .
```

## Hardware-Specific Installation

### Intel Loihi Support

```bash
pip install spikeflow[loihi]
# Requires Intel Loihi SDK (nxsdk)
```

### SpiNNaker Support

```bash
pip install spikeflow[spinnaker]
# Requires SpiNNaker tools
```

### Complete Installation (All Features)

```bash
pip install spikeflow[all]
```

## Verification

Test your installation:

```python
import spikeflow as sf
print(f"SpikeFlow version: {sf.__version__}")

# Test basic functionality
model = sf.LIF(shape=10)
spikes = model(torch.randn(10))
print(f"✅ SpikeFlow installed successfully!")
```

## Docker Installation

```bash
docker pull spikeflow/spikeflow:latest
docker run -it spikeflow/spikeflow:latest python
```

## Troubleshooting

### Common Issues

**Issue**: CUDA out of memory
```bash
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
```

**Issue**: Import errors
```bash
pip install --upgrade torch torchvision
pip install --force-reinstall spikeflow
```

### Getting Help

If you encounter issues:
1. Check our [FAQ](faq.md)
2. Search [GitHub Issues](https://github.com/JonusNattapong/SpikeFlow/issues)
3. Join our [Discord Community](https://discord.gg/spikeflow)
