# Installation Guide for HierarchicalVLM

Complete step-by-step instructions for installing and setting up HierarchicalVLM.

## Table of Contents
- [System Requirements](#system-requirements)
- [Quick Install](#quick-install)
- [Detailed Installation](#detailed-installation)
- [GPU Setup](#gpu-setup)
- [Verification](#verification)
- [Troubleshooting](#troubleshooting)
- [Development Setup](#development-setup)

## System Requirements

### Minimum Requirements
- **Python**: 3.8 or higher
- **RAM**: 16 GB (32 GB recommended for multi-GPU training)
- **Disk Space**: 50 GB for datasets and models

### GPU Requirements (Optional but Recommended)
- **NVIDIA GPU**: CUDA Compute Capability 7.0 or higher
  - Recommended: RTX 3090, RTX 4090, or A100
  - Minimum tested: V100, RTX 2080 Ti
- **CUDA**: 11.8 or higher
- **cuDNN**: 8.6 or higher

### Operating System
- Linux (Ubuntu 20.04+ recommended)
- macOS (CPU only, limited GPU support)
- Windows (WSL2 recommended)

## Quick Install

For users who just want to use the code:

```bash
# Clone the repository
git clone https://github.com/arrdel/hierarchical-vlm.git
cd HierarchicalVLM

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install package with dependencies
pip install --upgrade pip setuptools wheel
pip install -e .
```

## Detailed Installation

### Step 1: Clone Repository

```bash
git clone https://github.com/arrdel/hierarchical-vlm.git
cd HierarchicalVLM
```

### Step 2: Create Virtual Environment

**Using venv (built-in):**
```bash
python3 -m venv venv
source venv/bin/activate  # Linux/macOS
# or
venv\Scripts\activate  # Windows
```

**Using conda (alternative):**
```bash
conda create -n hierarchicalvlm python=3.10
conda activate hierarchicalvlm
```

### Step 3: Install Dependencies

**Core dependencies:**
```bash
pip install --upgrade pip setuptools wheel
pip install torch>=2.0.0 torchvision>=0.15.0 --index-url https://download.pytorch.org/whl/cu118
```

**Additional packages:**
```bash
pip install -r requirements.txt
```

### Step 4: Install HierarchicalVLM

```bash
# Install in development mode (recommended for development)
pip install -e .

# Or install normally
pip install .
```

### Step 5: Verify Installation

```bash
python -c "import hierarchicalvlm; print('✓ HierarchicalVLM installed successfully')"
python -c "import torch; print(f'✓ PyTorch version: {torch.__version__}')"
python -c "import torch; print(f'✓ CUDA available: {torch.cuda.is_available()}')"
```

## GPU Setup

### NVIDIA CUDA Installation

**Ubuntu/Debian:**
```bash
# Install CUDA toolkit
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-ubuntu2004
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda-repo-ubuntu2004-11-8-local_11.8.0-520.61.05-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2004-11-8-local_11.8.0-520.61.05-1_amd64.deb
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub
sudo apt-get update
sudo apt-get -y install cuda-11-8

# Add to PATH
echo 'export PATH=/usr/local/cuda-11.8/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

**Verify CUDA installation:**
```bash
nvcc --version
nvidia-smi
```

### PyTorch with CUDA

```bash
# Install PyTorch with CUDA support
pip install torch>=2.0.0 torchvision>=0.15.0 --index-url https://download.pytorch.org/whl/cu118

# Verify
python -c "import torch; print(torch.cuda.is_available())"
```

## Verification

### Test Installation

Run the verification script:
```bash
python -c """
import torch
import hierarchicalvlm
from hierarchicalvlm.model import HierarchicalVLM

print('✓ PyTorch:', torch.__version__)
print('✓ CUDA available:', torch.cuda.is_available())
print('✓ HierarchicalVLM imported successfully')
"""
```

### Test with Sample Code

```python
import torch
from hierarchicalvlm.model import HierarchicalVLM

# Initialize model
model = HierarchicalVLM(
    hidden_dim=1024,
    num_layers=6,
    num_heads=8
)

# Test forward pass
batch_size, seq_len, feat_dim = 2, 250, 2048
features = torch.randn(batch_size, seq_len, feat_dim)

output = model(features)
print(f"Input shape: {features.shape}")
print(f"Output shape: {output.shape}")
print("✓ Model forward pass successful")
```

## Troubleshooting

### Issue: CUDA Not Available

**Symptom:** `torch.cuda.is_available()` returns False

**Solution:**
```bash
# Check GPU detection
nvidia-smi

# Reinstall PyTorch with CUDA support
pip uninstall torch torchvision
pip install torch>=2.0.0 torchvision>=0.15.0 --index-url https://download.pytorch.org/whl/cu118

# Verify
python -c "import torch; print(torch.cuda.is_available())"
```

### Issue: Import Error for hierarchicalvlm

**Symptom:** `ModuleNotFoundError: No module named 'hierarchicalvlm'`

**Solution:**
```bash
# Ensure you're in the project directory
cd /path/to/HierarchicalVLM

# Reinstall in development mode
pip install -e .

# Or add to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:/path/to/HierarchicalVLM"
```

### Issue: Out of Memory (OOM)

**Symptom:** CUDA out of memory error during training

**Solution:**
1. Reduce batch size in config:
   ```yaml
   batch_size: 8  # Reduce from default
   ```
2. Use gradient accumulation:
   ```yaml
   gradient_accumulation_steps: 2
   ```
3. Enable mixed precision:
   ```yaml
   mixed_precision: "fp16"
   ```

### Issue: Slow Performance on GPU

**Symptom:** Training is slow despite GPU availability

**Solution:**
```bash
# Check GPU memory utilization
nvidia-smi

# Verify PyTorch using GPU
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"

# Check for CPU bottlenecks
python -u train.py  # Add -u for unbuffered output
```

### Issue: Version Conflicts

**Symptom:** Dependency conflicts between packages

**Solution:**
```bash
# Fresh environment
deactivate
rm -rf venv
python3 -m venv venv
source venv/bin/activate

# Install specific versions
pip install --upgrade pip
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

## Development Setup

### Install Development Tools

```bash
pip install -e ".[dev]"
```

This installs additional tools:
- `pytest`: Testing framework
- `black`: Code formatter
- `isort`: Import sorter
- `flake8`: Linter

### Setup Pre-commit Hooks

```bash
pip install pre-commit
pre-commit install
```

### Run Tests

```bash
pytest tests/ -v
```

### Format Code

```bash
black hierarchicalvlm/
isort hierarchicalvlm/
flake8 hierarchicalvlm/
```

## Next Steps

After successful installation:

1. **Download Dataset**: See [Dataset Setup](docs/dataset_setup.md)
2. **Quick Start**: Check [Quick Start Guide](README.md#quick-start)
3. **Training**: Run [Training Tutorial](docs/training_guide.md)
4. **Evaluation**: See [Evaluation Guide](docs/evaluation.md)

## Getting Help

- Check [Troubleshooting Guide](docs/troubleshooting.md)
- Search [GitHub Issues](https://github.com/arrdel/hierarchical-vlm/issues)
- Read [FAQ](docs/FAQ.md)
- Open new [GitHub Discussion](https://github.com/arrdel/hierarchical-vlm/discussions)

---

**Last Updated:** December 2025
