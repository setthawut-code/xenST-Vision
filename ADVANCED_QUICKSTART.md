# Advanced Architecture - Quick Start Guide

## Installation

```bash
cd /Users/setthawut/ProjectXenium

# Activate virtual environment
source venv/bin/activate

# Install timm
pip install timm

# Install scipy (for metrics)
pip install scipy
```

## Test Model

```bash
# In venv
python src/advanced_model.py
```

## Train (when data is ready)

```bash
python src/train_advanced.py
```

## Configuration

Edit `config/config.yaml`:

```yaml
model:
  type: advanced
  backbone: convnext_base  # or resnet50, swin_base_patch4_window7_224
  num_genes: 50
  
loss:
  type: negative_binomial  # or mse, hybrid
```

## Files Added

- `src/losses.py` - NB loss implementation
- `src/advanced_model.py` - timm backbone models
- `src/train_advanced.py` - Advanced training loop

## Status

✅ Core features implemented
⚠️ Requires PyTorch environment to run
📦 Ready for training with Xenium data
