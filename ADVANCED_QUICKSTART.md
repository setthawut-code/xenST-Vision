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

## Train (Running)

Make sure you are in the virtual environment:

```bash
source venv/bin/activate
```

Then run the training script:

```bash
python src/train_advanced.py
```

This will:
1. Load `morphology_mip.ome.tif` and `transcripts_full.csv` from `Xenium dataset file/`.
2. Generate image patches and gene expression targets.
3. Train the `convnext_base` model to predict gene expression from histology.
4. Save checkpoints to `checkpoints/`.


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
