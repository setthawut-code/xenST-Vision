# Verification Guide

## Quick Verification Steps

### 1. Installation Check

```bash
# Install all dependencies
pip install -r requirements.txt

# Verify PyTorch installation
python -c "import torch; print(f'PyTorch {torch.__version__} installed')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

**Expected Output:**
- PyTorch 2.0+ installed
- CUDA available: True (if GPU present)

### 2. Model Test

```bash
cd /Users/setthawut/ProjectXenium
python src/model.py
```

**Expected Output:**
```
Model created with 25,xxx,xxx trainable parameters
Input shape: torch.Size([4, 3, 224, 224])
Output shape: torch.Size([4, 50])
✓ Model test passed!
```

### 3. Training Script Check

```bash
# Dry run (will show data loading message)
python src/train.py --config config/config.yaml
```

**Expected Output:**
```
Using device: cuda/cpu
Model created with XX,XXX,XXX parameters
⚠️  Data loading not implemented yet - requires Xenium dataset
```

### 4. Web Interface Test

```bash
# Start interface
python web/app.py

# Should open on http://localhost:7860
```

**Expected Behavior:**
- Gradio interface loads
- Shows bilingual UI (Thai/English)
- Model dropdown shows "No models available" (until trained)
- Can upload images

### 5. Xenium Data Download (Optional)

Download sample dataset:
```bash
# Create data directory
mkdir -p data

# Download from 10x Genomics
# Example: Human breast cancer dataset
# https://www.10xgenomics.com/datasets/preview-data-ffpe-human-breast-cancer-with-xenium-multimodal-cell-segmentation-1-standard
```

## Full Training Verification

Once you have Xenium data:

```bash
# 1. Train model
python src/train.py --config config/config.yaml

# 2. Monitor training
tensorboard --logdir logs/

# 3. Check output
ls checkpoints/
# Should show: best_model.pth, latest_model.pth
```

## Inference Verification

```bash
# Test inference with trained model
python src/inference.py \
    --image path/to/test_image.png \
    --checkpoint checkpoints/best_model.pth \
    --output output/test_results

# Check outputs
ls output/test_results/
# Should show: predictions.csv, spatial_heatmaps.png, interactive_*.html
```

## Common Issues & Solutions

### Issue 1: ModuleNotFoundError
**Solution:** Install requirements
```bash
pip install -r requirements.txt
```

### Issue 2: CUDA out of memory
**Solution:** Reduce batch size in config.yaml
```yaml
training:
  batch_size: 16  # or 8
```

### Issue 3: No Xenium data
**Solution:** Download from 10x Genomics or use your own dataset

---

**All verification steps documented in walkthrough.md**
