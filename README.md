# Spatial Transcriptomics Prediction from H&E Images

ระบบทำนายการแสดงออกของยีนในเชิงพื้นที่จากภาพ H&E โดยใช้ Deep Learning 🧬

Spatial gene expression prediction from H&E histology images using Deep Learning, trained with Xenium data.

## Features / คุณสมบัติ

- 🖼️ **H&E Image Input** - Upload standard histology images
- 🧬 **Gene Expression Prediction** - Predict 50+ genes simultaneously  
- 🗺️ **Spatial Mapping** - Visualize expression across tissue
- 📊 **Interactive Plots** - Explore results with Plotly
- 🌐 **Bilingual Interface** - Thai/English web UI
- 💾 **Export Results** - Download predictions as CSV

## Quick Start / เริ่มต้นใช้งาน

### Installation / การติดตั้ง

```bash
# Clone repository
git clone <your-repo-url>
cd ProjectXenium

# Install dependencies
pip install -r requirements.txt
```

### Training / การเทรนโมเดล

```bash
# 1. Prepare Xenium data
# Download from: https://www.10xgenomics.com/datasets
# Place in data/ directory

# 2. Preprocess data
python src/prepare_data.py --xenium-dir data/xenium_output --output data/processed

# 3. Train model
python src/train.py --config config/config.yaml

# Monitor training
tensorboard --logdir logs/
```

### Web Interface / เว็บอินเตอร์เฟซ

```bash
# Start web app
python web/app.py

# With public sharing
python web/app.py --share

# Open browser to http://localhost:7860
```

### Inference / การใช้งานโมเดล

```bash
# Command line prediction
python src/inference.py \
    --image path/to/he_image.png \
    --checkpoint checkpoints/best_model.pth \
    --output output/results
```

## Model Architecture / สถาปัตยกรรมโมเดล

**Baseline Model:**
- **Encoder**: ResNet50 (pretrained on ImageNet)
- **Decoder**: 3-layer MLP regression head
- **Input**: 224×224 RGB H&E patches
- **Output**: Gene expression vector (50 genes)
- **Loss**: Mean Squared Error (MSE)

**Performance:**
- Expected Pearson correlation: 0.3-0.5
- Training time: 6-12 hours (GPU)  
- Inference: ~1-3 seconds per patch

## Project Structure / โครงสร้างโปรเจค

```
ProjectXenium/
├── config/
│   └── config.yaml          # Configuration
├── src/
│   ├── model.py            # Model architecture
│   ├── data_preprocessing.py   # Data loading
│   ├── train.py            # Training script
│   ├── inference.py        # Inference script
│   └── utils.py            # Utilities
├── web/
│   └── app.py              # Gradio web interface
├── data/                   # Data directory
├── checkpoints/            # Trained models
├── logs/                   # TensorBoard logs
├── output/                 # Prediction results
├── requirements.txt
└── README.md
```

## Configuration / การตั้งค่า

Edit `config/config.yaml` to customize:
- Number of genes to predict
- Training hyperparameters
- Data augmentation
- Model architecture

## Data / ข้อมูล

### Xenium Data Format

Expected structure:
```
xenium_output/
├── cell_feature_matrix.h5  # Gene expression
├── cells.csv.gz             # Cell locations
├── transcripts.csv.gz       # Transcripts
└── morphology.ome.tif       # H&E image
```

### Public Datasets

Download from 10x Genomics:
- Human breast cancer (FFPE)
- Mouse brain
- Human lung cancer

Link: https://www.10xgenomics.com/datasets

## Web Interface Screenshots

The bilingual web interface provides:
1. **Image Upload** - Drag & drop H&E images
2. **Model Selection** - Choose trained checkpoint
3. **Gene Selection** - Pick gene to visualize
4. **Interactive Map** - Plotly spatial heatmap
5. **Download** - Export predictions as CSV

## Requirements / ความต้องการระบบ

**Hardware:**
- GPU with ≥16GB VRAM (24GB+ recommended for training)
- 32GB+ RAM
- 50GB+ storage

**Software:**
- Python 3.8+
- CUDA 11.0+ (for GPU)
- PyTorch 2.0+

## Troubleshooting / การแก้ปัญหา

**Model not found:**
```bash
# Train model first
python src/train.py --config config/config.yaml
```

**Out of memory:**
```yaml
# Reduce batch size in config.yaml
training:
  batch_size: 16  # Try 8 or 4
```

**No GPU available:**
```yaml
# Use CPU in config.yaml
inference:
  device: "cpu"
```

## Citation / การอ้างอิง

If you use this code, please cite:
```bibtex
@software{spatial_transcriptomics_prediction,
  title={Spatial Transcriptomics Prediction from H\&E Images},
  author={Your Name},
  year={2025}
}
```

## License / ใบอนุญาต

MIT License

## Acknowledgments / กิตติกรรมประกาศ

- Built with PyTorch, Gradio, Scanpy
- Inspired by DeepSpot, CarHE, and Hist2ST
- Xenium data from 10x Genomics

---

**Made with ❤️ for computational pathology & spatial biology**
