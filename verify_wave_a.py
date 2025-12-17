
import sys
import os
import argparse
sys.path.append(os.path.abspath('src')) # adjust path to root/src

import torch
import yaml
from src.model import XenSTModel
from src.eval import per_gene_metrics
import numpy as np

# Parse args
parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='configs/experiment_multiscale.yaml', help='Path to config file')
args = parser.parse_args()

# Load Config
print(f"Loading config from {args.config}...")
# Assuming running from project root
with open(args.config, 'r') as f:
    config = yaml.safe_load(f)

print("Config Loaded:", config['model'])

# Instantiate Model
print("Instantiating model...")
model = XenSTModel(
    encoder_cfg={
        'backbone_name': config['model']['backbone_name'],
        'pretrained': False, # set False for speed in demo
        'out_dim': config['model']['out_dim']
    },
    embed_dim=config['model']['out_dim'],
    num_genes=50,
    use_gnn=config['model']['use_gnn']
)
print("Model instantiated successfully.")

# Dummy Forward Pass
print("Running dummy forward pass...")
batch_size = 4
dummy_img = torch.randn(batch_size, 3, 224, 224)
preds, cls_logits = model(dummy_img)

print(f"Preds shape: {preds.shape}")
print(f"Cls Logits shape: {cls_logits.shape}")

# Basic Eval Function Check
print("Checking metrics...")
y_true = np.random.rand(100, 50)
y_pred = np.random.rand(100, 50)
metrics = per_gene_metrics(y_true, y_pred)
print("Metrics calculated:")
print(metrics.head())
print("Verification Complete!")
