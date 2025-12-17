
import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
import time
import argparse
import yaml

# Add src to path
sys.path.append(os.path.abspath('src'))

from model import XenSTModel

def benchmark_batch_size(batch_size, config):
    print(f"\\nTesting batch size: {batch_size}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type != 'cuda':
        print("⚠️  CUDA not available! Benchmarking on CPU is meaningless for VRAM optimization.")
        return False

    try:
        # Create model
        model = XenSTModel(
            encoder_cfg={
                'backbone_name': config['model']['backbone_name'],
                'pretrained': False, # No need for weights for VRAM test
                'out_dim': config['model']['out_dim']
            },
            embed_dim=config['model']['out_dim'],
            num_genes=50,
            use_gnn=config['model']['use_gnn']
        ).to(device)
        
        optimizer = optim.Adam(model.parameters(), lr=1e-4)
        criterion = nn.MSELoss()
        scaler = torch.cuda.amp.GradScaler()
        
        model.train()
        
        # Helper to create batch
        def get_batch():
            return torch.randn(batch_size, 3, 224, 224).to(device), torch.randn(batch_size, 50).to(device)

        # Run a few steps to stabilize and trigger potential OOM
        print("  Running warmup steps...")
        for i in range(3):
            inputs, targets = get_batch()
            
            # AMP Context
            with torch.cuda.amp.autocast():
                outputs, _ = model(inputs)
                loss = criterion(outputs, targets)
            
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            del inputs, targets, outputs, loss
            torch.cuda.empty_cache()
            
        print(f"✅ Batch size {batch_size} passed.")
        return True
        
    except RuntimeError as e:
        if 'out of memory' in str(e):
            print(f"❌ Batch size {batch_size} failed: CUDA Out of Memory")
            torch.cuda.empty_cache()
            return False
        else:
            print(f"❌ Batch size {batch_size} failed with error: {e}")
            return False
    except Exception as e:
         print(f"❌ Batch size {batch_size} failed with error: {e}")
         return False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/config_low_vram.yaml')
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Test range
    batch_sizes = [4, 8, 16, 24, 32, 48, 64]
    max_safe_batch = 4
    
    print(f"Starting benchmark on {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print("This may check a while...")
    
    for bs in batch_sizes:
        success = benchmark_batch_size(bs, config)
        if success:
            max_safe_batch = bs
        else:
            print("Stopping benchmark as limit reached.")
            break
    
    print(f"\\n🏆 Maximum successful batch size: {max_safe_batch}")
    print(f"Recommendation: Set batch_size to {max_safe_batch} (or {max_safe_batch} - small buffer if unstable)")

if __name__ == "__main__":
    main()
