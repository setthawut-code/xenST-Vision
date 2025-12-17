"""
Training script for spatial transcriptomics prediction
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import numpy as np
from tqdm import tqdm
import argparse

# Add src to path
sys.path.append(str(Path(__file__).parent))

from model import create_model, count_parameters
from data_preprocessing import get_transforms
from dataset import XeniumDataset
from torch.utils.data import random_split
from utils import (
    load_config,
    setup_directories,
    calculate_pearson_correlation,
    plot_correlation_distribution,
    AverageMeter
)


class Trainer:
    """Training manager for spatial transcriptomics model"""
    
    def __init__(self, config: dict, model: nn.Module, device: str):
        self.config = config
        self.model = model.to(device)
        self.device = device
        
        # Setup directories
        setup_directories(config)
        
        # Optimizer and scheduler
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay']
        )
        
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )
        
        # Loss function
        self.criterion = nn.MSELoss()
        
        # Mixed precision training
        self.use_amp = config['training']['mixed_precision']
        self.scaler = torch.cuda.amp.GradScaler() if self.use_amp else None
        
        # TensorBoard
        self.writer = SummaryWriter(config['paths']['logs_dir'])
        
        # Tracking
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.current_epoch = 0
    
    def train_epoch(self, train_loader: DataLoader) -> dict:
        """Train for one epoch"""
        self.model.train()
        
        losses = AverageMeter()
        correlations_meter = AverageMeter()
        
        pbar = tqdm(train_loader, desc=f"Epoch {self.current_epoch}")
        
        for images, targets in pbar:
            images = images.to(self.device)
            targets = targets.to(self.device)
            
            # Forward pass with mixed precision
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    outputs = self.model(images)
                    if isinstance(outputs, tuple):
                        outputs = outputs[0]
                    loss = self.criterion(outputs, targets)
                
                # Backward pass
                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                
                # Gradient clipping
                if self.config['training']['grad_clip'] > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config['training']['grad_clip']
                    )
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, targets)
                
                self.optimizer.zero_grad()
                loss.backward()
                
                if self.config['training']['grad_clip'] > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config['training']['grad_clip']
                    )
                
                self.optimizer.step()
            
            # Calculate correlation
            with torch.no_grad():
                pred_np = outputs.cpu().numpy()
                target_np = targets.cpu().numpy()
                corrs = calculate_pearson_correlation(pred_np, target_np)
                mean_corr = np.mean(corrs[~np.isnan(corrs)])
            
            # Update meters
            losses.update(loss.item(), images.size(0))
            correlations_meter.update(mean_corr, images.size(0))
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{losses.avg:.4f}',
                'corr': f'{correlations_meter.avg:.3f}'
            })
        
        metrics = {
            'loss': losses.avg,
            'correlation': correlations_meter.avg
        }
        
        return metrics
    
    def validate(self, val_loader: DataLoader) -> dict:
        """Validate model"""
        self.model.eval()
        
        losses = AverageMeter()
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for images, targets in tqdm(val_loader, desc="Validating"):
                images = images.to(self.device)
                targets = targets.to(self.device)
                
                outputs = self.model(images)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                loss = self.criterion(outputs, targets)
                
                losses.update(loss.item(), images.size(0))
                
                all_predictions.append(outputs.cpu().numpy())
                all_targets.append(targets.cpu().numpy())
        
        # Concatenate all predictions
        predictions = np.concatenate(all_predictions, axis=0)
        targets = np.concatenate(all_targets, axis=0)
        
        # Calculate per-gene correlations
        correlations = calculate_pearson_correlation(predictions, targets)
        
        metrics = {
            'loss': losses.avg,
            'correlation_mean': np.mean(correlations[~np.isnan(correlations)]),
            'correlation_median': np.median(correlations[~np.isnan(correlations)]),
            'correlations': correlations
        }
        
        return metrics
    
    def save_checkpoint(self, metrics: dict, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint_dir = Path(self.config['paths']['checkpoints_dir'])
        
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'metrics': metrics,
            'config': self.config
        }
        
        # Save latest
        latest_path = checkpoint_dir / 'latest_model.pth'
        torch.save(checkpoint, latest_path)
        
        # Save best
        if is_best:
            best_path = checkpoint_dir / 'best_model.pth'
            torch.save(checkpoint, best_path)
            print(f"✓ Saved best model with val_loss={metrics['loss']:.4f}")
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader):
        """Main training loop"""
        print(f"\\n{'='*50}")
        print(f"Starting training for {self.config['training']['epochs']} epochs")
        print(f"Model parameters: {count_parameters(self.model):,}")
        print(f"Device: {self.device}")
        print(f"{'='*50}\\n")
        
        for epoch in range(self.config['training']['epochs']):
            self.current_epoch = epoch + 1
            
            # Train
            train_metrics = self.train_epoch(train_loader)
            
            # Validate
            val_metrics = self.validate(val_loader)
            
            # Log to TensorBoard
            self.writer.add_scalar('Loss/train', train_metrics['loss'], epoch)
            self.writer.add_scalar('Loss/val', val_metrics['loss'], epoch)
            self.writer.add_scalar('Correlation/train', train_metrics['correlation'], epoch)
            self.writer.add_scalar('Correlation/val', val_metrics['correlation_mean'], epoch)
            self.writer.add_scalar('LR', self.optimizer.param_groups[0]['lr'], epoch)
            
            # Update learning rate
            self.scheduler.step(val_metrics['loss'])
            
            # Print epoch summary
            print(f"\\nEpoch {epoch+1}/{self.config['training']['epochs']}")
            print(f"  Train Loss: {train_metrics['loss']:.4f} | Train Corr: {train_metrics['correlation']:.3f}")
            print(f"  Val Loss: {val_metrics['loss']:.4f} | Val Corr: {val_metrics['correlation_mean']:.3f}")
            
            # Save checkpoint
            is_best = val_metrics['loss'] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_metrics['loss']
                self.patience_counter = 0
            else:
                self.patience_counter += 1
            
            self.save_checkpoint(val_metrics, is_best)
            
            # Early stopping
            if self.patience_counter >= self.config['training']['early_stopping_patience']:
                print(f"\\nEarly stopping triggered after {epoch+1} epochs")
                break
        
        self.writer.close()
        print(f"\\nTraining completed! Best val loss: {self.best_val_loss:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='Path to config file')
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    

    
    # Load Config Paths
    paths = config.get('paths', {})
    image_path = paths.get('image_path')
    transcripts_path = paths.get('transcripts_path')
    
    if not image_path or not os.path.exists(image_path):
        print(f"\\nError: Image path not found: {image_path}")
        return

    print(f"\\nInitializing Dataset from: {image_path}")
    
    # Instantiate Dataset
    full_dataset = XeniumDataset(
        image_path=image_path,
        transcripts_path=transcripts_path,
        patch_size=config['data']['tile_size'],
        stride=config['data']['tile_size'], # Non-overlapping for now
        min_counts=10, 
        preload_image=False
    )
    
    print(f"Total patches: {len(full_dataset)}")
    
    # Update config with actual number of genes
    config['model']['num_genes'] = full_dataset.num_genes
    print(f"Dataset has {full_dataset.num_genes} genes. Updating model config.")
    
    # Create model
    model = create_model(config)
    print(f"Model created with {count_parameters(model):,} parameters")
    
    # Split Data
    train_size = int(config['data']['train_ratio'] * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(config['training']['seed']))
    
    print(f"Train size: {len(train_dataset)}, Val size: {len(val_dataset)}")
    
    # Create DataLoaders
    # Note: XeniumDataset returns raw numpy/PIL, so we need transforms here? 
    # Actually XeniumDataset __getitem__ applies self.transform if provided.
    # But we initialized it without transform. Let's fix that wrapper or apply transform via Collate or just wrap it?
    # Simpler: We can inject transforms into the subset if we hack it, OR (better) pass transform to XeniumDataset and let it handle IT.
    # But XeniumDataset takes one transform. We usually want different transforms for train/val.
    # Standard pattern: Create two dataset instances or use a wrapper. 
    # For now, let's keep it simple: No transforms (DL will receive tensors if dataset does basic conversion).
    # Checking dataset.py: __getitem__ returns (image, expression). 
    # Image is converted to Tensor if transform is None? - CHECK CODE. 
    # dataset.py line 208: if self.transform: image = self.transform(image). 
    # If not transform, it returns PIL? No line 206 converts to PIL. 
    # If no transform, it might fail to convert to tensor if not handled.
    # Let's check dataset.py again. 
    
    # To be safe and quick: Let's assume we want basic ToTensor at least.
    # train.py imports get_transforms.
    
    # Let's re-instantiate or wrap. 
    # Option: Modify dataset class to allow setter? No.
    # Option: Custom Collate? 
    # Option: Just use the same transform for both for now (basic normalization).
    
    common_transform = get_transforms(config, is_training=False) # Safe bet
    full_dataset.transform = common_transform 

    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['training']['batch_size'], 
        shuffle=True, 
        num_workers=config['training'].get('num_workers', 0),
        pin_memory=True,
        persistent_workers=(config['training'].get('num_workers', 0) > 0)
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config['training']['batch_size'],
        shuffle=False, 
        num_workers=config['training'].get('num_workers', 0),
        pin_memory=True,
        persistent_workers=(config['training'].get('num_workers', 0) > 0)
    )
    
    trainer = Trainer(config, model, device)
    trainer.train(train_loader, val_loader)


if __name__ == "__main__":
    main()
