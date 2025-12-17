"""
Advanced training script with Negative Binomial loss and timm backbones
Supports flexible model selection and comprehensive metrics
"""

import os
import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import numpy as np
from tqdm import tqdm

# from model import SpatialTranscriptomicsPredictor, create_model
from advanced_model import AdvancedSpatialPredictor, create_advanced_model, count_parameters
from losses import get_loss_function, compute_per_gene_metrics
from utils import load_config, setup_directories
from dataset import XeniumDataset
from torch.utils.data import random_split


class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class AdvancedTrainer:
    """Advanced trainer with Negative Binomial loss and comprehensive metrics"""
    
    def __init__(self, config: dict, model: nn.Module, device: str):
        """
        Args:
            config: Configuration dictionary
            model: Model instance (baseline or advanced)
            device: Device to train on
        """
        self.config = config
        self.model = model.to(device)
        self.device = device
        
        # Optimizer
        lr = config['training']['learning_rate']
        weight_decay = config['training'].get('weight_decay', 0.01)
        self.optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=config['training'].get('lr_patience', 10)
        )
        
        # Loss function
        loss_config = config.get('loss', {'type': 'negative_binomial'})
        self.criterion = get_loss_function(loss_config)
        self.loss_type = loss_config.get('type', 'negative_binomial')
        
        # Mixed precision training
        self.use_amp = config['training'].get('mixed_precision', True)
        self.scaler = torch.cuda.amp.GradScaler() if self.use_amp else None
        
        # Gradient clipping
        self.max_grad_norm = config['training'].get('max_grad_norm', 1.0)
        
        # Tensorboard
        log_dir = Path(config['paths']['logs_dir']) / config['training'].get('experiment_name', 'advanced_model')
        self.writer = SummaryWriter(str(log_dir))
        
        # Checkpointing
        self.checkpoint_dir = Path(config['paths']['checkpoints_dir'])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Early stopping
        self.best_val_loss = float('inf')
        self.best_val_pearson = -float('inf')
        self.patience = config['training'].get('early_stopping_patience', 15)
        self.patience_counter = 0
        
        # Tracking
        self.current_epoch = 0
        self.gene_names = None
    
    def train_epoch(self, train_loader: DataLoader) -> dict:
        """Train for one epoch"""
        self.model.train()
        
        losses = AverageMeter()
        
        pbar = tqdm(train_loader, desc=f"Epoch {self.current_epoch} [Train]")
        
        for batch_idx, (images, targets) in enumerate(pbar):
            images = images.to(self.device).float()
            targets = targets.to(self.device).float()
            
            self.optimizer.zero_grad()
            
            # Forward pass with mixed precision
            if self.use_amp:
                # Use standard torch.amp.autocast for PyTorch 2.0+ support on cuda/cpu/mps
                device_type = 'cuda' if self.device == 'cuda' else 'cpu'
                if self.device == 'mps':
                     # MPS autocast support might be limited or require 'cpu' fallback in some versions
                     # But let's try 'cpu' if 'mps' fails or just disable if mixed precision is tricky
                     device_type = 'cpu' # Safer fallback for stability, or 'mps' if supported in this torch version
                
                # Actually, torch.amp.autocast('mps') is supported in nightly/recent builds.
                # If user wants MPS, let's try it, but standard might be CPU for safety if warning persists.
                # Given user context, let's use 'cpu' for autocast context if not cuda, to avoid the specific "cuda not available" warning.
                # Or better: check device.
                
                dtype = torch.float16 if self.device == 'cuda' else torch.bfloat16
                # MPS often prefers bfloat16 or float16 depending on arch.
                
                # Simplified fix:
                device_type = self.device if self.device in ['cuda', 'cpu'] else 'cpu' 
                # Note: MPS autocast is 'mps' in newer versions, but 'cpu' is safe fallback for logic
                
                with torch.amp.autocast(device_type=device_type):
                    if self.loss_type in ['negative_binomial', 'hybrid']:
                        mu, theta = self.model(images)
                        loss = self.criterion(mu, theta, targets)
                    else:  # MSE
                        output = self.model(images) if hasattr(self.model, 'fc') else self.model.predict_mean(images)
                        loss = self.criterion(output, targets)
                
                # Backward pass
                self.scaler.scale(loss).backward()
                
                # Gradient clipping
                if self.max_grad_norm > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # Standard precision
                if self.loss_type in ['negative_binomial', 'hybrid']:
                    mu, theta = self.model(images)
                    loss = self.criterion(mu, theta, targets)
                else:
                    output = self.model(images) if hasattr(self.model, 'fc') else self.model.predict_mean(images)
                    loss = self.criterion(output, targets)
                
                loss.backward()
                
                if self.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                
                self.optimizer.step()
            
            losses.update(loss.item(), images.size(0))
            pbar.set_postfix({'loss': f'{losses.avg:.4f}'})
        
        return {'loss': losses.avg}
    
    @torch.no_grad()
    def validate(self, val_loader: DataLoader) -> dict:
        """Validate model"""
        self.model.eval()
        
        all_predictions = []
        all_targets = []
        val_loss = AverageMeter()
        
        pbar = tqdm(val_loader, desc=f"Epoch {self.current_epoch} [Val]")
        
        for images, targets in pbar:
            images = images.to(self.device).float()
            targets = targets.to(self.device).float()
            
            # Forward pass
            if self.loss_type in ['negative_binomial', 'hybrid']:
                mu, theta = self.model(images)
                loss = self.criterion(mu, theta, targets)
                predictions = mu
            else:
                output = self.model(images) if hasattr(self.model, 'fc') else self.model.predict_mean(images)
                loss = self.criterion(output, targets)
                predictions = output
            
            val_loss.update(loss.item(), images.size(0))
            
            all_predictions.append(predictions.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
            
            pbar.set_postfix({'loss': f'{val_loss.avg:.4f}'})
        
        # Concatenate all predictions and targets
        predictions = np.vstack(all_predictions)
        targets = np.vstack(all_targets)
        
        # Compute comprehensive metrics
        metrics = compute_per_gene_metrics(predictions, targets)
        metrics['loss'] = val_loss.avg
        
        return metrics
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint"""
        print(f"Loading checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_val_pearson = checkpoint.get('metrics', {}).get('pearson_median', -float('inf'))
        self.best_val_loss = checkpoint.get('metrics', {}).get('loss', float('inf'))
        print(f"✓ Resumed from epoch {self.current_epoch} (Best Pearson: {self.best_val_pearson:.4f})")
        return self.current_epoch

    def save_checkpoint(self, metrics: dict, is_best: bool = False, is_latest: bool = True, filename: str = None):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics,
            'config': self.config,
            'gene_names': self.gene_names
        }
        
        if filename:
            path = self.checkpoint_dir / filename
            torch.save(checkpoint, path)
            print(f"✓ Saved checkpoint: {filename}")
            return

        if is_latest:
            path = self.checkpoint_dir / 'latest_model.pth'
            torch.save(checkpoint, path)
        
        if is_best:
            path = self.checkpoint_dir / 'best_model.pth'
            torch.save(checkpoint, path)
            print(f"✓ Saved best model (Pearson: {metrics['pearson_median']:.4f})")
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader, gene_names: list = None, start_epoch: int = 1):
        """Main training loop"""
        self.gene_names = gene_names
        num_epochs = self.config['training']['epochs']
        
        print(f"\n{'='*60}")
        print(f"Starting Training")
        print(f"{'='*60}")
        print(f"Model: {self.model.__class__.__name__}")
        print(f"Parameters: {count_parameters(self.model):,}")
        print(f"Loss: {self.loss_type}")
        print(f"Device: {self.device}")
        print(f"Epochs: {num_epochs}")
        if start_epoch > 1:
            print(f"Resuming from epoch {start_epoch}")
        print(f"{'='*60}\n")
        
        try:
            for epoch in range(start_epoch, num_epochs + 1):
                self.current_epoch = epoch
                
                # Train
                train_metrics = self.train_epoch(train_loader)
                
                # Validate
                val_metrics = self.validate(val_loader)
                
                # Learning rate scheduling
                self.scheduler.step(val_metrics['loss'])
                
                # Logging
                self.writer.add_scalar('Loss/train', train_metrics['loss'], epoch)
                self.writer.add_scalar('Loss/val', val_metrics['loss'], epoch)
                self.writer.add_scalar('Metrics/pearson_median', val_metrics['pearson_median'], epoch)
                self.writer.add_scalar('Metrics/pearson_mean', val_metrics['pearson_mean'], epoch)
                self.writer.add_scalar('Metrics/spearman_median', val_metrics['spearman_median'], epoch)
                self.writer.add_scalar('Metrics/rmse_mean', val_metrics['rmse_mean'], epoch)
                self.writer.add_scalar('LR', self.optimizer.param_groups[0]['lr'], epoch)
                
                # Print progress
                print(f"\nEpoch {epoch}/{num_epochs}")
                print(f"  Train Loss: {train_metrics['loss']:.4f}")
                print(f"  Val Loss: {val_metrics['loss']:.4f}")
                print(f"  Pearson (median): {val_metrics['pearson_median']:.4f}")
                print(f"  Pearson (mean): {val_metrics['pearson_mean']:.4f}")
                print(f"  Spearman (median): {val_metrics['spearman_median']:.4f}")
                print(f"  RMSE (mean): {val_metrics['rmse_mean']:.4f}")
                print(f"  LR: {self.optimizer.param_groups[0]['lr']:.2e}")
                
                # Save checkpoints
                is_best = val_metrics['pearson_median'] > self.best_val_pearson
                self.save_checkpoint(val_metrics, is_best=is_best, is_latest=True)
                
                # Early stopping
                if val_metrics['loss'] < self.best_val_loss:
                    self.best_val_loss = val_metrics['loss']
                    self.patience_counter = 0
                else:
                    self.patience_counter += 1
                
                if val_metrics['pearson_median'] > self.best_val_pearson:
                    self.best_val_pearson = val_metrics['pearson_median']
                
                if self.patience_counter >= self.patience:
                    print(f"\nEarly stopping triggered (patience={self.patience})")
                    break
                    
        except KeyboardInterrupt:
            print("\n\nTraining interrupted by user!")
            print("Saving partial checkpoint to interrupted_model.pth...")
            # We assume train_metrics exists if at least one epoch finished, or try to use last known
            # But safer to just save current state
            dummy_metrics = {'loss': float('inf'), 'pearson_median': 0.0}
            self.save_checkpoint(dummy_metrics, is_best=False, is_latest=False, filename='interrupted_model.pth')
            print("Done. You can resume later.")
        
        self.writer.close()
        print(f"\n{'='*60}")
        print(f"Training Complete!")
        print(f"Best Validation Pearson: {self.best_val_pearson:.4f}")
        print(f"{'='*60}\n")


class WorkerInit:
    """Callable for worker initialization visibility"""
    def __init__(self, total_workers):
        self.total_workers = total_workers
        
    def __call__(self, worker_id):
        count = worker_id + 1
        percent = (count / self.total_workers) * 100
        print(f"[{os.getpid()}] Worker {count}/{self.total_workers} initialized ({percent:.1f}%)")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/config.yaml')
    parser.add_argument('--model-type', type=str, choices=['baseline', 'advanced'], default=None)
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from (or "latest" or "interrupted")')
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override model type if specified
    if args.model_type:
        config['model']['type'] = args.model_type
    
    # Setup device - prioritize MPS on Apple Silicon
    if torch.backends.mps.is_available():
        device = 'mps'
        print(f"Using device: mps (Apple Silicon GPU)")
    elif torch.cuda.is_available():
        device = 'cuda'
        print(f"Using device: cuda (NVIDIA GPU)")
    else:
        device = 'cpu'
        print(f"Using device: cpu")
    
    setup_directories(config)
    
    # Create model
    model_type = config['model'].get('type', 'advanced')
    
    if model_type == 'advanced':
        model = create_advanced_model(config)
        print(f"✓ Created advanced model: {config['model']['backbone']}")
    else:
        model = create_model(config)
        print(f"✓ Created baseline model")
    
    print(f"✓ Model parameters: {count_parameters(model):,}")
    
    # Create dataset
    print("\nInitializing dataset...")
    xenium_data_path = Path(config['data']['xenium_data_path'])
    if not xenium_data_path.exists():
        # Fallback to hardcoded path if config path is empty or invalid
        xenium_data_path = Path("Xenium dataset file")
    
    dataset = XeniumDataset(
        image_path=xenium_data_path / "morphology_mip.ome.tif",
        transcripts_path=xenium_data_path / "transcripts_full.csv",
        metadata_path=xenium_data_path / "metadata.json",
        patch_size=config['data']['patch_size'],
        # Use simple stride for now, or add to config
        stride=config['data'].get('stride', config['data']['patch_size']),
        preload_image=False
    )
    
    # Update config with actual number of genes
    config['model']['num_genes'] = dataset.num_genes
    gene_names = dataset.gene_names
    print(f"✓ Dataset loaded: {len(dataset)} patches, {dataset.num_genes} genes")
    
    # Split data
    train_ratio = config['data']['train_ratio']
    val_ratio = config['data']['val_ratio']
    test_ratio = config['data']['test_ratio']
    
    total_size = len(dataset)
    train_size = int(total_size * train_ratio)
    val_size = int(total_size * val_ratio)
    test_size = total_size - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    print(f"✓ Splits: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")
    
    # Create DataLoaders
    use_pin_memory = (device != 'mps')
    if device == 'mps':
        print("ℹ️  Disabling pin_memory for MPS device")

    # Define worker init function for visibility
    total_workers = config['training']['num_workers']
    worker_init = WorkerInit(total_workers)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=total_workers,
        pin_memory=use_pin_memory,
        worker_init_fn=worker_init
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=total_workers,
        pin_memory=use_pin_memory,
        worker_init_fn=worker_init
    )

    # Create model (re-create with correct num_genes)
    if model_type == 'advanced':
        model = create_advanced_model(config)
    else:
        model = create_model(config)
    model = model.to(device)
    print(f"✓ Re-created model with {dataset.num_genes} output genes")

    # Train
    trainer = AdvancedTrainer(config, model, device)
    
    # Handle Resume
    start_epoch = 1
    if args.resume:
        checkpoint_dir = Path(config['paths']['checkpoints_dir'])
        if args.resume == 'latest':
            resume_path = checkpoint_dir / 'latest_model.pth'
        elif args.resume == 'interrupted':
             resume_path = checkpoint_dir / 'interrupted_model.pth'
        else:
            resume_path = Path(args.resume)
            
        if resume_path.exists():
            last_epoch = trainer.load_checkpoint(resume_path)
            start_epoch = last_epoch + 1
        else:
            print(f"⚠️ Checkpoint not found at {resume_path}, starting from scratch.")
            
    trainer.train(train_loader, val_loader, gene_names, start_epoch=start_epoch)


if __name__ == "__main__":
    try:
        # MacOS requires 'spawn' start method
        import multiprocessing
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
        
    main()
