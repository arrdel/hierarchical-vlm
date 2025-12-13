"""
HierarchicalVLM Training Script

Unified training pipeline integrating:
- Efficient attention mechanisms for long-form videos
- Domain-specific fine-tuning modules (LoRA + multi-task heads)
- Adaptive token merging based on motion and saliency

This script provides flexible training with support for:
- Multiple domains (sports, tutorials, news, general content)
- Multiple tasks (action detection, VQA, captioning)
- Efficient memory usage through attention mechanisms
- Adaptive compression through token merging
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Optional, List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

import yaml
from tqdm import tqdm


class HierarchicalVLMTrainer:
    """Main trainer class for HierarchicalVLM.
    
    Handles:
    - Model initialization with all components
    - Data loading and preprocessing
    - Training loop with optimization
    - Evaluation and checkpointing
    - Multi-GPU and distributed training
    """
    
    def __init__(self, config_path: str):
        """Initialize trainer with configuration.
        
        Args:
            config_path: Path to YAML configuration file
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.device = self._setup_device()
        self.model = self._setup_model()
        self.optimizer = self._setup_optimizer()
        self.scheduler = self._setup_scheduler()
        
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        
        self.best_loss = float('inf')
        self.current_epoch = 0
        self.global_step = 0
    
    def _setup_device(self) -> torch.device:
        """Setup compute device (CPU/GPU/Multi-GPU)."""
        if torch.cuda.is_available():
            device = torch.device(f"cuda:{self.config['device']['gpu_id']}")
            torch.cuda.set_device(device)
            print(f"Using GPU: {torch.cuda.get_device_name(device)}")
        else:
            device = torch.device("cpu")
            print("Using CPU")
        
        return device
    
    def _setup_model(self) -> nn.Module:
        """Initialize HierarchicalVLM model with all components."""
        model_config = self.config['model']
        
        print("Initializing HierarchicalVLM model...")
        
        # Import model class (to be implemented)
        # from hierarchicalvlm.model import HierarchicalVLM
        # model = HierarchicalVLM(model_config)
        
        # Placeholder: create a simple model for now
        model = nn.Linear(768, 1000)  # Dummy model
        
        model.to(self.device)
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        
        return model
    
    def _setup_optimizer(self) -> optim.Optimizer:
        """Setup optimizer based on configuration."""
        optim_config = self.config['optimizer']
        
        if optim_config['type'] == 'adam':
            optimizer = optim.Adam(
                self.model.parameters(),
                lr=optim_config['lr'],
                weight_decay=optim_config.get('weight_decay', 1e-5)
            )
        elif optim_config['type'] == 'adamw':
            optimizer = optim.AdamW(
                self.model.parameters(),
                lr=optim_config['lr'],
                weight_decay=optim_config.get('weight_decay', 1e-5)
            )
        else:
            raise ValueError(f"Unknown optimizer: {optim_config['type']}")
        
        print(f"Optimizer: {optim_config['type']} (lr={optim_config['lr']})")
        
        return optimizer
    
    def _setup_scheduler(self) -> Optional[object]:
        """Setup learning rate scheduler."""
        sched_config = self.config.get('scheduler', {})
        
        if not sched_config:
            return None
        
        sched_type = sched_config['type']
        
        if sched_type == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=sched_config['T_max'],
                eta_min=sched_config.get('eta_min', 1e-5)
            )
        elif sched_type == 'step':
            scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=sched_config['step_size'],
                gamma=sched_config.get('gamma', 0.1)
            )
        else:
            return None
        
        print(f"LR Scheduler: {sched_type}")
        
        return scheduler
    
    def load_datasets(self, train_path: str, val_path: Optional[str] = None):
        """Load training and validation datasets.
        
        Args:
            train_path: Path to training data
            val_path: Path to validation data
        """
        data_config = self.config['data']
        
        print(f"Loading datasets from {train_path}")
        
        # TODO: Implement actual dataset loading
        # For now, create dummy dataloaders
        
        self.train_loader = DataLoader(
            torch.randn(100, 768),  # Dummy data
            batch_size=data_config['batch_size'],
            shuffle=True,
            num_workers=data_config.get('num_workers', 0)
        )
        
        if val_path:
            self.val_loader = DataLoader(
                torch.randn(20, 768),  # Dummy data
                batch_size=data_config['batch_size'],
                shuffle=False,
                num_workers=data_config.get('num_workers', 0)
            )
        
        print(f"Train batches: {len(self.train_loader)}")
        if self.val_loader:
            print(f"Val batches: {len(self.val_loader)}")
    
    def train_epoch(self) -> Dict[str, float]:
        """Run single training epoch.
        
        Returns:
            Dictionary with epoch statistics
        """
        self.model.train()
        
        total_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(
            self.train_loader,
            desc=f"Epoch {self.current_epoch + 1}",
            position=0
        )
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            if isinstance(batch, torch.Tensor):
                batch = batch.to(self.device)
            
            # Forward pass
            try:
                outputs = self.model(batch)
                loss = outputs.mean() if isinstance(outputs, torch.Tensor) else 0
            except Exception as e:
                print(f"Error in forward pass: {e}")
                loss = torch.tensor(0.0, device=self.device)
            
            # Backward pass
            self.optimizer.zero_grad()
            if loss > 0:
                loss.backward()
                
                # Gradient clipping
                if self.config['training'].get('gradient_clip', 0) > 0:
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config['training']['gradient_clip']
                    )
                
                self.optimizer.step()
            
            # Update statistics
            total_loss += loss.item() if isinstance(loss, torch.Tensor) else 0
            num_batches += 1
            self.global_step += 1
            
            # Update progress bar
            avg_loss = total_loss / (num_batches + 1e-8)
            progress_bar.set_postfix({'loss': f'{avg_loss:.4f}'})
        
        # Update scheduler
        if self.scheduler:
            self.scheduler.step()
        
        epoch_loss = total_loss / num_batches if num_batches > 0 else 0
        
        return {
            'loss': epoch_loss,
            'num_batches': num_batches,
            'global_step': self.global_step
        }
    
    def validate(self) -> Dict[str, float]:
        """Run validation.
        
        Returns:
            Dictionary with validation metrics
        """
        if not self.val_loader:
            return {}
        
        self.model.eval()
        
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validating"):
                if isinstance(batch, torch.Tensor):
                    batch = batch.to(self.device)
                
                try:
                    outputs = self.model(batch)
                    loss = outputs.mean() if isinstance(outputs, torch.Tensor) else 0
                except Exception as e:
                    print(f"Error in validation: {e}")
                    loss = 0
                
                total_loss += loss if isinstance(loss, (int, float)) else loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        
        return {'val_loss': avg_loss}
    
    def save_checkpoint(self, path: str, is_best: bool = False):
        """Save model checkpoint.
        
        Args:
            path: Path to save checkpoint
            is_best: Whether this is the best checkpoint
        """
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'config': self.config,
            'best_loss': self.best_loss
        }
        
        if self.scheduler:
            checkpoint['scheduler_state'] = self.scheduler.state_dict()
        
        os.makedirs(Path(path).parent, exist_ok=True)
        torch.save(checkpoint, path)
        
        print(f"Checkpoint saved to {path}")
        
        if is_best:
            best_path = str(Path(path).parent / 'best_model.pth')
            torch.save(checkpoint, best_path)
            print(f"Best model saved to {best_path}")
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint.
        
        Args:
            path: Path to checkpoint
        """
        if not os.path.exists(path):
            print(f"Checkpoint not found: {path}")
            return
        
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        
        if self.scheduler and 'scheduler_state' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state'])
        
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_loss = checkpoint.get('best_loss', float('inf'))
        
        print(f"Checkpoint loaded from {path}")
        print(f"Resuming from epoch {self.current_epoch}, step {self.global_step}")
    
    def train(self, num_epochs: int, save_dir: str = './checkpoints'):
        """Main training loop.
        
        Args:
            num_epochs: Number of epochs to train
            save_dir: Directory to save checkpoints
        """
        os.makedirs(save_dir, exist_ok=True)
        
        print(f"\n{'='*60}")
        print(f"Starting training for {num_epochs} epochs")
        print(f"{'='*60}\n")
        
        for epoch in range(self.current_epoch, num_epochs):
            self.current_epoch = epoch
            
            # Training
            train_stats = self.train_epoch()
            
            # Validation
            val_stats = self.validate()
            
            # Logging
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            print(f"  Train loss: {train_stats['loss']:.4f}")
            if val_stats:
                print(f"  Val loss: {val_stats['val_loss']:.4f}")
            
            # Save checkpoint
            checkpoint_path = os.path.join(save_dir, f'epoch_{epoch:03d}.pth')
            self.save_checkpoint(checkpoint_path)
            
            # Save best model
            current_loss = val_stats.get('val_loss', train_stats['loss'])
            if current_loss < self.best_loss:
                self.best_loss = current_loss
                self.save_checkpoint(checkpoint_path, is_best=True)
        
        print(f"\n{'='*60}")
        print(f"Training completed!")
        print(f"Best loss: {self.best_loss:.4f}")
        print(f"{'='*60}\n")
    
    def evaluate(self, test_path: str) -> Dict[str, float]:
        """Evaluate model on test set.
        
        Args:
            test_path: Path to test data
        
        Returns:
            Dictionary with test metrics
        """
        # TODO: Implement evaluation
        print(f"Evaluating on {test_path}")
        return {}


def create_default_config() -> Dict:
    """Create default training configuration.
    
    Returns:
        Default configuration dictionary
    """
    return {
        'model': {
            'num_attention_heads': 12,
            'attention_type': 'hierarchical',
            'use_domain_modules': True,
            'use_token_merging': True,
            'domain_modules': {
                'num_domains': 4,
                'lora_rank': 8,
                'lora_alpha': 16,
            },
            'token_merging': {
                'use_motion': True,
                'use_saliency': True,
                'compression_ratio': 0.5,
            }
        },
        'data': {
            'batch_size': 32,
            'num_workers': 4,
            'prefetch_factor': 2,
        },
        'optimizer': {
            'type': 'adamw',
            'lr': 1e-4,
            'weight_decay': 1e-5,
        },
        'scheduler': {
            'type': 'cosine',
            'T_max': 100,
            'eta_min': 1e-6,
        },
        'training': {
            'num_epochs': 100,
            'gradient_clip': 1.0,
            'warmup_steps': 1000,
            'log_interval': 100,
        },
        'device': {
            'gpu_id': 0,
            'use_distributed': False,
        }
    }


def main():
    """Main entry point for training script."""
    parser = argparse.ArgumentParser(
        description='Train HierarchicalVLM model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train with default config
  python train_hierarchical.py --train-data /path/to/train --val-data /path/to/val
  
  # Train with custom config
  python train_hierarchical.py --config config.yaml --train-data /path/to/train
  
  # Resume from checkpoint
  python train_hierarchical.py --resume checkpoints/best_model.pth --train-data /path/to/train
        """
    )
    
    parser.add_argument('--config', type=str, default=None,
                       help='Path to training config YAML file')
    parser.add_argument('--train-data', type=str, required=True,
                       help='Path to training data')
    parser.add_argument('--val-data', type=str, default=None,
                       help='Path to validation data')
    parser.add_argument('--test-data', type=str, default=None,
                       help='Path to test data')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume training')
    parser.add_argument('--save-dir', type=str, default='./checkpoints',
                       help='Directory to save checkpoints')
    parser.add_argument('--num-epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--gpu-id', type=int, default=0,
                       help='GPU device ID')
    
    args = parser.parse_args()
    
    # Load or create config
    if args.config:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    else:
        config = create_default_config()
    
    # Update config with command line arguments
    config['data']['batch_size'] = args.batch_size
    config['optimizer']['lr'] = args.lr
    config['training']['num_epochs'] = args.num_epochs
    config['device']['gpu_id'] = args.gpu_id
    
    # Initialize trainer
    # Save config temporarily to use in trainer
    temp_config_path = '/tmp/hierarchical_vlm_config.yaml'
    os.makedirs(os.path.dirname(temp_config_path), exist_ok=True)
    with open(temp_config_path, 'w') as f:
        yaml.dump(config, f)
    
    trainer = HierarchicalVLMTrainer(temp_config_path)
    
    # Load datasets
    trainer.load_datasets(args.train_data, args.val_data)
    
    # Resume from checkpoint if provided
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    # Train
    trainer.train(args.num_epochs, args.save_dir)
    
    # Evaluate if test data provided
    if args.test_data:
        trainer.evaluate(args.test_data)


if __name__ == '__main__':
    main()
