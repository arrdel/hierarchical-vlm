#!/usr/bin/env python3
"""
HierarchicalVLM Training Example

Quick-start script demonstrating how to train the complete model
with all three phases integrated:
- Efficient Attention
- Domain Modules  
- Adaptive Token Merging

Usage:
    # Basic training
    python train_example.py
    
    # With custom config
    python train_example.py --config configs/training_config.yaml
    
    # Resume training
    python train_example.py --resume checkpoints/best_model.pth
    
    # Multi-GPU training
    python -m torch.distributed.launch --nproc_per_node=4 train_example.py
"""

import os
import sys
import argparse
from pathlib import Path

# Add project to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
import yaml


def print_banner(text: str, width: int = 80):
    """Print formatted banner."""
    print("\n" + "="*width)
    print(text.center(width))
    print("="*width + "\n")


def print_section(title: str):
    """Print section header."""
    print(f"\n{'─'*60}")
    print(f"  {title}")
    print(f"{'─'*60}\n")


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def print_config_summary(config: dict) -> None:
    """Print summary of configuration."""
    print_section("Configuration Summary")
    
    # Model
    print("📦 Model Configuration:")
    print(f"   • Attention type: {config['model']['attention']['type']}")
    print(f"   • Domain modules: {'✓' if config['model']['domain_modules']['enabled'] else '✗'}")
    print(f"   • Token merging: {'✓' if config['model']['token_merging']['enabled'] else '✗'}")
    print(f"   • Hidden size: {config['model']['hidden_size']}")
    print(f"   • Num heads: {config['model']['num_attention_heads']}")
    
    # Data
    print("\n📊 Data Configuration:")
    print(f"   • Batch size: {config['data']['batch_size']}")
    print(f"   • Num frames: {config['data']['video']['num_frames']}")
    print(f"   • Frame size: {config['data']['video']['frame_size']}")
    print(f"   • Num workers: {config['data']['num_workers']}")
    
    # Training
    print("\n🚀 Training Configuration:")
    print(f"   • Optimizer: {config['optimizer']['type']}")
    print(f"   • Learning rate: {config['optimizer']['lr']}")
    print(f"   • Scheduler: {config['scheduler']['type']}")
    print(f"   • Num epochs: {config['training']['num_epochs']}")
    print(f"   • Mixed precision: {'✓' if config['training']['mixed_precision'] else '✗'}")
    
    print()


def create_model(config: dict) -> nn.Module:
    """Create model with all components.
    
    This is a placeholder that demonstrates how components would be integrated.
    In practice, this would import and instantiate:
    - Base vision model
    - Efficient attention layers
    - Domain module adapters
    - Token merging components
    """
    print_section("Model Initialization")
    
    print("Components to be initialized:")
    print("  ✓ Base vision encoder (CLIP or similar)")
    
    if config['model']['domain_modules']['enabled']:
        print("  ✓ LoRA adapters (99%+ parameter reduction)")
        print("  ✓ Task-specific heads (action, QA, captioning)")
        print("  ✓ Domain routing (4 domains)")
    
    if config['model']['attention']['type'] != 'standard':
        print(f"  ✓ {config['model']['attention']['type']} attention")
    
    if config['model']['token_merging']['enabled']:
        print("  ✓ Optical flow (motion detection)")
        print("  ✓ Saliency detection (5 saliency types)")
        print("  ✓ Adaptive token merging")
    
    # Placeholder model
    model = nn.Identity()
    
    return model


def setup_training(config: dict, model: nn.Module):
    """Setup optimizer, scheduler, and other training components."""
    print_section("Training Setup")
    
    print(f"Optimizer: {config['optimizer']['type']}")
    print(f"  • Learning rate: {config['optimizer']['lr']}")
    print(f"  • Weight decay: {config['optimizer']['weight_decay']}")
    print(f"  • Gradient clip: {config['training']['gradient_clip']}")
    
    print(f"\nScheduler: {config['scheduler']['type']}")
    print(f"  • Warmup steps: {config['scheduler']['warmup_steps']}")
    print(f"  • T_max: {config['scheduler']['T_max']}")
    
    if config['training']['mixed_precision']:
        print(f"\nMixed precision training enabled: {config['training']['amp_dtype']}")
    
    if config['training']['gradient_checkpointing']:
        print("Gradient checkpointing enabled (reduced memory usage)")


def print_training_info(config: dict) -> None:
    """Print information about training setup."""
    print_section("Training Information")
    
    print("🎯 Training Strategy:")
    print(f"  • Epochs: {config['training']['num_epochs']}")
    print(f"  • Batch size: {config['data']['batch_size']}")
    print(f"  • Eval interval: Every {config['training']['eval_interval']} epoch(s)")
    print(f"  • Save interval: Every {config['training']['save_interval']} epoch(s)")
    
    print("\n📈 Optimization:")
    if config['training']['domain_balancing']['enabled']:
        print(f"  • Domain balancing: {config['training']['domain_balancing']['sample_strategy']}")
        print(f"    - Weights: {config['training']['domain_balancing']['domain_weights']}")
    
    if config['training']['temporal_smoothing']['enabled']:
        print(f"  • Temporal smoothing: Window={config['training']['temporal_smoothing']['window_size']}")
    
    print("\n💾 Checkpointing:")
    print(f"  • Save dir: {config['checkpoint']['save_dir']}")
    print(f"  • Save best: {config['checkpoint']['save_best']}")
    print(f"  • Keep top k: {config['checkpoint']['keep_top_k']}")
    if config['checkpoint']['use_ema']:
        print(f"  • EMA enabled: decay={config['checkpoint']['ema_decay']}")


def print_dataset_info(config: dict) -> None:
    """Print information about datasets."""
    print_section("Dataset Configuration")
    
    print("📹 Video Preprocessing:")
    print(f"  • Number of frames: {config['data']['video']['num_frames']}")
    print(f"  • Frame resolution: {config['data']['video']['frame_size']}x{config['data']['video']['frame_size']}")
    print(f"  • Sampling strategy: {config['data']['video']['frame_sampling']}")
    
    print("\n📝 Text Preprocessing:")
    print(f"  • Max length: {config['data']['text']['max_length']}")
    print(f"  • Tokenizer: {config['data']['text']['tokenizer']}")
    
    print("\n🔄 Data Augmentation:")
    if config['data']['augmentation']['enabled']:
        print(f"  • Random crop: {config['data']['augmentation']['random_crop']}")
        print(f"  • Color jitter: {config['data']['augmentation']['color_jitter']}")
        print(f"  • Random flip: {config['data']['augmentation']['random_flip']}")
        print(f"  • Temporal shift: {config['data']['augmentation']['temporal_shift']}")
        print(f"  • Dropout: {config['data']['augmentation']['dropout']}")


def print_training_commands(config_path: str) -> None:
    """Print useful training commands."""
    print_section("Useful Commands")
    
    print("Single GPU training:")
    print(f"  python hierarchicalvlm/train/train_hierarchical.py \\")
    print(f"    --config {config_path} \\")
    print(f"    --train-data /path/to/train \\")
    print(f"    --val-data /path/to/val")
    
    print("\nMulti-GPU training (DDP):")
    print(f"  python -m torch.distributed.launch --nproc_per_node=4 \\")
    print(f"    hierarchicalvlm/train/train_hierarchical.py \\")
    print(f"    --config {config_path} \\")
    print(f"    --train-data /path/to/train")
    
    print("\nResume training:")
    print(f"  python hierarchicalvlm/train/train_hierarchical.py \\")
    print(f"    --config {config_path} \\")
    print(f"    --resume checkpoints/best_model.pth \\")
    print(f"    --train-data /path/to/train")
    
    print("\nEvaluation only:")
    print(f"  python hierarchicalvlm/eval/run_inference_benchmark.py \\")
    print(f"    --checkpoint checkpoints/best_model.pth \\")
    print(f"    --test-data /path/to/test")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='HierarchicalVLM Training Example',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--config', type=str, default='configs/training_config.yaml',
                       help='Path to training configuration')
    parser.add_argument('--show-only', action='store_true',
                       help='Only show configuration without training')
    args = parser.parse_args()
    
    # Print header
    print_banner("🎬 HierarchicalVLM Training")
    
    # Load configuration
    print_section("Loading Configuration")
    config = load_config(args.config)
    print(f"✓ Configuration loaded from: {args.config}\n")
    
    # Print all configuration information
    print_config_summary(config)
    print_dataset_info(config)
    print_training_info(config)
    
    # Create model
    model = create_model(config)
    
    # Setup training
    setup_training(config, model)
    
    # Print useful commands
    print_training_commands(args.config)
    
    # Instructions
    print_section("Next Steps")
    print("1️⃣  Prepare your dataset (videos and annotations)")
    print("2️⃣  Implement custom DataLoader in hierarchicalvlm/train/")
    print("3️⃣  Run training with:")
    print(f"    python hierarchicalvlm/train/train_hierarchical.py --config {args.config}")
    print("\n4️⃣  Monitor training:")
    print("    tensorboard --logdir ./runs")
    
    print("\n✨ For more information:")
    print("   • Attention docs: docs/ATTENTION.md")
    print("   • Domain modules: docs/DOMAIN_MODULES.md")
    print("   • Token merging: docs/TOKEN_MERGING.md")
    print("   • Training guide: docs/TRAINING.md")
    
    print_banner("Ready to train! 🚀")


if __name__ == '__main__':
    main()
