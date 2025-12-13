#!/usr/bin/env python3
"""Multi-GPU training with Weights & Biases integration."""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
import wandb
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from hierarchicalvlm.data import ActivityNetFeaturesDataset, collate_features

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class FeatureTrainingConfig:
    def __init__(
        self,
        train_feature_dir="/media/scratch/adele/activitynet/ActivityNet-13/train/train",
        val_feature_dir="/media/scratch/adele/activitynet/ActivityNet-13/test/test",
        annotations_file="/media/scratch/adele/activitynet/ActivityNet-13/gt.json",
        batch_size=32,
        num_epochs=50,
        learning_rate=1e-4,
        weight_decay=1e-4,
        feature_dim=2048,
        hidden_dim=1024,
        num_attention_heads=8,
        num_layers=6,
        num_workers=4,
        output_dir="./runs",
        log_steps=50,
        seed=42,
    ):
        self.train_feature_dir = train_feature_dir
        self.val_feature_dir = val_feature_dir
        self.annotations_file = annotations_file
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.num_attention_heads = num_attention_heads
        self.num_layers = num_layers
        self.num_workers = num_workers
        self.output_dir = output_dir
        self.log_steps = log_steps
        self.seed = seed

    def to_dict(self):
        return {k: v for k, v in self.__dict__.items()}


class FeatureVLMModel(nn.Module):
    def __init__(self, config: FeatureTrainingConfig):
        super().__init__()
        self.config = config
        self.feature_proj = nn.Linear(config.feature_dim, config.hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=config.num_attention_heads,
            dim_feedforward=config.hidden_dim * 4,
            batch_first=True,
            dropout=0.1,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)
        self.classifier = nn.Sequential(
            nn.LayerNorm(config.hidden_dim),
            nn.Dropout(0.1),
            nn.Linear(config.hidden_dim, 200),
        )
        
    def forward(self, features, attention_mask=None):
        x = self.feature_proj(features)
        if attention_mask is not None:
            attn_mask = attention_mask == 0
            x = self.transformer(x, src_key_padding_mask=attn_mask)
        else:
            x = self.transformer(x)
        return self.classifier(x)


def train_epoch(model, train_loader, optimizer, device, config, rank=0, epoch=0, use_wandb=False):
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for batch_idx, batch in enumerate(train_loader):
        features = batch["features"].to(device)
        attention_masks = batch["attention_masks"].to(device)
        logits = model(features, attention_masks)
        loss = torch.mean(logits ** 2)
        
        optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        if rank == 0 and (batch_idx + 1) % config.log_steps == 0:
            avg_loss = total_loss / num_batches
            logger.info(
                f"Epoch {epoch + 1} | Batch [{batch_idx + 1}/{len(train_loader)}] | "
                f"Loss: {avg_loss:.4f} | Grad: {grad_norm:.4f}"
            )
            
            if use_wandb:
                wandb.log({
                    "train/loss": loss.item(),
                    "train/avg_loss": avg_loss,
                    "train/batch": batch_idx + 1,
                    "train/epoch": epoch + 1,
                    "train/grad_norm": grad_norm,
                    "train/lr": optimizer.param_groups[0]['lr'],
                })
    
    return total_loss / num_batches


def validate(model, val_loader, device, rank=0, epoch=0, use_wandb=False):
    model.eval()
    total_loss = 0.0
    losses = []
    
    with torch.no_grad():
        for batch in val_loader:
            features = batch["features"].to(device)
            attention_masks = batch["attention_masks"].to(device)
            logits = model(features, attention_masks)
            loss = torch.mean(logits ** 2)
            total_loss += loss.item()
            losses.append(loss.item())
    
    avg_val_loss = total_loss / len(losses)
    
    if rank == 0:
        logger.info(f"Validation Epoch {epoch + 1} | Loss: {avg_val_loss:.4f}")
        
        if use_wandb:
            wandb.log({
                "val/loss": avg_val_loss,
                "val/loss_min": min(losses),
                "val/loss_max": max(losses),
                "val/loss_std": np.std(losses),
                "val/epoch": epoch + 1,
            })
    
    return avg_val_loss


def main():
    parser = argparse.ArgumentParser(description="Train HierarchicalVLM with W&B logging")
    
    parser.add_argument("--train-data", type=str, default="/media/scratch/adele/activitynet/ActivityNet-13/train/train")
    parser.add_argument("--val-data", type=str, default="/media/scratch/adele/activitynet/ActivityNet-13/test/test")
    parser.add_argument("--annotations", type=str, default="/media/scratch/adele/activitynet/ActivityNet-13/gt.json")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=1024)
    parser.add_argument("--num-layers", type=int, default=6)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--output-dir", type=str, default="./runs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-cuda", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="hierarchical-vlm-features")
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--wandb-run-name", type=str, default=None)
    parser.add_argument("--disable-wandb", action="store_true")
    
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    
    use_ddp = "RANK" in os.environ
    if use_ddp:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        local_rank = 0
        rank = 0
        world_size = 1
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    
    if rank == 0:
        logger.info(f"Using device: {device}")
        if use_ddp:
            logger.info(f"DDP: rank {rank}/{world_size}")
    
    config = FeatureTrainingConfig(
        train_feature_dir=args.train_data,
        val_feature_dir=args.val_data,
        annotations_file=args.annotations,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_attention_heads=args.num_heads,
        num_workers=args.num_workers,
        output_dir=args.output_dir,
        seed=args.seed,
    )
    
    if rank == 0:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        config_file = output_dir / "config.json"
        with open(config_file, "w") as f:
            json.dump(config.to_dict(), f, indent=2)
        logger.info(f"Config: {config_file}")
    
    if use_ddp:
        dist.barrier()
    
    use_wandb = rank == 0 and not args.disable_wandb
    if use_wandb:
        logger.info("🔐 Logging into Weights & Biases...")
        wandb.login()  # Login to wandb - required for online logging
        
        run_name = args.wandb_run_name or f"feature_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=run_name,
            config=config.to_dict(),
            tags=["multi-gpu", "activitynet", "features", "DDP", "verbose-logging"],
            mode="online",  # Explicitly set to online mode (not offline)
        )
        logger.info(f"✅ W&B Online: {wandb.run.url}")
    else:
        if rank == 0:
            logger.info("W&B disabled")
    
    if rank == 0:
        logger.info("Creating data loaders...")
    
    if use_ddp:
        train_dataset = ActivityNetFeaturesDataset(
            feature_dir=config.train_feature_dir,
            annotations_file=config.annotations_file,
            split="train",
            normalize=True,
        )
        val_dataset = ActivityNetFeaturesDataset(
            feature_dir=config.val_feature_dir,
            annotations_file=config.annotations_file,
            split="val",
            normalize=True,
        )
        
        train_sampler = DistributedSampler(
            train_dataset, num_replicas=world_size, rank=rank, shuffle=True
        )
        val_sampler = DistributedSampler(
            val_dataset, num_replicas=world_size, rank=rank, shuffle=False
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            sampler=train_sampler,
            num_workers=config.num_workers,
            collate_fn=collate_features,
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            sampler=val_sampler,
            num_workers=config.num_workers,
            collate_fn=collate_features,
            pin_memory=True,
        )
    else:
        train_dataset = ActivityNetFeaturesDataset(
            feature_dir=config.train_feature_dir,
            annotations_file=config.annotations_file,
            split="train",
            normalize=True,
        )
        val_dataset = ActivityNetFeaturesDataset(
            feature_dir=config.val_feature_dir,
            annotations_file=config.annotations_file,
            split="val",
            normalize=True,
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            collate_fn=collate_features,
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            collate_fn=collate_features,
            pin_memory=True,
        )
    
    if rank == 0:
        logger.info(f"Train: {len(train_loader)} batches ({len(train_loader.dataset)} samples)")
        logger.info(f"Val: {len(val_loader)} batches ({len(val_loader.dataset)} samples)")
    
    if rank == 0:
        logger.info("Creating model...")
    model = FeatureVLMModel(config).to(device)
    
    if use_ddp:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    
    if rank == 0:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Model: {total_params:,} params ({trainable_params:,} trainable)")
        
        if use_wandb:
            wandb.log({
                "model/total_parameters": total_params,
                "model/trainable_parameters": trainable_params,
            })
    
    optimizer = optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = CosineAnnealingLR(
        optimizer, T_max=config.num_epochs * len(train_loader), eta_min=1e-6
    )
    
    if rank == 0:
        logger.info("\n" + "="*70)
        logger.info("STARTING TRAINING")
        logger.info("="*70 + "\n")
    
    best_val_loss = float("inf")
    
    for epoch in range(config.num_epochs):
        if rank == 0:
            logger.info(f"\nEpoch {epoch + 1}/{config.num_epochs}")
        
        if use_ddp:
            train_loader.sampler.set_epoch(epoch)
        
        train_loss = train_epoch(model, train_loader, optimizer, device, config, rank, epoch, use_wandb)
        scheduler.step()
        
        if rank == 0:
            logger.info(f"Train Loss: {train_loss:.4f}")
        
        if (epoch + 1) % 5 == 0:
            val_loss = validate(model, val_loader, device, rank, epoch, use_wandb)
            
            if rank == 0:
                logger.info(f"Val Loss: {val_loss:.4f}")
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    output_dir = Path(args.output_dir)
                    checkpoint_path = output_dir / "best_model.pt"
                    torch.save(model.state_dict(), checkpoint_path)
                    logger.info(f"✅ Best model saved")
                    
                    if use_wandb:
                        wandb.log({"best_val_loss": best_val_loss})
        
        if rank == 0 and (epoch + 1) % 10 == 0:
            output_dir = Path(args.output_dir)
            checkpoint_path = output_dir / f"checkpoint_epoch_{epoch + 1}.pt"
            torch.save(model.state_dict(), checkpoint_path)
            logger.info(f"Checkpoint saved: epoch {epoch + 1}")
    
    if rank == 0:
        output_dir = Path(args.output_dir)
        final_model_path = output_dir / "final_model.pt"
        torch.save(model.state_dict(), final_model_path)
        logger.info(f"\n✨ TRAINING COMPLETE!")
        logger.info(f"Best val loss: {best_val_loss:.4f}")
        logger.info(f"Output: {output_dir}")
        
        if use_wandb:
            wandb.summary["best_val_loss"] = best_val_loss
            wandb.summary["total_epochs"] = config.num_epochs
            wandb.finish()
    
    if use_ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
