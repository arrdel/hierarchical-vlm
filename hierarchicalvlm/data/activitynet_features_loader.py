"""ActivityNet Features DataLoader for pre-extracted feature training."""

import json
import numpy as np
import torch
from pathlib import Path
from typing import Dict, Optional, Tuple, Any
from torch.utils.data import Dataset, DataLoader
import logging

logger = logging.getLogger(__name__)


class ActivityNetFeaturesDataset(Dataset):
    """
    PyTorch Dataset for ActivityNet pre-extracted features.
    
    Loads pre-computed video features (.npy files) and their annotations
    from ground truth JSON file.
    """

    def __init__(
        self,
        feature_dir: str,
        annotations_file: Optional[str] = None,
        split: str = "train",
        max_frames: Optional[int] = None,
        normalize: bool = True,
    ):
        """
        Args:
            feature_dir: Path to directory containing .npy feature files
            annotations_file: Path to gt.json ground truth file
            split: 'train' or 'test' (for logging purposes)
            max_frames: Maximum number of frames to use per video (None = use all)
            normalize: Whether to normalize features to unit norm
        """
        self.feature_dir = Path(feature_dir)
        self.split = split
        self.max_frames = max_frames
        self.normalize = normalize
        self.annotations = {}

        # Collect all feature files
        self.features = sorted(list(self.feature_dir.glob("*.npy")))
        logger.info(f"Found {len(self.features)} feature files in {self.split} split")

        # Load annotations if provided
        if annotations_file and Path(annotations_file).exists():
            with open(annotations_file, "r") as f:
                gt_data = json.load(f)
                if "database" in gt_data:
                    self.annotations = gt_data["database"]
            logger.info(f"Loaded annotations for {len(self.annotations)} videos")

    def __len__(self) -> int:
        """Returns number of videos in dataset."""
        return len(self.features)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Load a video's features and metadata.

        Returns:
            Dict containing:
                - video_id: Video identifier
                - features: Tensor of shape (T, D) where T=frames, D=feature_dim
                - num_frames: Number of frames in the feature
                - annotations: List of action annotations (if available)
        """
        feature_file = self.features[idx]
        video_id = feature_file.stem

        # Load features
        features = np.load(feature_file).astype(np.float32)

        # Handle multi-frame features - take mean if needed
        if len(features.shape) > 1:
            # Features shape: (num_frames, feature_dim)
            num_frames = features.shape[0]
            if self.max_frames is not None and num_frames > self.max_frames:
                # Uniformly sample max_frames from the sequence
                indices = np.linspace(
                    0, num_frames - 1, self.max_frames, dtype=np.int32
                )
                features = features[indices]
                num_frames = self.max_frames
        else:
            # Single feature vector
            num_frames = 1
            features = features.reshape(1, -1)

        # Normalize features
        if self.normalize:
            # L2 normalization
            norms = np.linalg.norm(features, axis=1, keepdims=True)
            norms[norms == 0] = 1  # Avoid division by zero
            features = features / norms

        # Get annotations if available
        annotations = []
        if video_id in self.annotations:
            video_info = self.annotations[video_id]
            if "annotations" in video_info:
                annotations = video_info["annotations"]

        return {
            "video_id": video_id,
            "features": torch.from_numpy(features),
            "num_frames": num_frames,
            "annotations": annotations,
        }


def collate_features(batch):
    """
    Custom collate function for variable-length feature sequences.
    
    Pads shorter sequences with zeros.
    """
    video_ids = [item["video_id"] for item in batch]
    num_frames = [item["num_frames"] for item in batch]
    annotations = [item["annotations"] for item in batch]

    # Get max length in batch
    max_length = max(num_frames)
    feature_dim = batch[0]["features"].shape[1]

    # Pad all features to max length
    padded_features = torch.zeros(len(batch), max_length, feature_dim)
    attention_masks = torch.zeros(len(batch), max_length)

    for i, item in enumerate(batch):
        length = item["num_frames"]
        padded_features[i, :length] = item["features"]
        attention_masks[i, :length] = 1.0

    return {
        "video_ids": video_ids,
        "features": padded_features,
        "attention_masks": attention_masks,
        "num_frames": num_frames,
        "annotations": annotations,
    }


def get_activitynet_loaders(
    train_feature_dir: str,
    val_feature_dir: str,
    annotations_file: str,
    batch_size: int = 32,
    num_workers: int = 4,
    max_frames: Optional[int] = None,
    normalize: bool = True,
    pin_memory: bool = True,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create PyTorch DataLoaders for training and validation.

    Args:
        train_feature_dir: Path to training features directory
        val_feature_dir: Path to validation features directory
        annotations_file: Path to gt.json
        batch_size: Batch size for training
        num_workers: Number of workers for data loading
        max_frames: Maximum frames to use per video
        normalize: Whether to L2-normalize features
        pin_memory: Whether to pin memory for faster transfer to GPU

    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Training dataset
    train_dataset = ActivityNetFeaturesDataset(
        feature_dir=train_feature_dir,
        annotations_file=annotations_file,
        split="train",
        max_frames=max_frames,
        normalize=normalize,
    )

    # Validation dataset
    val_dataset = ActivityNetFeaturesDataset(
        feature_dir=val_feature_dir,
        annotations_file=annotations_file,
        split="val",
        max_frames=max_frames,
        normalize=normalize,
    )

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_features,
        pin_memory=pin_memory,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_features,
        pin_memory=pin_memory,
    )

    logger.info(
        f"Created DataLoaders: "
        f"train={len(train_dataset)}, val={len(val_dataset)}"
    )

    return train_loader, val_loader


if __name__ == "__main__":
    """Test the DataLoader."""
    logging.basicConfig(level=logging.INFO)

    # Test paths
    train_dir = "/media/scratch/adele/activitynet/ActivityNet-13/train/train"
    val_dir = "/media/scratch/adele/activitynet/ActivityNet-13/test/test"
    anno_file = "/media/scratch/adele/activitynet/ActivityNet-13/gt.json"

    print("\n" + "=" * 70)
    print("🧪 TESTING ACTIVITYNET FEATURES DATALOADER")
    print("=" * 70)

    # Create loaders
    train_loader, val_loader = get_activitynet_loaders(
        train_feature_dir=train_dir,
        val_feature_dir=val_dir,
        annotations_file=anno_file,
        batch_size=4,
        num_workers=0,
        max_frames=None,
        normalize=True,
    )

    # Test training batch
    print("\n📊 TRAINING BATCH:")
    for batch in train_loader:
        print(f"   Batch size: {len(batch['video_ids'])}")
        print(f"   Feature shape: {batch['features'].shape}")
        print(f"   Attention mask shape: {batch['attention_masks'].shape}")
        print(f"   Sample video IDs: {batch['video_ids'][:2]}")
        print(f"   Sample num_frames: {batch['num_frames'][:2]}")
        break

    # Test validation batch
    print("\n✅ VALIDATION BATCH:")
    for batch in val_loader:
        print(f"   Batch size: {len(batch['video_ids'])}")
        print(f"   Feature shape: {batch['features'].shape}")
        print(f"   Attention mask shape: {batch['attention_masks'].shape}")
        break

    print("\n" + "=" * 70)
    print("✨ DataLoader test completed successfully!")
    print("=" * 70 + "\n")
