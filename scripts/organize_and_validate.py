#!/usr/bin/env python3
"""
Dataset Organization and Validation Script
==========================================

Organizes videos and validates the dataset structure for training.

Usage:
    python scripts/organize_and_validate.py --data-root data/
    python scripts/organize_and_validate.py --validate --data-root data/
    python scripts/organize_and_validate.py --stats --data-root data/
"""

import os
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple
import subprocess

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DatasetValidator:
    """Validate dataset structure and integrity"""
    
    def __init__(self, data_root: str):
        self.data_root = Path(data_root)
        self.processed_dir = self.data_root / 'processed'
        self.annotations_dir = self.data_root / 'annotations'
        self.raw_dir = self.data_root / 'raw'
    
    def validate_structure(self) -> Tuple[bool, List[str]]:
        """Validate directory structure"""
        errors = []
        
        # Check required directories
        required_dirs = [
            self.processed_dir / 'training' / 'videos',
            self.processed_dir / 'validation' / 'videos',
            self.processed_dir / 'test' / 'videos',
            self.annotations_dir
        ]
        
        for d in required_dirs:
            if not d.exists():
                errors.append(f"Missing directory: {d}")
        
        # Check annotation files
        required_files = [
            self.annotations_dir / 'action_detection.json',
            self.annotations_dir / 'video_qa.json',
            self.annotations_dir / 'video_captioning.json'
        ]
        
        for f in required_files:
            if not f.exists():
                errors.append(f"Missing annotation file: {f}")
        
        # Check metadata
        if not (self.data_root / 'metadata.json').exists():
            errors.append(f"Missing metadata.json")
        
        if not (self.data_root / 'dataloader_config.json').exists():
            errors.append(f"Missing dataloader_config.json")
        
        return len(errors) == 0, errors
    
    def validate_annotations(self) -> Tuple[bool, List[str]]:
        """Validate annotation files"""
        errors = []
        
        for task in ['action_detection', 'video_qa', 'video_captioning']:
            ann_file = self.annotations_dir / f"{task}.json"
            
            if not ann_file.exists():
                errors.append(f"Missing {task}.json")
                continue
            
            try:
                with open(ann_file) as f:
                    data = json.load(f)
                
                if not isinstance(data, list):
                    errors.append(f"{task}.json: Expected list, got {type(data).__name__}")
                    continue
                
                if len(data) == 0:
                    errors.append(f"{task}.json: Empty annotation file")
                    continue
                
                # Check required fields
                required_fields = {'video_id', 'video_file', 'split'}
                for item in data[:5]:  # Check first 5 items
                    missing = required_fields - set(item.keys())
                    if missing:
                        errors.append(f"{task}.json: Missing fields {missing} in entry {item.get('video_id', 'unknown')}")
                        break
                
                logger.info(f"✓ {task}.json: {len(data)} annotations")
            
            except json.JSONDecodeError as e:
                errors.append(f"{task}.json: Invalid JSON - {e}")
            except Exception as e:
                errors.append(f"{task}.json: Error - {e}")
        
        return len(errors) == 0, errors
    
    def count_videos(self) -> Dict[str, int]:
        """Count videos in each split"""
        counts = {
            'training': 0,
            'validation': 0,
            'test': 0
        }
        
        video_dirs = {
            'training': self.processed_dir / 'training' / 'videos',
            'validation': self.processed_dir / 'validation' / 'videos',
            'test': self.processed_dir / 'test' / 'videos'
        }
        
        for split, video_dir in video_dirs.items():
            if video_dir.exists():
                videos = list(video_dir.glob('*'))
                counts[split] = len([v for v in videos if v.is_file()])
        
        return counts
    
    def get_video_stats(self) -> Dict:
        """Get statistics about video files"""
        stats = {
            'total_size_gb': 0,
            'video_formats': {},
            'videos_per_split': self.count_videos()
        }
        
        video_dirs = {
            'training': self.processed_dir / 'training' / 'videos',
            'validation': self.processed_dir / 'validation' / 'videos',
            'test': self.processed_dir / 'test' / 'videos'
        }
        
        for split, video_dir in video_dirs.items():
            if video_dir.exists():
                for video_file in video_dir.glob('*'):
                    if video_file.is_file():
                        # Count formats
                        ext = video_file.suffix.lower()
                        stats['video_formats'][ext] = stats['video_formats'].get(ext, 0) + 1
                        
                        # Calculate size
                        stats['total_size_gb'] += video_file.stat().st_size / (1024**3)
        
        return stats
    
    def validate_all(self) -> bool:
        """Run all validations"""
        print("\n" + "="*80)
        print("DATASET VALIDATION".center(80))
        print("="*80 + "\n")
        
        # Check structure
        print("📁 Checking directory structure...")
        struct_ok, struct_errors = self.validate_structure()
        if struct_ok:
            print("   ✅ Directory structure is valid\n")
        else:
            print("   ❌ Directory structure issues:")
            for error in struct_errors:
                print(f"      - {error}")
            print()
        
        # Check annotations
        print("📋 Checking annotations...")
        ann_ok, ann_errors = self.validate_annotations()
        if ann_ok:
            print("   ✅ Annotations are valid\n")
        else:
            print("   ❌ Annotation issues:")
            for error in ann_errors:
                print(f"      - {error}")
            print()
        
        # Get statistics
        print("📊 Dataset Statistics:")
        stats = self.get_video_stats()
        video_counts = stats['videos_per_split']
        total_videos = sum(video_counts.values())
        
        print(f"   Training videos: {video_counts['training']:,}")
        print(f"   Validation videos: {video_counts['validation']:,}")
        print(f"   Test videos: {video_counts['test']:,}")
        print(f"   Total videos: {total_videos:,}")
        print(f"   Total size: {stats['total_size_gb']:.2f} GB")
        
        if stats['video_formats']:
            print(f"   Video formats: {', '.join(f'{ext}({count})' for ext, count in stats['video_formats'].items())}")
        print()
        
        # Check metadata
        print("📋 Checking metadata...")
        metadata_file = self.data_root / 'metadata.json'
        config_file = self.data_root / 'dataloader_config.json'
        
        if metadata_file.exists():
            print(f"   ✅ metadata.json exists")
        else:
            print(f"   ❌ metadata.json missing")
        
        if config_file.exists():
            print(f"   ✅ dataloader_config.json exists")
        else:
            print(f"   ❌ dataloader_config.json missing")
        print()
        
        # Summary
        all_ok = struct_ok and ann_ok and (total_videos > 0 or True)  # Allow empty for now
        
        print("="*80)
        if all_ok and total_videos > 0:
            print("✅ DATASET VALIDATION PASSED".center(80))
            print("="*80)
            print("\n🚀 Dataset is ready for training!\n")
            return True
        elif all_ok:
            print("⚠️  DATASET STRUCTURE VALID (No videos yet)".center(80))
            print("="*80)
            print("\n📥 Download videos to complete setup\n")
            return False
        else:
            print("❌ DATASET VALIDATION FAILED".center(80))
            print("="*80)
            print("\n⚠️  Fix the issues above before training\n")
            return False
    
    def print_setup_guide(self):
        """Print setup guide"""
        print("\n" + "="*80)
        print("DATASET SETUP GUIDE".center(80))
        print("="*80 + "\n")
        
        print("Step 1: Download Videos")
        print("-" * 80)
        print("  From ActivityNet official website:")
        print("  http://activity-net.org/download.html")
        print()
        print("  Or download sample videos (for testing):")
        print("  mkdir -p data/raw")
        print("  # ... download or copy videos to data/raw/")
        print()
        
        print("Step 2: Organize Videos (if not already done)")
        print("-" * 80)
        print("  python scripts/organize_and_validate.py --organize --data-root data/")
        print()
        
        print("Step 3: Validate Dataset")
        print("-" * 80)
        print("  python scripts/organize_and_validate.py --validate --data-root data/")
        print()
        
        print("Step 4: Start Training")
        print("-" * 80)
        print("  python hierarchicalvlm/train/train_hierarchical.py \\")
        print("    --config configs/training_config.yaml \\")
        print("    --train-data data/processed/training/videos \\")
        print("    --val-data data/processed/validation/videos")
        print()
        
        print("="*80 + "\n")


class VideoOrganizer:
    """Organize videos from raw directory"""
    
    def __init__(self, data_root: str):
        self.data_root = Path(data_root)
        self.raw_dir = self.data_root / 'raw'
        self.processed_dir = self.data_root / 'processed'
    
    def organize_from_lists(self) -> dict:
        """Organize videos using split lists"""
        stats = {
            'training': 0,
            'validation': 0,
            'test': 0,
            'organized': 0,
            'missing': 0,
            'already_organized': 0
        }
        
        splits_file = self.data_root / 'splits.json'
        if not splits_file.exists():
            logger.error(f"Splits file not found: {splits_file}")
            return stats
        
        with open(splits_file) as f:
            splits = json.load(f)
        
        # Map split names
        split_mapping = {
            'train': 'training',
            'val': 'validation',
            'test': 'test'
        }
        
        for split_name, video_list in splits.items():
            processed_split = split_mapping.get(split_name, split_name)
            target_dir = self.processed_dir / processed_split / 'videos'
            target_dir.mkdir(parents=True, exist_ok=True)
            
            for video_file in video_list:
                src = self.raw_dir / video_file
                dst = target_dir / video_file
                
                if dst.exists() or dst.is_symlink():
                    stats['already_organized'] += 1
                    stats[processed_split] += 1
                elif src.exists():
                    # Create symlink instead of copying to save space
                    try:
                        os.symlink(src, dst)
                        stats['organized'] += 1
                        stats[processed_split] += 1
                    except Exception as e:
                        logger.warning(f"Failed to create symlink for {video_file}: {e}")
                else:
                    stats['missing'] += 1
        
        return stats


def main():
    parser = argparse.ArgumentParser(
        description='Organize and validate dataset for training',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Validate dataset structure
  python scripts/organize_and_validate.py --validate --data-root data/

  # Show statistics
  python scripts/organize_and_validate.py --stats --data-root data/

  # Organize videos from raw directory
  python scripts/organize_and_validate.py --organize --data-root data/

  # Show setup guide
  python scripts/organize_and_validate.py --guide
        """
    )
    
    parser.add_argument('--data-root', type=str, default='./data', help='Data root directory')
    parser.add_argument('--validate', action='store_true', help='Validate dataset')
    parser.add_argument('--organize', action='store_true', help='Organize videos')
    parser.add_argument('--stats', action='store_true', help='Show statistics')
    parser.add_argument('--guide', action='store_true', help='Show setup guide')
    
    args = parser.parse_args()
    
    validator = DatasetValidator(args.data_root)
    
    if args.guide:
        validator.print_setup_guide()
        return 0
    
    if args.organize:
        logger.info("Organizing videos...")
        organizer = VideoOrganizer(args.data_root)
        stats = organizer.organize_from_lists()
        
        print("\n" + "="*60)
        print("Video Organization Summary".center(60))
        print("="*60)
        print(f"  Training organized: {stats['training']:,}")
        print(f"  Validation organized: {stats['validation']:,}")
        print(f"  Test organized: {stats['test']:,}")
        print(f"  Total organized: {stats['organized']:,}")
        print(f"  Already organized: {stats['already_organized']:,}")
        print(f"  Missing: {stats['missing']:,}")
        print("="*60 + "\n")
    
    if args.validate or (not args.organize and not args.guide and not args.stats):
        validator.validate_all()
    elif args.stats:
        stats = validator.get_video_stats()
        print("\n" + "="*60)
        print("Dataset Statistics".center(60))
        print("="*60)
        print(f"  Training videos: {stats['videos_per_split']['training']:,}")
        print(f"  Validation videos: {stats['videos_per_split']['validation']:,}")
        print(f"  Test videos: {stats['videos_per_split']['test']:,}")
        print(f"  Total size: {stats['total_size_gb']:.2f} GB")
        if stats['video_formats']:
            print(f"  Formats: {stats['video_formats']}")
        print("="*60 + "\n")
    
    return 0


if __name__ == '__main__':
    exit(main())
