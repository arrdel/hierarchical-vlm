#!/usr/bin/env python3
"""
ActivityNet Dataset Downloader
==============================

Downloads ActivityNet videos and organizes them for training.

Usage:
    python scripts/download_anet_videos.py --output data/raw/ --list data/train_videos.txt
    python scripts/download_anet_videos.py --output data/raw/ --download-all --num-workers 4
"""

import os
import json
import argparse
import subprocess
import logging
from pathlib import Path
from typing import List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import urllib.request

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ActivityNetDownloader:
    """Handle ActivityNet video downloads"""
    
    # ActivityNet download URLs
    DOWNLOAD_URLS = {
        'direct': 'http://ec2-52-26-14-91.us-west-2.compute.amazonaws.com/share/activity_net_videos/',
        'mirror': 'https://www.dropbox.com/sh/ttcf2iswe10l4ut/',
        'youtube': True  # Videos available on YouTube
    }
    
    def __init__(self, output_dir: str, list_file: Optional[str] = None):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.list_file = list_file
        self.videos = self._load_video_list()
        
        logger.info(f"Downloader initialized")
        logger.info(f"Output directory: {self.output_dir}")
        if self.videos:
            logger.info(f"Videos to download: {len(self.videos)}")
    
    def _load_video_list(self) -> List[str]:
        """Load list of videos to download"""
        videos = []
        
        if self.list_file and Path(self.list_file).exists():
            with open(self.list_file) as f:
                videos = [line.strip() for line in f if line.strip()]
            logger.info(f"Loaded {len(videos)} videos from {self.list_file}")
        
        return videos
    
    def download_with_wget(self, video_name: str, base_url: str) -> bool:
        """Download video using wget"""
        output_file = self.output_dir / video_name
        
        if output_file.exists():
            logger.info(f"⏭️  Skipping {video_name} (already exists)")
            return True
        
        video_url = f"{base_url}{video_name}"
        
        try:
            logger.info(f"⬇️  Downloading: {video_name}")
            cmd = [
                'wget',
                '-q',  # Quiet mode
                '--no-check-certificate',
                '-O', str(output_file),
                video_url
            ]
            
            result = subprocess.run(cmd, timeout=3600, capture_output=True)
            
            if result.returncode == 0 and output_file.exists():
                logger.info(f"✅ Downloaded: {video_name}")
                return True
            else:
                logger.warning(f"❌ Failed to download: {video_name}")
                if output_file.exists():
                    output_file.unlink()
                return False
        
        except subprocess.TimeoutExpired:
            logger.error(f"⏱️  Timeout downloading: {video_name}")
            if output_file.exists():
                output_file.unlink()
            return False
        except Exception as e:
            logger.error(f"❌ Error downloading {video_name}: {e}")
            if output_file.exists():
                output_file.unlink()
            return False
    
    def print_instructions(self):
        """Print download instructions"""
        print("\n" + "="*80)
        print("ActivityNet Video Download Instructions".center(80))
        print("="*80)
        
        print("\n📍 Option 1: Direct Download (Recommended)")
        print("  ActivityNet videos can be downloaded from:")
        print("  https://www.dropbox.com/sh/ttcf2iswe10l4ut/?dl=0")
        print("  ")
        print("  Or use this script with wget installed")
        
        print("\n📍 Option 2: YouTube Download")
        print("  Use yt-dlp to download from YouTube URLs:")
        print("  $ pip install yt-dlp")
        print("  $ yt-dlp <VIDEO_URL> -o '%(id)s.%(ext)s'")
        
        print("\n📍 Option 3: Using Our Script")
        print(f"  Ensure you have the video list file at:")
        print(f"  {self.list_file}")
        print(f"  ")
        print(f"  Then run:")
        print(f"  python scripts/download_anet_videos.py \\")
        print(f"    --output data/raw/ \\")
        print(f"    --list data/train_videos.txt \\")
        print(f"    --num-workers 4")
        
        print("\n📋 Manual Download Process:")
        print("  1. Visit: http://activity-net.org/download.html")
        print("  2. Register for ActivityNet")
        print("  3. Download videos (or use wget/yt-dlp)")
        print("  4. Place videos in data/raw/")
        print(f"  5. Organize with: python scripts/organize_videos.py")
        
        print("\n📊 Video List Info:")
        print(f"  Total videos to download: {len(self.videos)}")
        if self.videos:
            print(f"  Sample videos:")
            for video in self.videos[:5]:
                print(f"    - {video}")
        
        print("\n" + "="*80 + "\n")
    
    def parallel_download(self, base_url: str, num_workers: int = 4) -> dict:
        """Download videos in parallel"""
        results = {
            'successful': [],
            'failed': [],
            'skipped': []
        }
        
        if not self.videos:
            logger.warning("No videos to download")
            return results
        
        logger.info(f"Starting parallel download with {num_workers} workers")
        
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {
                executor.submit(self.download_with_wget, video, base_url): video
                for video in self.videos
            }
            
            completed = 0
            for future in as_completed(futures):
                video = futures[future]
                completed += 1
                
                try:
                    if future.result():
                        if (self.output_dir / video).exists():
                            results['successful'].append(video)
                        else:
                            results['skipped'].append(video)
                    else:
                        results['failed'].append(video)
                except Exception as e:
                    logger.error(f"Error processing {video}: {e}")
                    results['failed'].append(video)
                
                # Progress report
                if completed % max(1, len(self.videos) // 10) == 0:
                    logger.info(f"Progress: {completed}/{len(self.videos)} ({completed/len(self.videos)*100:.1f}%)")
        
        return results


class VideoOrganizer:
    """Organize downloaded videos into train/val/test structure"""
    
    def __init__(self, raw_dir: str, output_dir: str):
        self.raw_dir = Path(raw_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def organize_videos(self, video_lists: dict) -> dict:
        """Organize videos into structured directories"""
        stats = {
            'training': 0,
            'validation': 0,
            'test': 0,
            'missing': 0
        }
        
        # Training videos
        for video_name in video_lists.get('train', []):
            src = self.raw_dir / video_name
            if src.exists():
                dst = self.output_dir / 'training' / 'videos' / video_name
                dst.parent.mkdir(parents=True, exist_ok=True)
                if not dst.exists():
                    os.symlink(src, dst)  # Create symlink to save space
                stats['training'] += 1
            else:
                stats['missing'] += 1
        
        # Validation videos
        for video_name in video_lists.get('val', []):
            src = self.raw_dir / video_name
            if src.exists():
                dst = self.output_dir / 'validation' / 'videos' / video_name
                dst.parent.mkdir(parents=True, exist_ok=True)
                if not dst.exists():
                    os.symlink(src, dst)
                stats['validation'] += 1
            else:
                stats['missing'] += 1
        
        # Test videos
        for video_name in video_lists.get('test', []):
            src = self.raw_dir / video_name
            if src.exists():
                dst = self.output_dir / 'test' / 'videos' / video_name
                dst.parent.mkdir(parents=True, exist_ok=True)
                if not dst.exists():
                    os.symlink(src, dst)
                stats['test'] += 1
            else:
                stats['missing'] += 1
        
        return stats


def main():
    parser = argparse.ArgumentParser(
        description='Download and organize ActivityNet videos',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Show download instructions
  python scripts/download_anet_videos.py --instructions

  # Download videos from a list
  python scripts/download_anet_videos.py --output data/raw/ --list data/train_videos.txt

  # Organize downloaded videos
  python scripts/download_anet_videos.py --organize --raw data/raw/ --output data/processed/
        """
    )
    
    parser.add_argument('--instructions', action='store_true', help='Show download instructions')
    parser.add_argument('--output', type=str, help='Output directory for videos')
    parser.add_argument('--list', type=str, help='File containing list of videos to download')
    parser.add_argument('--download', action='store_true', help='Download videos')
    parser.add_argument('--organize', action='store_true', help='Organize videos into train/val/test')
    parser.add_argument('--raw', type=str, help='Raw video directory for organization')
    parser.add_argument('--num-workers', type=int, default=2, help='Number of parallel downloads')
    
    args = parser.parse_args()
    
    if args.instructions or (not args.download and not args.organize):
        downloader = ActivityNetDownloader(args.output or 'data/raw/', args.list)
        downloader.print_instructions()
        return 0
    
    if args.download and args.output:
        downloader = ActivityNetDownloader(args.output, args.list)
        logger.warning("⚠️  Direct download requires wget and ActivityNet server access")
        logger.warning("⚠️  Please manually download videos from: http://activity-net.org/download.html")
        downloader.print_instructions()
        return 0
    
    if args.organize and args.raw and args.output:
        organizer = VideoOrganizer(args.raw, args.output)
        
        # Load video lists
        splits_file = Path(args.output).parent / 'splits.json'
        if splits_file.exists():
            with open(splits_file) as f:
                video_lists = json.load(f)
            
            logger.info("Organizing videos...")
            stats = organizer.organize_videos(video_lists)
            
            print("\n" + "="*60)
            print("Video Organization Summary".center(60))
            print("="*60)
            for split, count in stats.items():
                print(f"  {split:15s}: {count:5d}")
            print("="*60 + "\n")
        else:
            logger.error(f"Splits file not found: {splits_file}")
            return 1
    
    return 0


if __name__ == '__main__':
    exit(main())
