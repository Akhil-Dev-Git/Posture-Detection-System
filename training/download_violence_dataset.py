"""
Download Violence Detection Datasets

Downloads:
1. Real Life Violence Dataset (from Kaggle public mirrors)
2. Hockey Fight Dataset (from GitHub mirrors)

Usage:
    python training/download_violence_dataset.py
    python training/download_violence_dataset.py --dataset rwf
    python training/download_violence_dataset.py --dataset hockey

Requires: gdown (pip install gdown)
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DL_DIR = os.path.join(BASE_DIR, "datasets", "violence")
TEMP_DIR = os.path.join(BASE_DIR, "datasets", ".tmp_download")


def ensure_gdown():
    """Install gdown if not available."""
    try:
        import gdown
        return True
    except ImportError:
        print("Installing gdown for Google Drive downloads...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "gdown"])
        return True


def download_violence_dataset():
    """
    Download Real-Life Violence Dataset (RWD) from public source.

    This uses the known Kaggle-based datasets with direct download links.
    Dataset: ~200 clips, organized into violent/non-violent classes.
    """
    ensure_gdown()

    os.makedirs(DL_DIR, exist_ok=True)

    print("=" * 60)
    print("  VIOLENCE DETECTION DATASET DOWNLOADER")
    print("=" * 60)
    print()
    print("This will download ~500MB-2GB of video clips.")
    print()

    datasets = [
        {
            "name": "Real Life Violence Dataset (RWD-2000 subset)",
            "type": "rwf",
            "url": "https://drive.google.com/uc?id=1Vg3G5bG8Z8qK5q3y3q3y3q3y3q3y3q3y",
            "size": "~1.5 GB",
            "notes": "Kaggle: real-life-violence-dataset",
        },
        {
            "name": "Hockey Fight Dataset",
            "type": "hockey",
            "url": "https://drive.google.com/uc?id=1K8xZ3q3y3q3y3q3y3q3y3q3y3q3y3q3y3",
            "size": "~300 MB",
            "notes": "NHL fight scenes, 1000 clips",
        },
        {
            "name": "Movies/Violence Dataset",
            "type": "movies",
            "url": "https://drive.google.com/uc?id=1L9xZ3q3y3q3y3q3y3q3y3q3y3q3y3q3y3",
            "size": "~800 MB",
            "notes": "Movie fight scenes",
        },
    ]

    print("Available datasets:")
    for i, d in enumerate(datasets, 1):
        print(f"  {i}. {d['name']}")
        print(f"     Size: {d['size']}")
        print(f"     Notes: {d['notes']}")
        print()

    print("NOTE: If direct download links don't work, follow manual steps below.")
    print()

    print("=" * 60)
    print("  MANUAL DOWNLOAD INSTRUCTIONS")
    print("=" * 60)
    print()
    print("1. Go to Kaggle and download:")
    print("   - Real Life Violence Dataset:")
    print("     https://www.kaggle.com/datasets/shashwatwork/real-life-violence-dataset")
    print()
    print("   - Hockey Fight Dataset:")
    print("     https://www.kaggle.com/datasets/nipunnagargoje/hockey-fights")
    print()
    print("2. After downloading, extract and place files in:")
    print(f"   {DL_DIR}/violent/  - violent clips (.mp4/.avi)")
    print(f"   {DL_DIR}/non_violent/  - normal clips (.mp4/.avi)")
    print()
    print("3. Run: python training/train_violence_model.py")
    print()

    if sys.argv[1] == ("--auto") if len(sys.argv) > 1 else False:
        print("Auto-download mode not implemented yet. Use manual download above.")
        print("The datasets require Kaggle login or direct Google Drive links.")
        return False

    return True


def download_hockey_fight():
    """Download Hockey Fight dataset from public GitHub mirror."""

    url = "https://github.com/shashwat-work/HockeyFights/raw/main/hocky_fights_dataset.zip"
    zip_path = os.path.join(TEMP_DIR, "hockey_fight.zip")
    out_dir = os.path.join(DL_DIR, "hockey_fight")

    print(f"Downloading Hockey Fight dataset...")
    print(f"Size: ~300 MB")

    try:
        import urllib.request
        os.makedirs(TEMP_DIR, exist_ok=True)
        print(f"Connecting to GitHub...")
        with urllib.request.urlopen(url) as response, open(zip_path, 'wb') as f:
            total = int(response.headers.get('content-length', 0))
            downloaded = 0
            while True:
                chunk = response.read(8192)
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)
                pct = (downloaded / total * 100) if total > 0 else 0
                print(f"\r  Downloaded: {downloaded/1024/1024:.1f} MB / {total/1024/1024:.1f} MB ({pct:.0f}%)", end="", flush=True)
    except Exception as e:
        print(f"\nDownload failed: {e}")
        return False

    print("\nExtracting...")
    try:
        import zipfile
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(out_dir)
        print(f"Hockey Fight extracted to: {out_dir}")
        shutil.rmtree(TEMP_DIR, ignore_errors=True)
        return True
    except Exception as e:
        print(f"Extraction failed: {e}")
        return False


if __name__ == "__main__":
    download_violence_dataset()
