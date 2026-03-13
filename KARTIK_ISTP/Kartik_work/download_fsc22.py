"""
Download FSC22 (Forest Soundscapes) dataset - PRIMARY DATASET per paper
"FSC22 will be utilized more due to its overall focus on forest acoustics"
"""

import os
import sys
import json
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import shutil
import subprocess


def setup_kaggle_credentials():
    """Setup Kaggle API credentials"""
    kaggle_dir = Path.home() / '.kaggle'
    kaggle_dir.mkdir(exist_ok=True)
    
    kaggle_json = kaggle_dir / 'kaggle.json'
    
    credentials = {
        "username": "kartikverma2328",
        "key": "KGAT_299a668e38e46fd492b91f1d9c338e7e"
    }
    
    with open(kaggle_json, 'w') as f:
        json.dump(credentials, f)
    
    if sys.platform != 'win32':
        os.chmod(kaggle_json, 0o600)
    
    print(f"✓ Kaggle credentials saved to {kaggle_json}")


def download_fsd50k_kaggle():
    """Download FSD50K via Kaggle API"""
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        
        api = KaggleApi()
        api.authenticate()
        
        print("\n" + "=" * 70)
        print("DOWNLOADING FSD50K - PRIMARY DATASET")
        print("=" * 70)
        print("\n⚠ Large download (~30GB)")
        
        confirm = input("\nProceed with download? (y/n): ").strip().lower()
        if confirm != 'y':
            print("Download cancelled")
            return
        
        print("\n📥 Downloading FSD50K...")
        api.dataset_download_files(
            'tonyarobertson/fsd50k',
            path='data/raw/fsd50k',
            unzip=True
        )
        
        print("\n✓ Download complete")
        filter_forest_sounds()
        
    except Exception as e:
        print(f"\n❌ Download failed: {e}")
        print("\nManual download:")
        print("1. Visit: https://www.kaggle.com/datasets/tonyarobertson/fsd50k")
        print("2. Extract to: data/raw/fsd50k/")


def download_audioset_chainsaws():
    """Download chainsaw sounds via YouTube"""
    print("\n" + "=" * 70)
    print("DOWNLOADING CHAINSAW AUDIO FROM YOUTUBE")
    print("=" * 70)
    
    try:
        result = subprocess.run(['yt-dlp', '--version'], 
                              capture_output=True, text=True)
        print(f"✓ yt-dlp installed")
    except:
        print("❌ yt-dlp not found - installing...")
        subprocess.run([sys.executable, '-m', 'pip', 'install', 'yt-dlp'])
    
    output_dir = Path("data/raw/chainsaw")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    queries = [
        "chainsaw cutting tree forest",
        "chainsaw logging sound",
        "tree cutting chainsaw",
    ]
    
    for query in queries:
        print(f"\n🔍 Downloading: {query}")
        cmd = [
            'python', '-m', 'yt_dlp',
            f'ytsearch30:{query}',
            '-f', 'bestaudio',
            '-o', str(output_dir / '%(id)s.%(ext)s'),
        ]
        
        try:
            subprocess.run(cmd, timeout=300)
        except Exception as e:
            print(f"⚠ Failed: {e}")
    
    print(f"\n✓ Downloaded to: {output_dir}")


def download_fsc22():
    """Main download function"""
    
    print("=" * 70)
    print("FSC22 DATASET DOWNLOAD - PRIMARY PER PAPER")
    print("=" * 70)
    
    print("\n📚 Paper's Strategy:")
    print("  PRIMARY: FSC22 (forest acoustics) - MORE DATA")
    print("  SECONDARY: ESC-50 - LESS DATA")
    
    # Check Kaggle credentials
    kaggle_json = Path.home() / '.kaggle' / 'kaggle.json'
    
    if not kaggle_json.exists():
        print("\n⚠ Kaggle credentials not found")
        setup_kaggle = input("\nSetup Kaggle? (y/n): ").strip().lower()
        if setup_kaggle == 'y':
            setup_kaggle_credentials()
    
    print("\n📦 Download Options:")
    print("1. FSD50K via Kaggle (~30GB)")
    print("2. AudioSet Chainsaw Subset via YouTube (~100 samples)")
    print("3. Skip (use ESC-50 only - NOT recommended)")
    
    choice = input("\nSelect (1/2/3): ").strip()
    
    if choice == "1":
        download_fsd50k_kaggle()
    elif choice == "2":
        download_audioset_chainsaws()
    else:
        print("\n⚠ WARNING: ESC-50 only has 40 chainsaw samples!")
        print("Paper recommends FSC22 as primary dataset")


def main():
    download_fsc22()


if __name__ == "__main__":
    main()
