"""
Script to download chainsaw detection datasets
"""

import os
import json
import pandas as pd
from pathlib import Path
import subprocess
import argparse
from tqdm import tqdm
import getpass


def setup_kaggle_credentials():
    """Setup Kaggle API credentials"""
    kaggle_dir = Path.home() / '.kaggle'
    kaggle_json = kaggle_dir / 'kaggle.json'
    
    # Check if kaggle.json already exists
    if kaggle_json.exists():
        print("✓ Kaggle credentials found")
        return True
    
    print("\n=== Kaggle API Setup ===")
    print("You need Kaggle API credentials to download datasets.")
    print("Get them from: https://www.kaggle.com/settings → API → Create New API Token")
    
    choice = input("\nDo you want to enter your Kaggle credentials now? (y/n): ").lower()
    
    if choice != 'y':
        print("\nPlease setup Kaggle credentials manually:")
        print(f"1. Download kaggle.json from Kaggle website")
        print(f"2. Place it in: {kaggle_dir}")
        return False
    
    # Get credentials from user
    print("\nEnter your Kaggle credentials:")
    username = input("Kaggle Username: ").strip()
    api_key = getpass.getpass("Kaggle API Key (hidden): ").strip()
    
    if not username or not api_key:
        print("⚠ Invalid credentials. Please try again.")
        return False
    
    # Create kaggle directory if it doesn't exist
    kaggle_dir.mkdir(parents=True, exist_ok=True)
    
    # Create kaggle.json
    credentials = {
        "username": username,
        "key": api_key
    }
    
    with open(kaggle_json, 'w') as f:
        json.dump(credentials, f, indent=2)
    
    # Set permissions (important for security)
    if os.name != 'nt':  # Unix/Linux/Mac
        os.chmod(kaggle_json, 0o600)
    
    print(f"✓ Kaggle credentials saved to: {kaggle_json}")
    return True


def setup_directories(config):
    """Create necessary directories"""
    dirs = [
        config['data']['raw_data_dir'],
        config['data']['processed_data_dir'],
        config['data']['rfcx_dataset'],
        config['data']['audioset_dataset'],
        config['data']['output_dir']
    ]
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    print("✓ Directories created")


def download_rfcx_dataset(output_dir):
    """
    Download Rainforest Connection dataset from Kaggle
    You need to setup Kaggle API first: https://www.kaggle.com/docs/api
    """
    print("\n=== Downloading Rainforest Connection Dataset ===")
    
    try:
        # Import kaggle after credentials are set up
        import kaggle
        
        # Download the RFCx competition dataset
        kaggle.api.competition_download_files(
            'rfcx-species-audio-detection',
            path=output_dir,
            quiet=False
        )
        print("✓ RFCx dataset downloaded successfully")
        
        # Unzip
        import zipfile
        zip_path = os.path.join(output_dir, 'rfcx-species-audio-detection.zip')
        if os.path.exists(zip_path):
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(output_dir)
            print("✓ RFCx dataset extracted")
            
    except Exception as e:
        print(f"⚠ Could not download RFCx dataset: {e}")
        print("Please download manually from: https://www.kaggle.com/competitions/rfcx-species-audio-detection")


def download_audioset_chainsaw(output_dir, max_clips=500):
    """
    Download chainsaw sounds from AudioSet
    Note: This requires youtube-dl or yt-dlp
    """
    print("\n=== Downloading AudioSet Chainsaw Clips ===")
    
    # AudioSet ontology ID for chainsaw: /m/01b82r
    # You would need to parse AudioSet CSV and download from YouTube
    
    audioset_csv_url = "http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/balanced_train_segments.csv"
    
    print(f"Download AudioSet CSV from: {audioset_csv_url}")
    print("Then filter for chainsaw class (/m/01b82r)")
    print("Use yt-dlp to download the audio clips")
    
    # Example code structure (requires manual setup):
    instructions = """
    To download AudioSet chainsaw clips:
    
    1. Download balanced_train_segments.csv from AudioSet
    2. Filter rows containing '/m/01b82r' (chainsaw class)
    3. Use yt-dlp to download:
       
       yt-dlp -x --audio-format wav -o "%(id)s.%(ext)s" <youtube_url>
    
    Script template provided in utils/download_audioset_clips.py
    """
    print(instructions)


def download_esc50(output_dir):
    """Download ESC-50 dataset"""
    print("\n=== Downloading ESC-50 Dataset ===")
    
    esc50_url = "https://github.com/karoldvl/ESC-50/archive/master.zip"
    
    try:
        import urllib.request
        import zipfile
        
        zip_path = os.path.join(output_dir, "esc50.zip")
        print("Downloading ESC-50...")
        urllib.request.urlretrieve(esc50_url, zip_path)
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(output_dir)
        
        print("✓ ESC-50 downloaded and extracted")
        
    except Exception as e:
        print(f"⚠ Could not download ESC-50: {e}")
        print("Please download manually from: https://github.com/karoldvl/ESC-50")


def create_dataset_structure(base_dir):
    """Create organized dataset structure"""
    structure = {
        'train': ['chainsaw', 'non_chainsaw'],
        'val': ['chainsaw', 'non_chainsaw'],
        'test': ['chainsaw', 'non_chainsaw']
    }
    
    for split, classes in structure.items():
        for cls in classes:
            path = os.path.join(base_dir, split, cls)
            Path(path).mkdir(parents=True, exist_ok=True)
    
    print("✓ Dataset directory structure created")


def main():
    parser = argparse.ArgumentParser(description='Download chainsaw detection datasets')
    parser.add_argument('--config', type=str, default='config.yaml', help='Config file path')
    parser.add_argument('--skip-rfcx', action='store_true', help='Skip RFCx download')
    parser.add_argument('--kaggle-username', type=str, help='Kaggle username')
    parser.add_argument('--kaggle-key', type=str, help='Kaggle API key')
    
    args = parser.parse_args()
    
    # Load config
    import yaml
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    print("=" * 60)
    print("CHAINSAW DETECTION - DATASET DOWNLOADER")
    print("=" * 60)
    
    # Setup Kaggle credentials
    if args.kaggle_username and args.kaggle_key:
        # Use provided credentials
        kaggle_dir = Path.home() / '.kaggle'
        kaggle_dir.mkdir(parents=True, exist_ok=True)
        kaggle_json = kaggle_dir / 'kaggle.json'
        
        credentials = {
            "username": args.kaggle_username,
            "key": args.kaggle_key
        }
        
        with open(kaggle_json, 'w') as f:
            json.dump(credentials, f, indent=2)
        
        if os.name != 'nt':
            os.chmod(kaggle_json, 0o600)
        
        print("✓ Kaggle credentials configured from command line")
    else:
        # Interactive setup
        if not args.skip_rfcx:
            if not setup_kaggle_credentials():
                print("\n⚠ Skipping RFCx download (no credentials)")
                args.skip_rfcx = True
    print("CHAINSAW DETECTION - DATASET DOWNLOADER")
    print("=" * 60)
    
    # Setup directories
    setup_directories(config)
    
    # Download datasets
    if not getattr(args, 'skip_rfcx', False):
        download_rfcx_dataset(config['data']['rfcx_dataset'])
    
    if not getattr(args, 'skip_audioset', False):
        download_audioset_chainsaw(config['data']['audioset_dataset'])
    
    if not getattr(args, 'skip_esc50', False):
        download_esc50(config['data']['raw_data_dir'])
    
    # Create dataset structure
    create_dataset_structure(config['data']['processed_data_dir'])
    
    print("\n" + "=" * 60)
    print("NEXT STEPS:")
    print("1. Organize downloaded files into train/val/test splits")
    print("2. Run: python preprocess.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
