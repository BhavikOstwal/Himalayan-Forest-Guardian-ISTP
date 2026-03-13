"""
Simple dataset downloader for chainsaw detection
Downloads public datasets that don't require competition acceptance
"""

import os
from pathlib import Path
import urllib.request
import zipfile
from tqdm import tqdm


class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url, output_path):
    """Download file with progress bar"""
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=url.split('/')[-1]) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)


def download_esc50():
    """Download ESC-50 dataset (includes chainsaw sounds)"""
    print("\n=== Downloading ESC-50 Dataset ===")
    print("This dataset includes chainsaw and various environmental sounds")
    
    url = "https://github.com/karoldvl/ESC-50/archive/master.zip"
    output_dir = Path("data/raw")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    zip_path = output_dir / "esc50.zip"
    
    if zip_path.exists():
        print(f"✓ ESC-50 already downloaded")
    else:
        print(f"Downloading from: {url}")
        download_url(url, str(zip_path))
        print(f"✓ Downloaded to: {zip_path}")
    
    # Extract
    extract_dir = output_dir / "ESC-50"
    if extract_dir.exists():
        print("✓ ESC-50 already extracted")
    else:
        print("Extracting...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(output_dir)
        
        # Rename if needed
        if (output_dir / "ESC-50-master").exists():
            (output_dir / "ESC-50-master").rename(extract_dir)
        
        print(f"✓ Extracted to: {extract_dir}")
    
    return extract_dir


def download_freesound_dataset():
    """Download FSD50K - Free Sound Dataset"""
    print("\n=== FSD50K Dataset ===")
    print("For FSD50K, please download from:")
    print("https://zenodo.org/record/4060432")
    print("(Contains chainsaw and various tool sounds)")
    print("")


def download_audioset_ontology():
    """Download AudioSet CSV files for reference"""
    print("\n=== AudioSet Metadata ===")
    
    output_dir = Path("data/raw/audioset_metadata")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    files = {
        "ontology.json": "https://raw.githubusercontent.com/audioset/ontology/master/ontology.json",
        "balanced_train.csv": "http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/balanced_train_segments.csv",
        "eval.csv": "http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/eval_segments.csv"
    }
    
    for filename, url in files.items():
        output_path = output_dir / filename
        if output_path.exists():
            print(f"✓ {filename} already downloaded")
        else:
            print(f"Downloading {filename}...")
            try:
                download_url(url, str(output_path))
                print(f"✓ {filename} downloaded")
            except Exception as e:
                print(f"⚠ Could not download {filename}: {e}")
    
    print("\nTo download actual AudioSet audio:")
    print("1. Filter CSVs for chainsaw class (/m/01b82r)")
    print("2. Use yt-dlp: yt-dlp -x --audio-format wav <youtube_url>")


def organize_esc50():
    """Organize ESC-50 files for training"""
    print("\n=== Organizing ESC-50 ===")
    
    esc_dir = Path("data/raw/ESC-50/audio")
    if not esc_dir.exists():
        print("⚠ ESC-50 audio directory not found")
        return
    
    import pandas as pd
    meta_file = Path("data/raw/ESC-50/meta/esc50.csv")
    
    if not meta_file.exists():
        print("⚠ ESC-50 metadata not found")
        return
    
    df = pd.read_csv(meta_file)
    
    # Create organized structure
    chainsaw_dir = Path("data/raw/chainsaw")
    non_chainsaw_dir = Path("data/raw/non_chainsaw")
    chainsaw_dir.mkdir(parents=True, exist_ok=True)
    non_chainsaw_dir.mkdir(parents=True, exist_ok=True)
    
    # Chainsaw class in ESC-50 is "chainsaw" (class 38)
    chainsaw_files = df[df['category'] == 'chainsaw']
    
    # Copy chainsaw files
    import shutil
    for _, row in chainsaw_files.iterrows():
        src = esc_dir / row['filename']
        dst = chainsaw_dir / row['filename']
        if src.exists() and not dst.exists():
            shutil.copy2(src, dst)
    
    print(f"✓ Copied {len(chainsaw_files)} chainsaw files to: {chainsaw_dir}")
    
    # Copy some non-chainsaw environmental sounds as negative samples
    non_chainsaw_categories = [
        'rain', 'wind', 'crickets', 'insects', 'rooster', 'dog',
        'crackling_fire', 'footsteps', 'breathing', 'crow'
    ]
    
    non_chainsaw_files = df[df['category'].isin(non_chainsaw_categories)]
    
    for _, row in non_chainsaw_files.iterrows():
        src = esc_dir / row['filename']
        dst = non_chainsaw_dir / row['filename']
        if src.exists() and not dst.exists():
            shutil.copy2(src, dst)
    
    print(f"✓ Copied {len(non_chainsaw_files)} non-chainsaw files to: {non_chainsaw_dir}")
    
    return len(chainsaw_files), len(non_chainsaw_files)


def main():
    print("=" * 70)
    print("CHAINSAW DETECTION - SIMPLE DATASET DOWNLOADER")
    print("=" * 70)
    
    # Download ESC-50
    esc_dir = download_esc50()
    
    # Download AudioSet metadata
    download_audioset_ontology()
    
    # Info about other datasets
    download_freesound_dataset()
    
    print("\n" + "=" * 70)
    print("ORGANIZING DATA")
    print("=" * 70)
    
    # Organize ESC-50
    try:
        counts = organize_esc50()
        if counts:
            chainsaw_count, non_chainsaw_count = counts
            print(f"\n✓ Dataset organized:")
            print(f"  Chainsaw samples: {chainsaw_count}")
            print(f"  Non-chainsaw samples: {non_chainsaw_count}")
    except Exception as e:
        print(f"⚠ Error organizing data: {e}")
    
    print("\n" + "=" * 70)
    print("ADDITIONAL DATA SOURCES (Optional):")
    print("=" * 70)
    print("""
1. Rainforest Connection (RFCx):
   - Visit: https://www.kaggle.com/c/rfcx-species-audio-detection
   - Accept competition rules
   - Download using Kaggle API or web interface
   
2. Record your own chainsaw sounds (recommended!):
   - Find YouTube videos of chainsaws
   - Use: yt-dlp -x --audio-format wav <url>
   - This adds diversity to your dataset
   
3. Freesound.org:
   - Search for "chainsaw" and "forest ambient"
   - Download individual files
   - Free with attribution
""")
    
    print("\n" + "=" * 70)
    print("NEXT STEPS:")
    print("=" * 70)
    print("""
1. Check data folders:
   - data/raw/chainsaw/       (chainsaw sounds)
   - data/raw/non_chainsaw/   (forest ambient sounds)

2. Add more data if needed (100+ samples per class recommended)

3. Run preprocessing:
   python preprocess.py

4. Start training:
   python train.py
""")


if __name__ == "__main__":
    main()
