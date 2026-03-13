"""
Helper script to download chainsaw audio from YouTube
"""

import subprocess
import os
from pathlib import Path


# Create directory for downloaded chainsaw sounds
download_dir = Path("data/raw/chainsaw_youtube")
download_dir.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("CHAINSAW AUDIO DOWNLOADER FROM YOUTUBE")
print("=" * 70)

# List of YouTube URLs with chainsaw sounds
chainsaw_videos = [
    # Add actual chainsaw video URLs here
    # These are examples - replace with real URLs
    "https://www.youtube.com/watch?v=qP7Goc9jjZ4",  # Chainsaw sound effects
    "https://www.youtube.com/watch?v=2OtBJUGCL4I",  # Tree cutting sounds
    "https://www.youtube.com/watch?v=vVLMOy8YXSY",  # Chainsaw compilation
]

print(f"\nDownload directory: {download_dir}")
print(f"\nVideos to download: {len(chainsaw_videos)}")
print("\nStarting downloads...")
print("-" * 70)

for i, url in enumerate(chainsaw_videos, 1):
    print(f"\n[{i}/{len(chainsaw_videos)}] Downloading: {url}")
    
    try:
        # Download audio as WAV file
        cmd = [
            "yt-dlp",
            "-x",  # Extract audio
            "--audio-format", "wav",  # Convert to WAV
            "--audio-quality", "0",  # Best quality
            "-o", str(download_dir / "%(id)s.%(ext)s"),  # Output filename
            url
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"✓ Downloaded successfully")
        else:
            print(f"⚠ Failed: {result.stderr}")
    
    except Exception as e:
        print(f"⚠ Error: {e}")

print("\n" + "=" * 70)
print("DOWNLOAD COMPLETE")
print("=" * 70)

# Count downloaded files
downloaded_files = list(download_dir.glob("*.wav"))
print(f"\nTotal WAV files: {len(downloaded_files)}")

if downloaded_files:
    print(f"\nFiles saved to: {download_dir}")
    print("\nNext steps:")
    print("1. Listen to the files and verify they contain chainsaw sounds")
    print("2. Move good files to: data/raw/chainsaw/")
    print("3. Re-run: python preprocess.py")
    print("4. Re-train: python train_simple.py")
else:
    print("\nNo files downloaded. Please:")
    print("1. Check your internet connection")
    print("2. Verify the YouTube URLs are correct")
    print("3. Try manual search: Search YouTube for 'chainsaw sound'")
    print("4. Use this command for individual downloads:")
    print('   yt-dlp -x --audio-format wav -o "data/raw/chainsaw/%(title)s.%(ext)s" <URL>')
