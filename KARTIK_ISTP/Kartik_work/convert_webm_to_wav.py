"""
Convert downloaded webm files to wav using Python libraries
"""

from pydub import AudioSegment
from pathlib import Path
from tqdm import tqdm

# Directories
webm_dir = Path("data/raw/chainsaw_youtube")
wav_dir = Path("data/raw/chainsaw")
wav_dir.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("CONVERTING WEBM TO WAV")
print("=" * 70)

# Find all webm files
webm_files = list(webm_dir.glob("*.webm"))

if not webm_files:
    print("\n⚠ No WebM files found in", webm_dir)
    print("\nPlease run the download command first:")
    print('python -m yt_dlp "ytsearch5:chainsaw sound effect" -o "data/raw/chainsaw_youtube/%(title)s.%(ext)s"')
else:
    print(f"\nFound {len(webm_files)} WebM files")
    print(f"Converting to WAV format...\n")
    
    for webm_file in tqdm(webm_files, desc="Converting"):
        try:
            # Load webm
            audio = AudioSegment.from_file(str(webm_file), format="webm")
            
            # Export as wav
            wav_file = wav_dir / (webm_file.stem + ".wav")
            audio.export(str(wav_file), format="wav")
            
            print(f"✓ Converted: {webm_file.name}")
        
        except Exception as e:
            print(f"⚠ Failed {webm_file.name}: {e}")
    
    # Count converted files
    wav_files = list(wav_dir.glob("*.wav"))
    print(f"\n✓ Conversion complete!")
    print(f"Total WAV files in {wav_dir}: {len(wav_files)}")
    
    print("\nNext steps:")
    print("1. Listen to the files to verify quality")
    print("2. Delete any that don't contain chainsaw sounds")
    print("3. Run: python preprocess.py")
    print("4. Run: python train_simple.py")
