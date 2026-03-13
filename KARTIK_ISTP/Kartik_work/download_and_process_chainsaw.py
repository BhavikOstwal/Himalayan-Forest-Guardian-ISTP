"""
Download 10-hour chainsaw video from YouTube and process it into dataset
"""

import os
from pathlib import Path
import librosa
import soundfile as sf
import numpy as np
from tqdm import tqdm
import json
import yt_dlp
import subprocess
import imageio_ffmpeg

# Configuration
YOUTUBE_URL = "https://www.youtube.com/watch?v=qRhq46TKYBk"
DOWNLOAD_DIR = Path("D:/chainsaw_youtube_download")  # Using D drive for more space
CHAINSAW_DIR = Path("data/raw/chainsaw")
PROCESSED_TRAIN_DIR = Path("data/processed/train/chainsaw")
PROCESSED_VAL_DIR = Path("data/processed/val/chainsaw")

# Audio processing parameters
SAMPLE_RATE = 22050
SEGMENT_DURATION = 5.0  # seconds per clip
MIN_AMPLITUDE_THRESHOLD = 0.01  # Filter out silent segments
TRAIN_SPLIT = 0.9  # 90% train, 10% validation

# Create directories
DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
CHAINSAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_TRAIN_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_VAL_DIR.mkdir(parents=True, exist_ok=True)


def download_video():
    """Download YouTube video audio"""
    print("=" * 80)
    print("STEP 1: DOWNLOADING YOUTUBE VIDEO")
    print("=" * 80)
    print(f"URL: {YOUTUBE_URL}")
    print(f"Destination: {DOWNLOAD_DIR}")
    
    # Check for existing webm or wav file
    output_webm = DOWNLOAD_DIR / "chainsaw_10hrs.webm"
    output_wav = DOWNLOAD_DIR / "chainsaw_10hrs.wav"
    
    if output_wav.exists():
        print(f"\n✓ WAV file already exists: {output_wav}")
        return output_wav
    elif output_webm.exists():
        print(f"\n✓ WebM file already exists: {output_webm}")
        print("  (Will be used directly - no conversion needed)")
        return output_webm
    
    print("\nDownloading (this may take several minutes for a 10-hour video)...")
    print("Progress will be shown below:")
    
    try:
        # Download as webm (no conversion needed, librosa will read it)
        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': str(DOWNLOAD_DIR / 'chainsaw_10hrs.%(ext)s'),
            'quiet': False,
            'no_warnings': False,
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([YOUTUBE_URL])
        
        # Check which file was created
        if output_webm.exists():
            print(f"\n✓ Download successful: {output_webm}")
            return output_webm
        elif output_wav.exists():
            print(f"\n✓ Download successful: {output_wav}")
            return output_wav
        else:
            print(f"\n✗ Expected output file not found")
            return None
    
    except Exception as e:
        # If download fails but file exists, use it
        if output_webm.exists():
            print(f"\n⚠ Download had errors but file exists: {output_webm}")
            return output_webm
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return None


def segment_audio(audio_file):
    """Segment long audio file into smaller clips"""
    print("\n" + "=" * 80)
    print("STEP 2: SEGMENTING AUDIO")
    print("=" * 80)
    print(f"Input file: {audio_file}")
    print(f"Segment duration: {SEGMENT_DURATION}s")
    print(f"Sample rate: {SAMPLE_RATE} Hz")
    
    try:
        # Convert webm to wav if needed
        wav_file = audio_file
        if audio_file.suffix == '.webm':
            wav_file = audio_file.with_suffix('.wav')
            if not wav_file.exists():
                print("\nConverting webm to wav (this may take several minutes)...")
                ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
                cmd = [
                    ffmpeg_path,
                    '-i', str(audio_file),
                    '-ar', str(SAMPLE_RATE),
                    '-ac', '1',  # mono
                    '-y',  # overwrite
                    str(wav_file)
                ]
                subprocess.run(cmd, check=True)
                print(f"✓ Converted to: {wav_file}")
            else:
                print(f"\n✓ WAV file already exists: {wav_file}")
        
        # Load audio
        print("\nLoading audio file (this may take a while for 10 hours)...")
        audio, sr = librosa.load(wav_file, sr=SAMPLE_RATE, mono=True)
        duration = len(audio) / sr
        
        print(f"✓ Audio loaded:")
        print(f"  - Duration: {duration/3600:.2f} hours ({duration:.0f} seconds)")
        print(f"  - Sample rate: {sr} Hz")
        print(f"  - Samples: {len(audio):,}")
        
        # Calculate segments
        segment_samples = int(SEGMENT_DURATION * sr)
        total_segments = int(len(audio) / segment_samples)
        
        print(f"\n  - Expected segments: {total_segments:,}")
        
        # Split into train and validation
        train_count = int(total_segments * TRAIN_SPLIT)
        val_count = total_segments - train_count
        
        print(f"  - Training segments: {train_count:,}")
        print(f"  - Validation segments: {val_count:,}")
        
        # Process segments
        print("\nExtracting segments...")
        train_saved = 0
        val_saved = 0
        skipped = 0
        
        for i in tqdm(range(total_segments), desc="Processing segments"):
            start = i * segment_samples
            end = start + segment_samples
            segment = audio[start:end]
            
            # Filter out silent or very quiet segments
            if np.max(np.abs(segment)) < MIN_AMPLITUDE_THRESHOLD:
                skipped += 1
                continue
            
            # Normalize segment
            segment = segment / (np.max(np.abs(segment)) + 1e-8)
            
            # Decide train or validation
            if i < train_count:
                output_dir = PROCESSED_TRAIN_DIR
                filename = f"youtube_chainsaw_train_{train_saved:05d}.wav"
                train_saved += 1
            else:
                output_dir = PROCESSED_VAL_DIR
                filename = f"youtube_chainsaw_val_{val_saved:05d}.wav"
                val_saved += 1
            
            # Save segment
            output_path = output_dir / filename
            sf.write(output_path, segment, sr)
        
        print(f"\n✓ Segmentation complete:")
        print(f"  - Training clips saved: {train_saved:,}")
        print(f"  - Validation clips saved: {val_saved:,}")
        print(f"  - Skipped (too quiet): {skipped:,}")
        
        # Save metadata
        metadata = {
            "source_url": YOUTUBE_URL,
            "source_file": str(audio_file),
            "source_duration_hours": duration / 3600,
            "segment_duration_seconds": SEGMENT_DURATION,
            "sample_rate": sr,
            "train_segments": train_saved,
            "val_segments": val_saved,
            "skipped_segments": skipped,
            "min_amplitude_threshold": MIN_AMPLITUDE_THRESHOLD
        }
        
        metadata_file = DOWNLOAD_DIR / "processing_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\nMetadata saved: {metadata_file}")
        
        return train_saved, val_saved
    
    except Exception as e:
        print(f"✗ Error during segmentation: {e}")
        import traceback
        traceback.print_exc()
        return 0, 0


def update_dataset_info():
    """Display dataset statistics"""
    print("\n" + "=" * 80)
    print("STEP 3: UPDATED DATASET STATISTICS")
    print("=" * 80)
    
    # Count files
    train_chainsaw = len(list(PROCESSED_TRAIN_DIR.glob("*.wav")))
    val_chainsaw = len(list(PROCESSED_VAL_DIR.glob("*.wav")))
    
    train_non_chainsaw_dir = Path("data/processed/train/non_chainsaw")
    val_non_chainsaw_dir = Path("data/processed/val/non_chainsaw")
    
    train_non_chainsaw = len(list(train_non_chainsaw_dir.glob("*.wav"))) if train_non_chainsaw_dir.exists() else 0
    val_non_chainsaw = len(list(val_non_chainsaw_dir.glob("*.wav"))) if val_non_chainsaw_dir.exists() else 0
    
    print("\nTraining Set:")
    print(f"  - Chainsaw: {train_chainsaw:,} clips")
    print(f"  - Non-chainsaw: {train_non_chainsaw:,} clips")
    print(f"  - Total: {train_chainsaw + train_non_chainsaw:,} clips")
    
    print("\nValidation Set:")
    print(f"  - Chainsaw: {val_chainsaw:,} clips")
    print(f"  - Non-chainsaw: {val_non_chainsaw:,} clips")
    print(f"  - Total: {val_chainsaw + val_non_chainsaw:,} clips")
    
    print("\nOverall:")
    total = train_chainsaw + val_chainsaw + train_non_chainsaw + val_non_chainsaw
    print(f"  - Total dataset size: {total:,} clips")
    
    if train_non_chainsaw > 0:
        balance_ratio = train_chainsaw / train_non_chainsaw
        print(f"  - Training balance ratio (chainsaw/non-chainsaw): {balance_ratio:.2f}")


def main():
    print("\n" + "=" * 80)
    print("CHAINSAW DATASET AUGMENTATION FROM YOUTUBE")
    print("=" * 80)
    print(f"\nVideo URL: {YOUTUBE_URL}")
    print(f"This will download a 10-hour video and extract chainsaw sounds")
    
    # Step 1: Download
    audio_file = download_video()
    if not audio_file:
        print("\n✗ Download failed. Please check:")
        print("  1. yt-dlp is installed: pip install yt-dlp")
        print("  2. Internet connection is working")
        print("  3. YouTube URL is accessible")
        return
    
    # Step 2: Segment
    train_count, val_count = segment_audio(audio_file)
    if train_count == 0 and val_count == 0:
        print("\n✗ No segments were created. Check the error messages above.")
        return
    
    # Step 3: Show stats
    update_dataset_info()
    
    # Step 4: Next steps
    print("\n" + "=" * 80)
    print("NEXT STEPS")
    print("=" * 80)
    print("\nYour dataset has been augmented with chainsaw sounds!")
    print("\nTo train the model with the new data:")
    print("  1. Simple model:    python train_simple.py")
    print("  2. ML models:       python train_ml_models.py")
    print("  3. Advanced model:  python train.py")
    print("\nThe new training data is already in the correct location.")
    print("Just run the training script and it will use the augmented dataset.")


if __name__ == "__main__":
    main()
