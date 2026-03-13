"""
Process Chainsaw_dataset m4a files and prepare balanced training data
"""

import os
import librosa
import soundfile as sf
import numpy as np
from pathlib import Path
from tqdm import tqdm
import shutil
import json

# Paths
CHAINSAW_M4A_DIR = Path("Chainsaw_dataset/Chainsaw_dataset")
TRAIN_CHAINSAW_DIR = Path("data/processed/train/chainsaw")
TRAIN_NON_CHAINSAW_DIR = Path("data/processed/train/non_chainsaw")
VAL_CHAINSAW_DIR = Path("data/processed/val/chainsaw")
VAL_NON_CHAINSAW_DIR = Path("data/processed/val/non_chainsaw")

# Audio parameters
SAMPLE_RATE = 22050
TARGET_DURATION = 5.0  # seconds

def count_files():
    """Count current dataset files"""
    train_chainsaw = len(list(TRAIN_CHAINSAW_DIR.glob("*.wav")))
    train_non_chainsaw = len(list(TRAIN_NON_CHAINSAW_DIR.glob("*.wav")))
    val_chainsaw = len(list(VAL_CHAINSAW_DIR.glob("*.wav")))
    val_non_chainsaw = len(list(VAL_NON_CHAINSAW_DIR.glob("*.wav")))
    m4a_files = len(list(CHAINSAW_M4A_DIR.glob("*.m4a")))
    
    return {
        'train_chainsaw': train_chainsaw,
        'train_non_chainsaw': train_non_chainsaw,
        'val_chainsaw': val_chainsaw,
        'val_non_chainsaw': val_non_chainsaw,
        'm4a_files': m4a_files
    }

def process_m4a_to_wav(m4a_file, output_dir, sample_rate=22050):
    """Convert m4a to wav with proper formatting"""
    try:
        # Load m4a file
        audio, sr = librosa.load(m4a_file, sr=sample_rate, mono=True)
        
        # Check if audio is too short
        if len(audio) < sample_rate * 0.5:  # Less than 0.5 seconds
            return False
        
        # Normalize audio
        if np.max(np.abs(audio)) > 0:
            audio = audio / (np.max(np.abs(audio)) + 1e-8)
        
        # Save as wav
        output_file = output_dir / f"{m4a_file.stem}.wav"
        sf.write(output_file, audio, sample_rate)
        return True
    except Exception as e:
        print(f"Error processing {m4a_file.name}: {e}")
        return False

def main():
    print("=" * 80)
    print("CHAINSAW DATASET PROCESSOR")
    print("=" * 80)
    
    # Count current files
    counts = count_files()
    print(f"\n📊 Current Dataset Status:")
    print(f"  Training:")
    print(f"    - Chainsaw: {counts['train_chainsaw']:,}")
    print(f"    - Non-chainsaw: {counts['train_non_chainsaw']:,}")
    print(f"    - Ratio: {counts['train_chainsaw']/(counts['train_chainsaw']+counts['train_non_chainsaw'])*100:.1f}% chainsaw")
    print(f"  Validation:")
    print(f"    - Chainsaw: {counts['val_chainsaw']:,}")
    print(f"    - Non-chainsaw: {counts['val_non_chainsaw']:,}")
    print(f"\n  📁 Available m4a files: {counts['m4a_files']:,}")
    
    # Ask user how many to process
    print(f"\n" + "=" * 80)
    print("PROCESSING OPTIONS")
    print("=" * 80)
    
    # Calculate balanced amount
    current_non_chainsaw = counts['train_non_chainsaw']
    # For 50/50 balance, we need equal chainsaw and non-chainsaw
    # We have 1399 non-chainsaw, so ideally we want ~1400-1500 chainsaw
    # But we already have 6788 chainsaw, so we might want to use fewer from the m4a files
    
    # Option 1: Use all m4a files (will create imbalance favoring chainsaw)
    # Option 2: Use only enough to match non-chainsaw count
    # Option 3: Use a reasonable subset (e.g., 2000-3000)
    
    print(f"\nRecommendation:")
    print(f"  - You have {current_non_chainsaw:,} non-chainsaw samples")
    print(f"  - You already have {counts['train_chainsaw']:,} chainsaw samples")
    print(f"  - For better balance, consider:")
    print(f"    a) Using 500-1000 more chainsaw clips (for diversity)")
    print(f"    b) The ML training will handle class imbalance with SMOTE")
    
    num_to_process = 1000  # Default: add 1000 more diverse chainsaw samples
    
    print(f"\n🔄 Processing {num_to_process:,} m4a files...")
    print("=" * 80)
    
    # Get list of m4a files
    m4a_files = list(CHAINSAW_M4A_DIR.glob("*.m4a"))
    
    # Randomly select files to add diversity
    np.random.seed(42)
    if len(m4a_files) > num_to_process:
        selected_files = np.random.choice(m4a_files, num_to_process, replace=False)
    else:
        selected_files = m4a_files
    
    # Process files
    successful = 0
    failed = 0
    
    print("\nConverting m4a to wav format...")
    for m4a_file in tqdm(selected_files, desc="Processing"):
        # 90% to train, 10% to validation
        if np.random.random() < 0.9:
            output_dir = TRAIN_CHAINSAW_DIR
        else:
            output_dir = VAL_CHAINSAW_DIR
        
        if process_m4a_to_wav(m4a_file, output_dir):
            successful += 1
        else:
            failed += 1
    
    print(f"\n✅ Successfully processed: {successful:,}")
    print(f"❌ Failed: {failed:,}")
    
    # Count files after processing
    final_counts = count_files()
    
    print(f"\n" + "=" * 80)
    print("FINAL DATASET STATUS")
    print("=" * 80)
    print(f"\n📊 Updated Dataset:")
    print(f"  Training:")
    print(f"    - Chainsaw: {final_counts['train_chainsaw']:,}")
    print(f"    - Non-chainsaw: {final_counts['train_non_chainsaw']:,}")
    total_train = final_counts['train_chainsaw'] + final_counts['train_non_chainsaw']
    print(f"    - Total: {total_train:,}")
    print(f"    - Ratio: {final_counts['train_chainsaw']/total_train*100:.1f}% chainsaw")
    print(f"  Validation:")
    print(f"    - Chainsaw: {final_counts['val_chainsaw']:,}")
    print(f"    - Non-chainsaw: {final_counts['val_non_chainsaw']:,}")
    
    print(f"\n" + "=" * 80)
    print("NEXT STEP: TRAIN ML MODELS")
    print("=" * 80)
    print("\nRun: python train_ml_balanced.py")
    print("\nThe balanced training script will handle class imbalance using:")
    print("  - SMOTE (Synthetic Minority Over-sampling)")
    print("  - Class weights")
    print("  - Balanced Random Forest")
    print("  - This should give you excellent accuracy!")
    
    # Save processing log
    log = {
        'files_processed': successful,
        'files_failed': failed,
        'before': counts,
        'after': final_counts
    }
    
    with open('chainsaw_processing_log.json', 'w') as f:
        json.dump(log, f, indent=2)
    print(f"\n📝 Processing log saved to: chainsaw_processing_log.json")

if __name__ == "__main__":
    main()
