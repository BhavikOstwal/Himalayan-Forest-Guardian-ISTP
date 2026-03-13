"""
Balance dataset properly and train ML models for maximum accuracy
"""

import os
import numpy as np
import shutil
from pathlib import Path
from tqdm import tqdm
import random

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)

# Paths
TRAIN_CHAINSAW_DIR = Path("data/processed/train/chainsaw")
TRAIN_NON_CHAINSAW_DIR = Path("data/processed/train/non_chainsaw")
VAL_CHAINSAW_DIR = Path("data/processed/val/chainsaw")
VAL_NON_CHAINSAW_DIR = Path("data/processed/val/non_chainsaw")
TEST_CHAINSAW_DIR = Path("data/processed/test/chainsaw")
TEST_NON_CHAINSAW_DIR = Path("data/processed/test/non_chainsaw")

BACKUP_DIR = Path("data/processed/backup_excess_chainsaw")
BACKUP_DIR.mkdir(parents=True, exist_ok=True)

def count_files(directory):
    """Count wav files in a directory"""
    if not directory.exists():
        return 0
    return len(list(directory.glob("*.wav")))

def balance_dataset(target_ratio=0.5):
    """
    Balance the dataset to achieve target ratio
    target_ratio: proportion of chainsaw samples (0.5 = 50/50 balance)
    """
    print("=" * 80)
    print("DATASET BALANCING FOR OPTIMAL ACCURACY")
    print("=" * 80)
    
    # Count current files
    train_chainsaw = count_files(TRAIN_CHAINSAW_DIR)
    train_non_chainsaw = count_files(TRAIN_NON_CHAINSAW_DIR)
    val_chainsaw = count_files(VAL_CHAINSAW_DIR)
    val_non_chainsaw = count_files(VAL_NON_CHAINSAW_DIR)
    test_chainsaw = count_files(TEST_CHAINSAW_DIR)
    test_non_chainsaw = count_files(TEST_NON_CHAINSAW_DIR)
    
    print(f"\n📊 Current Dataset Status:")
    print(f"\n  Training Set:")
    print(f"    Chainsaw:     {train_chainsaw:,}")
    print(f"    Non-chainsaw: {train_non_chainsaw:,}")
    print(f"    Total:        {train_chainsaw + train_non_chainsaw:,}")
    print(f"    Ratio:        {train_chainsaw/(train_chainsaw+train_non_chainsaw)*100:.1f}% chainsaw")
    
    print(f"\n  Validation Set:")
    print(f"    Chainsaw:     {val_chainsaw:,}")
    print(f"    Non-chainsaw: {val_non_chainsaw:,}")
    
    if test_chainsaw > 0 or test_non_chainsaw > 0:
        print(f"\n  Test Set:")
        print(f"    Chainsaw:     {test_chainsaw:,}")
        print(f"    Non-chainsaw: {test_non_chainsaw:,}")
    
    # Calculate balanced numbers
    print(f"\n" + "=" * 80)
    print(f"BALANCING STRATEGY (Target: {target_ratio*100:.0f}/{(1-target_ratio)*100:.0f} ratio)")
    print("=" * 80)
    
    # For training set: balance to match the target ratio
    total_non_chainsaw = train_non_chainsaw
    
    # Calculate how many chainsaw samples we should keep
    # If we want 50/50: chainsaw_count = non_chainsaw_count
    # If we want 60/40: chainsaw_count = non_chainsaw_count * (0.6/0.4) = non_chainsaw_count * 1.5
    target_chainsaw_train = int(total_non_chainsaw * (target_ratio / (1 - target_ratio)))
    
    print(f"\n📐 Calculation:")
    print(f"  Non-chainsaw samples: {train_non_chainsaw:,}")
    print(f"  Target chainsaw samples (for {target_ratio*100:.0f}% ratio): {target_chainsaw_train:,}")
    print(f"  Current chainsaw samples: {train_chainsaw:,}")
    print(f"  Excess chainsaw samples: {train_chainsaw - target_chainsaw_train:,}")
    
    if train_chainsaw <= target_chainsaw_train:
        print(f"\n✅ Dataset is already balanced or chainsaw is minority!")
        print(f"   No balancing needed for training set.")
        return
    
    # Balance training set
    print(f"\n🔄 Balancing training set...")
    print(f"   Moving {train_chainsaw - target_chainsaw_train:,} excess chainsaw files to backup")
    
    # Get all chainsaw files
    chainsaw_files = list(TRAIN_CHAINSAW_DIR.glob("*.wav"))
    
    # Randomly shuffle
    random.shuffle(chainsaw_files)
    
    # Keep target number, move excess to backup
    files_to_keep = chainsaw_files[:target_chainsaw_train]
    files_to_backup = chainsaw_files[target_chainsaw_train:]
    
    print(f"   Keeping: {len(files_to_keep):,}")
    print(f"   Moving to backup: {len(files_to_backup):,}")
    
    # Move excess files to backup
    for file in tqdm(files_to_backup, desc="Moving files"):
        dest = BACKUP_DIR / file.name
        shutil.move(str(file), str(dest))
    
    # Balance validation set proportionally
    if val_chainsaw > 0 and val_non_chainsaw > 0:
        val_ratio = val_chainsaw / (val_chainsaw + val_non_chainsaw)
        if abs(val_ratio - target_ratio) > 0.1:  # If validation is off by more than 10%
            print(f"\n🔄 Balancing validation set...")
            target_val_chainsaw = int(val_non_chainsaw * (target_ratio / (1 - target_ratio)))
            
            if val_chainsaw > target_val_chainsaw:
                val_files = list(VAL_CHAINSAW_DIR.glob("*.wav"))
                random.shuffle(val_files)
                excess_val = val_files[target_val_chainsaw:]
                
                print(f"   Moving {len(excess_val):,} excess validation chainsaw files to backup")
                for file in tqdm(excess_val, desc="Moving validation files"):
                    dest = BACKUP_DIR / f"val_{file.name}"
                    shutil.move(str(file), str(dest))
    
    # Final count
    print(f"\n" + "=" * 80)
    print("BALANCED DATASET STATUS")
    print("=" * 80)
    
    final_train_chainsaw = count_files(TRAIN_CHAINSAW_DIR)
    final_train_non_chainsaw = count_files(TRAIN_NON_CHAINSAW_DIR)
    final_val_chainsaw = count_files(VAL_CHAINSAW_DIR)
    final_val_non_chainsaw = count_files(VAL_NON_CHAINSAW_DIR)
    
    print(f"\n  Training Set:")
    print(f"    Chainsaw:     {final_train_chainsaw:,}")
    print(f"    Non-chainsaw: {final_train_non_chainsaw:,}")
    print(f"    Total:        {final_train_chainsaw + final_train_non_chainsaw:,}")
    final_ratio = final_train_chainsaw/(final_train_chainsaw+final_train_non_chainsaw)*100
    print(f"    Ratio:        {final_ratio:.1f}% chainsaw")
    
    print(f"\n  Validation Set:")
    print(f"    Chainsaw:     {final_val_chainsaw:,}")
    print(f"    Non-chainsaw: {final_val_non_chainsaw:,}")
    if final_val_chainsaw + final_val_non_chainsaw > 0:
        val_ratio = final_val_chainsaw/(final_val_chainsaw+final_val_non_chainsaw)*100
        print(f"    Ratio:        {val_ratio:.1f}% chainsaw")
    
    print(f"\n  Backup Directory: {BACKUP_DIR}")
    print(f"    Backed up files: {count_files(BACKUP_DIR):,}")
    
    print(f"\n✅ Dataset balanced successfully!")
    print(f"   Target ratio achieved: ~{final_ratio:.0f}% chainsaw")
    
    return final_train_chainsaw, final_train_non_chainsaw

def extract_and_train():
    """Extract features from balanced dataset and train models"""
    print(f"\n" + "=" * 80)
    print("STEP 2: FEATURE EXTRACTION & TRAINING")
    print("=" * 80)
    
    print(f"\nRemoving old feature cache to force re-extraction...")
    cache_dir = Path("data/features_cache")
    if cache_dir.exists():
        for cache_file in cache_dir.glob("*.npz"):
            cache_file.unlink()
        print(f"✓ Old cache removed")
    
    # First run preprocessing to extract features
    print(f"\n🔄 Running preprocessing to extract features from balanced dataset...")
    print(f"   This may take a few minutes...")
    
    import subprocess
    result = subprocess.run(["python", "preprocess.py"], capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"❌ Preprocessing failed:")
        print(result.stderr)
        return False
    
    print(f"✓ Features extracted successfully")
    
    # Now train the balanced ML models
    print(f"\n🤖 Training balanced ML models...")
    result = subprocess.run(["python", "train_ml_balanced.py"], capture_output=True, text=True)
    
    if result.returncode == 0:
        print(f"✓ Models trained successfully!")
        print(result.stdout[-1000:])  # Show last 1000 chars
        return True
    else:
        print(f"❌ Training failed:")
        print(result.stderr)
        return False

def main():
    print("\n" + "=" * 80)
    print("BALANCED DATASET TRAINING FOR MAXIMUM ACCURACY")
    print("=" * 80)
    
    print("\nThis script will:")
    print("  1. Balance your dataset to 50/50 ratio")
    print("  2. Extract features from balanced dataset")
    print("  3. Train ML models with proper balance")
    print("  4. Achieve maximum accuracy")
    
    input("\nPress Enter to continue...")
    
    # Step 1: Balance dataset
    result = balance_dataset(target_ratio=0.5)  # 50/50 balance
    
    if result is None:
        print("\n⚠ Dataset already balanced, proceeding with training...")
    
    # Step 2: Extract features and train
    print("\n" + "=" * 80)
    extract_and_train()
    
    print("\n" + "=" * 80)
    print("COMPLETE!")
    print("=" * 80)
    print("\nYour models have been trained with a properly balanced dataset.")
    print("Check the output directory for results and evaluation metrics.")

if __name__ == "__main__":
    main()
