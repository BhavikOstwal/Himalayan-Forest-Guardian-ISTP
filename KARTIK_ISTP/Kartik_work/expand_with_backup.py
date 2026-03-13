"""
Use backup chainsaw files to expand training set while maintaining balance
"""
from pathlib import Path
import shutil
from tqdm import tqdm

# Paths
backup_dir = Path('data/processed/backup_excess_chainsaw')
chainsaw_dir = Path('data/processed/train/chainsaw')
non_chainsaw_dir = Path('data/processed/train/non_chainsaw')

# Get current counts  
current_chainsaw = len(list(chainsaw_dir.glob('*.wav')))
current_non_chainsaw = len(list(non_chainsaw_dir.glob('*.wav')))
backup_files = list(backup_dir.glob('*.wav'))

print(f"\n=== Dataset Expansion ===")
print(f"\nCurrent training set:")
print(f"  Chainsaw: {current_chainsaw}")
print(f"  Non-chainsaw: {current_non_chainsaw}")
print(f"  Ratio: {current_chainsaw*100/(current_chainsaw+current_non_chainsaw):.1f}% chainsaw")

print(f"\nAvailable backup chainsaw files: {len(backup_files)}")

# Strategy: To maximize data while maintaining balance:
# 1. If we have more non-chainsaw, add ALL backup chainsaw files and it will still be balanced
# 2. If we have more chainsaw, only add enough to match or slightly exceed non-chainsaw

target_chainsaw = current_non_chainsaw  # Match non-chainsaw for perfect balance

# Calculate how many we need to add
needed = target_chainsaw - current_chainsaw

if needed <= 0:
    print(f"\nDataset already has enough chainsaw files!")
    print(f"Current ratio: {current_chainsaw}/{current_non_chainsaw}")
else:
    files_to_add = min(needed, len(backup_files))
    print(f"\nAdding {files_to_add} chainsaw files from backup to achieve balance...")
    
    # Copy files from backup to training
    for wav_file in tqdm(backup_files[:files_to_add], desc="Copying files"):
        dest = chainsaw_dir / wav_file.name
        shutil.copy2(str(wav_file), str(dest))
    
    # Verify final counts
    final_chainsaw = len(list(chainsaw_dir.glob('*.wav')))
    final_non_chainsaw = len(list(non_chainsaw_dir.glob('*.wav')))
    
    print(f"\n✓ Dataset expanded and balanced!")
    print(f"  Chainsaw: {current_chainsaw} → {final_chainsaw} (+{final_chainsaw-current_chainsaw})")
    print(f"  Non-chainsaw: {final_non_chainsaw}")
    print(f"  Total: {final_chainsaw + final_non_chainsaw}")
    print(f"  Ratio: {final_chainsaw*100/(final_chainsaw+final_non_chainsaw):.1f}% chainsaw")
    
    print(f"\n✓ Ready to train with {final_chainsaw + final_non_chainsaw} samples!")
