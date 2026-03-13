"""
Balance test set to 50/50 ratio for fair evaluation
"""
import shutil
from pathlib import Path
import random

# Set seed for reproducibility
random.seed(42)

# Define paths
test_chainsaw = Path('data/processed/test/chainsaw')
test_non_chainsaw = Path('data/processed/test/non_chainsaw')
backup_dir = Path('data/processed/backup_excess_test_non_chainsaw')

# Get file counts
chainsaw_files = list(test_chainsaw.glob('*.wav'))
non_chainsaw_files = list(test_non_chainsaw.glob('*.wav'))

chainsaw_count = len(chainsaw_files)
non_chainsaw_count = len(non_chainsaw_files)

print(f"\nCurrent test set distribution:")
print(f"  Chainsaw: {chainsaw_count}")
print(f"  Non-chainsaw: {non_chainsaw_count}")
print(f"  Ratio: {chainsaw_count*100/(chainsaw_count+non_chainsaw_count):.1f}% chainsaw")

# Determine target counts (50/50 ratio)
target_count = chainsaw_count  # Match chainsaw count
excess_count = non_chainsaw_count - target_count

if excess_count > 0:
    print(f"\nBalancing test set...")
    print(f"  Target: {target_count} files per class")
    print(f"  Excess non-chainsaw files: {excess_count}")
    
    # Create backup directory
    backup_dir.mkdir(parents=True, exist_ok=True)
    
    # Randomly select files to move
    random.shuffle(non_chainsaw_files)
    files_to_move = non_chainsaw_files[target_count:]
    
    # Move excess files to backup
    for i, file in enumerate(files_to_move, 1):
        dest = backup_dir / file.name
        shutil.move(str(file), str(dest))
        if i % 50 == 0 or i == len(files_to_move):
            print(f"  Moved {i}/{len(files_to_move)} files...", end='\r')
    
    print(f"\n  ✓ Moved {len(files_to_move)} files to backup")
    
    # Verify new distribution
    new_chainsaw_count = len(list(test_chainsaw.glob('*.wav')))
    new_non_chainsaw_count = len(list(test_non_chainsaw.glob('*.wav')))
    
    print(f"\n✓ Test set balanced successfully!")
    print(f"  Chainsaw: {new_chainsaw_count}")
    print(f"  Non-chainsaw: {new_non_chainsaw_count}")
    print(f"  Ratio: {new_chainsaw_count*100/(new_chainsaw_count+new_non_chainsaw_count):.1f}% chainsaw")
else:
    print(f"\n✓ Test set already balanced!")
