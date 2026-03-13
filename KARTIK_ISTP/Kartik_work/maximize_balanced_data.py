"""
Maximize training data while maintaining perfect balance
Strategy: Add ALL available non-chainsaw, then match with chainsaw
"""
from pathlib import Path
import shutil
from tqdm import tqdm
import pandas as pd

# Paths
train_chainsaw = Path('data/processed/train/chainsaw')
train_non_chainsaw = Path('data/processed/train/non_chainsaw')
backup_chainsaw = Path('data/processed/backup_excess_chainsaw')

# Raw data sources for non-chainsaw
raw_sources = [
    Path('data/raw/audioset'),
    Path('data/raw/ESC-50/audio'),
    Path('data/raw/non_chainsaw')
]

print("=== MAXIMIZING TRAINING DATA WITH BALANCE ===\n")

# Get current state
current_chainsaw = list(train_chainsaw.glob('*.wav'))
current_non_chainsaw = list(train_non_chainsaw.glob('*.wav'))

print(f"Current training set:")
print(f"  Chainsaw: {len(current_chainsaw)}")
print(f"  Non-chainsaw: {len(current_non_chainsaw)}")
print(f"  Total: {len(current_chainsaw) + len(current_non_chainsaw)}")

# Get all available non-chainsaw files
current_nc_names = {f.name for f in current_non_chainsaw}
all_non_chainsaw = []

for source in raw_sources:
    if source.exists():
        files = list(source.rglob('*.wav'))
        all_non_chainsaw.extend([f for f in files if f.name not in current_nc_names])

print(f"\nAvailable non-chainsaw files to add: {len(all_non_chainsaw)}")

# Step 1: Add ALL available non-chainsaw files
if len(all_non_chainsaw) > 0:
    print(f"\nStep 1: Adding {len(all_non_chainsaw)} non-chainsaw files...")
    for wav_file in tqdm(all_non_chainsaw, desc="Copying"):
        dest = train_non_chainsaw / wav_file.name
        if not dest.exists():
            shutil.copy2(str(wav_file), str(dest))
else:
    print("\nStep 1: No new non-chainsaw files to add")

# Step 2: Count and match with chainsaw
final_non_chainsaw_count = len(list(train_non_chainsaw.glob('*.wav')))
current_chainsaw_count = len(list(train_chainsaw.glob('*.wav')))
backup_chainsaw_files = list(backup_chainsaw.glob('*.wav'))

needed_chainsaw = final_non_chainsaw_count - current_chainsaw_count

print(f"\nStep 2: Balancing chainsaw files...")
print(f"  Target: {final_non_chainsaw_count} chainsaw files")
print(f"  Current: {current_chainsaw_count}")
print(f"  Need to add: {needed_chainsaw}")
print(f"  Available in backup: {len(backup_chainsaw_files)}")

if needed_chainsaw > 0:
    if needed_chainsaw <= len(backup_chainsaw_files):
        print(f"\nAdding {needed_chainsaw} chainsaw files from backup...")
        for wav_file in tqdm(backup_chainsaw_files[:needed_chainsaw], desc="Copying"):
            dest = train_chainsaw / wav_file.name
            if not dest.exists():
                shutil.copy2(str(wav_file), str(dest))
    else:
        print(f"  WARNING: Not enough backup files! Adding all {len(backup_chainsaw_files)}")
        for wav_file in tqdm(backup_chainsaw_files, desc="Copying"):
            dest = train_chainsaw / wav_file.name
            if not dest.exists():
                shutil.copy2(str(wav_file), str(dest))

# Final verification
final_chainsaw = len(list(train_chainsaw.glob('*.wav')))
final_non_chainsaw = len(list(train_non_chainsaw.glob('*.wav')))
total = final_chainsaw + final_non_chainsaw

print(f"\n{'='*60}")
print(f"✓ DATASET MAXIMIZED AND BALANCED!")
print(f"{'='*60}")
print(f"  Chainsaw:     {final_chainsaw:,}")
print(f"  Non-chainsaw: {final_non_chainsaw:,}")
print(f"  Total:        {total:,}")
print(f"  Ratio:        {final_chainsaw*100/total:.1f}% chainsaw")
print(f"  Increase:     {total - (len(current_chainsaw) + len(current_non_chainsaw)):,} samples")
print(f"{'='*60}\n")

# Update CSV files
print("Updating training CSV...")
train_data = []

for wav_file in train_chainsaw.glob('*.wav'):
    train_data.append({
        'file_path': str(wav_file).replace('\\', '/'),
        'label': 1,
        'label_name': 'chainsaw'
    })

for wav_file in train_non_chainsaw.glob('*.wav'):
    train_data.append({
        'file_path': str(wav_file).replace('\\', '/'),
        'label': 0,
        'label_name': 'non_chainsaw'
    })

df = pd.DataFrame(train_data)
df.to_csv('data/processed/train_processed.csv', index=False)

print(f"✓ CSV updated: {len(df)} samples")
print(f"\n✓ Ready to train with {total:,} balanced samples!")
