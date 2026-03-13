"""
Add more chainsaw data from m4a files while maintaining balance
"""
from pathlib import Path
from tqdm import tqdm
import shutil
import librosa
import soundfile as sf
import numpy as np

# Paths
m4a_dir = Path('Chainsaw_dataset/Chainsaw_dataset')
output_dir = Path('data/processed/train/chainsaw')
temp_convert_dir = Path('data/processed/train/chainsaw_converted')
temp_convert_dir.mkdir(parents=True, exist_ok=True)

# Get m4a files
m4a_files = list(m4a_dir.glob('*.m4a'))
print(f"\nFound {len(m4a_files)} m4a chainsaw files")

# Check current training set
current_chainsaw = len(list(output_dir.glob('*.wav')))
current_non_chainsaw = len(list(Path('data/processed/train/non_chainsaw').glob('*.wav')))

print(f"\nCurrent training set:")
print(f"  Chainsaw: {current_chainsaw}")
print(f"  Non-chainsaw: {current_non_chainsaw}")
print(f"  Ratio: {current_chainsaw*100/(current_chainsaw+current_non_chainsaw):.1f}% chainsaw")

# Convert m4a to wav using librosa
print(f"\nConverting {len(m4a_files)} m4a files to WAV...")
print("This may take several minutes...")

success_count = 0
failed_count = 0
failed_files = []

for i, m4a_file in enumerate(tqdm(m4a_files, desc="Converting")):
    output_wav = temp_convert_dir / f"youtube_chainsaw_{m4a_file.stem}.wav"
    
    try:
        # Load m4a and save as wav
        audio, sr = librosa.load(str(m4a_file), sr=16000, mono=True)
        
        # Check if audio is valid
        if len(audio) > 0:
            # Save as WAV
            sf.write(str(output_wav), audio, sr)
            success_count += 1
        else:
            failed_count += 1
            if len(failed_files) < 5:
                failed_files.append(f"{m4a_file.name}: empty audio")
    except Exception as e:
        failed_count += 1
        if len(failed_files) < 5:
            failed_files.append(f"{m4a_file.name}: {str(e)[:30]}")

print(f"\n✓ Conversion complete!")
print(f"  Successful: {success_count}")
print(f"  Failed: {failed_count}")
if failed_files:
    print(f"  Example failures: {failed_files[:5]}")

# Now balance the dataset
print(f"\nBalancing dataset...")

# Count total chainsaw files (existing + new)
existing_chainsaw_files = list(output_dir.glob('*.wav'))
new_chainsaw_files = list(temp_convert_dir.glob('*.wav'))
total_chainsaw = len(existing_chainsaw_files) + len(new_chainsaw_files)

print(f"  Existing chainsaw: {len(existing_chainsaw_files)}")
print(f"  New chainsaw: {len(new_chainsaw_files)}")
print(f"  Total chainsaw: {total_chainsaw}")
print(f"  Non-chainsaw: {current_non_chainsaw}")

# Determine target count (match non-chainsaw count for perfect balance)
target_chainsaw_count = current_non_chainsaw

if total_chainsaw <= target_chainsaw_count:
    # Use all chainsaw files
    print(f"\n  Moving all {len(new_chainsaw_files)} new chainsaw files to training set...")
    for wav_file in tqdm(new_chainsaw_files, desc="Moving files"):
        dest = output_dir / wav_file.name
        shutil.move(str(wav_file), str(dest))
    
    final_chainsaw = len(list(output_dir.glob('*.wav')))
    print(f"\n✓ Dataset updated!")
    print(f"  Chainsaw: {final_chainsaw}")
    print(f"  Non-chainsaw: {current_non_chainsaw}")
    print(f"  Ratio: {final_chainsaw*100/(final_chainsaw+current_non_chainsaw):.1f}% chainsaw")
else:
    # Use only enough to match non-chainsaw count
    needed_count = target_chainsaw_count - len(existing_chainsaw_files)
    
    if needed_count > 0:
        print(f"\n  Using {needed_count} out of {len(new_chainsaw_files)} new files to maintain balance...")
        files_to_move = new_chainsaw_files[:needed_count]
        
        for wav_file in tqdm(files_to_move, desc="Moving files"):
            dest = output_dir / wav_file.name
            shutil.move(str(wav_file), str(dest))
        
        # Move excess to backup
        excess_files = new_chainsaw_files[needed_count:]
        if excess_files:
            backup_dir = Path('data/processed/backup_excess_youtube_chainsaw')
            backup_dir.mkdir(parents=True, exist_ok=True)
            print(f"  Moving {len(excess_files)} excess files to backup...")
            for wav_file in excess_files:
                dest = backup_dir / wav_file.name
                shutil.move(str(wav_file), str(dest))
    
    final_chainsaw = len(list(output_dir.glob('*.wav')))
    print(f"\n✓ Dataset balanced!")
    print(f"  Chainsaw: {final_chainsaw}")
    print(f"  Non-chainsaw: {current_non_chainsaw}")
    print(f"  Ratio: {final_chainsaw*100/(final_chainsaw+current_non_chainsaw):.1f}% chainsaw")

# Clean up temp directory
if temp_convert_dir.exists():
    remaining = list(temp_convert_dir.glob('*'))
    if not remaining:
        temp_convert_dir.rmdir()
    else:
        print(f"  Note: {len(remaining)} files remain in {temp_convert_dir}")

print(f"\n✓ Ready to train with expanded balanced dataset!")
