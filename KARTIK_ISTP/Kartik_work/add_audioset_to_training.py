"""
Add downloaded AudioSet clips to train_processed.csv, keeping the dataset balanced.
Run this after download_audioset_clips.py completes.
"""

import pandas as pd
from pathlib import Path

PROCESSED_DIR     = Path("data/processed")
TRAIN_CSV         = PROCESSED_DIR / "train_processed.csv"

AUDIOSET_CHAINSAW     = Path("data/raw/audioset/chainsaw")
AUDIOSET_NON_CHAINSAW = Path("data/raw/audioset/non_chainsaw")


def collect_wavs(folder, label, label_name):
    rows = []
    for wav in sorted(folder.glob("*.wav")):
        rows.append({
            "file_path":  str(wav),
            "label":      label,
            "label_name": label_name,
        })
    return rows


def main():
    # Load existing training CSV
    df = pd.read_csv(TRAIN_CSV)
    existing_paths = set(df["file_path"].tolist())

    print(f"Existing training samples: {len(df)}")
    print(f"  Chainsaw:     {(df['label'] == 1).sum()}")
    print(f"  Non-chainsaw: {(df['label'] == 0).sum()}")

    # Collect new clips (skip any already in CSV)
    new_chainsaw = [
        r for r in collect_wavs(AUDIOSET_CHAINSAW, 1, "chainsaw")
        if r["file_path"] not in existing_paths
    ]
    new_non_chainsaw = [
        r for r in collect_wavs(AUDIOSET_NON_CHAINSAW, 0, "non_chainsaw")
        if r["file_path"] not in existing_paths
    ]

    print(f"\nNew AudioSet clips found:")
    print(f"  Chainsaw:     {len(new_chainsaw)}")
    print(f"  Non-chainsaw: {len(new_non_chainsaw)}")

    # Add equal numbers from each class (keep balance)
    n_add = min(len(new_chainsaw), len(new_non_chainsaw))
    if n_add == 0:
        print("\nNo new clips to add. Have you run download_audioset_clips.py?")
        return

    rows_to_add = new_chainsaw[:n_add] + new_non_chainsaw[:n_add]
    new_df = pd.DataFrame(rows_to_add)
    combined = pd.concat([df, new_df], ignore_index=True)

    # Shuffle
    combined = combined.sample(frac=1, random_state=42).reset_index(drop=True)

    combined.to_csv(TRAIN_CSV, index=False)

    print(f"\nUpdated training CSV:")
    print(f"  Total:        {len(combined)}")
    print(f"  Chainsaw:     {(combined['label'] == 1).sum()}")
    print(f"  Non-chainsaw: {(combined['label'] == 0).sum()}")
    print(f"\nSaved to: {TRAIN_CSV}")
    print(f"\nNext: run  python train_simple.py  to retrain")


if __name__ == "__main__":
    main()
