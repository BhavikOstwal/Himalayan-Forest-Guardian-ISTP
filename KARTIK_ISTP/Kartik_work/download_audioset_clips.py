"""
Download AudioSet chainsaw and non-chainsaw clips from YouTube.
Uses yt-dlp + imageio-ffmpeg (no system ffmpeg needed).
"""

import os
import sys
import subprocess
import csv
import random
from pathlib import Path

import imageio_ffmpeg

# Full path to yt-dlp in the same venv as this script
YTDLP = str(Path(sys.executable).parent / "yt-dlp.exe")

# ── Config ────────────────────────────────────────────────────────────────────
FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
SAMPLE_RATE = 16000
DURATION = 10          # seconds per clip (AudioSet clips are always 10 s)

CHAINSAW_LABEL   = "/m/01j4z9"
NON_CHAINSAW_TARGETS = 120   # match however many chainsaw clips we download

# Diverse non-chainsaw labels (clearly distinguishable from chainsaw)
NON_CHAINSAW_LABELS = [
    "/m/09x0r",   # Speech
    "/m/04rlf",   # Music
    "/m/0jbk",    # Animal
    "/m/015p6",   # Bird
    "/m/07yv9",   # Vehicle
    "/m/012n7d",  # Waterfall
    "/m/06mb1",   # Rain
    "/m/03k3r",   # Horse
    "/m/01b_21",  # Motorcycle
    "/m/0199g",   # Bicycle
    "/m/0285c",   # Piano
    "/m/07s0s5s", # Boat/Water vehicle
]

META_DIR   = Path("data/raw/audioset_metadata")
TRAIN_META = META_DIR / "balanced_train.csv"
EVAL_META  = META_DIR / "eval.csv"

OUT_CHAINSAW     = Path("data/raw/audioset/chainsaw")
OUT_NON_CHAINSAW = Path("data/raw/audioset/non_chainsaw")

OUT_CHAINSAW.mkdir(parents=True, exist_ok=True)
OUT_NON_CHAINSAW.mkdir(parents=True, exist_ok=True)

# ── Helpers ───────────────────────────────────────────────────────────────────

def parse_metadata(csv_path):
    """Return list of (ytid, start_sec, end_sec, labels_str) rows."""
    rows = []
    with open(csv_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("#") or not line:
                continue
            # Format: YTID, start, end, "label1,label2,..."
            # Labels field may be quoted
            parts = line.split(", ", 3)
            if len(parts) < 4:
                continue
            ytid, start, end, labels = parts
            labels = labels.strip().strip('"')
            rows.append((ytid.strip(), float(start), float(end), labels))
    return rows


def download_clip(ytid, start_sec, out_path):
    """
    Download a 10-sec segment from YouTube using yt-dlp + ffmpeg trim.
    Returns True on success.
    """
    url = f"https://www.youtube.com/watch?v={ytid}"
    tmp_audio = out_path.with_suffix(".tmp.webm")

    # Step 1: download best audio (webm/opus, no video) — no ffmpeg needed here
    dl_cmd = [
        YTDLP,
        "--quiet",
        "--no-warnings",
        "--format", "bestaudio[ext=webm]/bestaudio/best",
        "--output", str(tmp_audio),
        "--no-playlist",
        url,
    ]
    try:
        result = subprocess.run(dl_cmd, capture_output=True, timeout=60)
        if result.returncode != 0 or not tmp_audio.exists():
            return False
    except Exception:
        return False

    # Step 2: trim + convert to mono WAV at 16 kHz
    ffmpeg_cmd = [
        FFMPEG,
        "-y",                        # overwrite
        "-ss", str(start_sec),       # seek to start
        "-i", str(tmp_audio),
        "-t", str(DURATION),         # exactly 10 s
        "-ac", "1",                  # mono
        "-ar", str(SAMPLE_RATE),     # 16 kHz
        "-vn",                       # no video
        str(out_path),
    ]
    try:
        result = subprocess.run(ffmpeg_cmd, capture_output=True, timeout=60)
        success = result.returncode == 0 and out_path.exists()
    except Exception:
        success = False

    if tmp_audio.exists():
        tmp_audio.unlink()

    return success


# ── Main ──────────────────────────────────────────────────────────────────────

def collect_clips(meta_files, label_filter_fn):
    """Return all rows matching the label filter."""
    all_rows = []
    for f in meta_files:
        all_rows.extend(parse_metadata(f))
    return [r for r in all_rows if label_filter_fn(r[3])]


def main():
    meta_files = [TRAIN_META, EVAL_META]

    # ── Chainsaw clips ────────────────────────────────────────────────────────
    chainsaw_rows = collect_clips(
        meta_files,
        lambda labels: CHAINSAW_LABEL in labels
    )
    print(f"Found {len(chainsaw_rows)} chainsaw clips in AudioSet metadata")

    # ── Non-chainsaw clips ────────────────────────────────────────────────────
    non_chainsaw_rows = collect_clips(
        meta_files,
        lambda labels: not any(lbl in labels for lbl in [CHAINSAW_LABEL]) and
                       any(lbl in labels for lbl in NON_CHAINSAW_LABELS)
    )
    random.seed(42)
    random.shuffle(non_chainsaw_rows)
    non_chainsaw_rows = non_chainsaw_rows[:NON_CHAINSAW_TARGETS]
    print(f"Selected {len(non_chainsaw_rows)} non-chainsaw clips")

    # ── Download chainsaw ─────────────────────────────────────────────────────
    print("\n=== Downloading CHAINSAW clips ===")
    chainsaw_ok = 0
    for ytid, start, end, _ in chainsaw_rows:
        out = OUT_CHAINSAW / f"{ytid}_{int(start)}.wav"
        if out.exists():
            print(f"  [skip] {out.name}")
            chainsaw_ok += 1
            continue
        print(f"  Downloading {ytid} @ {start}s ... ", end="", flush=True)
        if download_clip(ytid, start, out):
            print("OK")
            chainsaw_ok += 1
        else:
            print("FAIL")

    # ── Download non-chainsaw ─────────────────────────────────────────────────
    print("\n=== Downloading NON-CHAINSAW clips ===")
    non_chainsaw_ok = 0
    for ytid, start, end, _ in non_chainsaw_rows:
        out = OUT_NON_CHAINSAW / f"{ytid}_{int(start)}.wav"
        if out.exists():
            print(f"  [skip] {out.name}")
            non_chainsaw_ok += 1
            continue
        print(f"  Downloading {ytid} @ {start}s ... ", end="", flush=True)
        if download_clip(ytid, start, out):
            print("OK")
            non_chainsaw_ok += 1
        else:
            print("FAIL")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n=== Done ===")
    print(f"Chainsaw downloaded:     {chainsaw_ok}/{len(chainsaw_rows)}")
    print(f"Non-chainsaw downloaded: {non_chainsaw_ok}/{len(non_chainsaw_rows)}")
    print(f"\nFiles saved to:")
    print(f"  {OUT_CHAINSAW}")
    print(f"  {OUT_NON_CHAINSAW}")
    print(f"\nNext: run  python add_audioset_to_training.py  to merge into train CSV")


if __name__ == "__main__":
    main()
