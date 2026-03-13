# Chainsaw Audio Collection Guide

## Quick Commands to Download Chainsaw Sounds

### Option 1: Search and Download from YouTube

```bash
# Search for chainsaw videos
yt-dlp "ytsearch10:chainsaw sound" -x --audio-format wav -o "data/raw/chainsaw_youtube/%(title)s.%(ext)s"

# Search for tree cutting
yt-dlp "ytsearch5:tree cutting chainsaw" -x --audio-format wav -o "data/raw/chainsaw_youtube/%(title)s.%(ext)s"

# Search for logging sounds
yt-dlp "ytsearch5:logging chainsaw" -x --audio-format wav -o "data/raw/chainsaw_youtube/%(title)s.%(ext)s"
```

### Option 2: Download from Specific URLs

```bash
# Single video
yt-dlp -x --audio-format wav -o "data/raw/chainsaw/%(title)s.%(ext)s" "https://youtube.com/watch?v=VIDEO_ID"

# Multiple videos (create a file urls.txt with one URL per line)
yt-dlp -x --audio-format wav -o "data/raw/chainsaw/%(title)s.%(ext)s" -a urls.txt
```

### Option 3: Download from Playlists

```bash
# Download entire playlist
yt-dlp -x --audio-format wav -o "data/raw/chainsaw/%(playlist_index)s-%(title)s.%(ext)s" "PLAYLIST_URL"
```

## Recommended YouTube Search Terms

1. "chainsaw sound effect"
2. "tree cutting chainsaw"
3. "logger chainsaw"
4. "timber chainsaw"
5. "wood cutting chainsaw"
6. "chainsaw compilation"
7. "husqvarna chainsaw"
8. "stihl chainsaw sound"

## Example Video IDs (as of 2024)

Search YouTube for these types of videos:
- Chainsaw sound effects (10-hour compilations)
- Professional logging videos
- Chainsaw review videos
- Forest logging operations

## After Downloading

1. **Listen to all files** - Ensure they actually contain chainsaw sounds
2. **Remove bad files** - Delete music, talking, or non-chainsaw sounds
3. **Trim if needed** - Use audacity or ffmpeg to extract only chainsaw parts
4. **Move to training folder**:
   ```bash
   # PowerShell
   Move-Item data/raw/chainsaw_youtube/*.wav data/raw/chainsaw/
   ```

## Alternative: Free Sound Websites

### Freesound.org
```
1. Visit: https://freesound.org
2. Search: "chainsaw"
3. Filter: License (Creative Commons)
4. Download WAV files
5. Save to: data/raw/chainsaw/
```

### YouTube Audio Library
```
1. Visit: https://www.youtube.com/audiolibrary
2. Search sound effects for "chainsaw", "saw", "woodcutting"
3. Download directly
```

### Other Sources
- **AudioJungle** (paid): High-quality chainsaw sound effects
- **Soundsnap** (subscription): Professional sound library
- **Zapsplat** (free): Sound effects including tools/machinery

## Improving Your Dataset

### Target Numbers:
- **Minimum**: 100 chainsaw samples (25x what you have now)
- **Good**: 200-300 samples
- **Excellent**: 500+ samples

### Augmentation Ideas:
Once you have 100+ samples, the preprocessing script will:
- Apply time stretching
- Apply pitch shifting  
- Add background noise
- Create variations

This effectively multiplies your dataset!

## Quick Start Script

Run the automated downloader:
```bash
python download_chainsaw_youtube.py
```

Or manual download:
```bash
# Download 20 chainsaw sound videos
yt-dlp "ytsearch20:chainsaw sound effect" -x --audio-format wav -o "data/raw/chainsaw_youtube/%(title)s.%(ext)s"
```

Then verify and move good files to the training directory.
