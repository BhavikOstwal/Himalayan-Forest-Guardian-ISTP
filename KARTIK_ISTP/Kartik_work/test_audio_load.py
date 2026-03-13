import librosa
import numpy as np

test_file = r"data\processed\train\chainsaw\1.wav"

print(f"Testing: {test_file}")
try:
    y, sr = librosa.load(test_file, sr=22050, duration=10)
    print(f"✓ Loaded successfully: {len(y)} samples, sample rate: {sr}")
    print(f"  Audio shape: {y.shape}")
    print(f"  Duration: {len(y)/sr:.2f} seconds")
    
    # Try extracting MFCC
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    print(f"✓ MFCC extracted: {mfccs.shape}")
    
except Exception as e:
    print(f"✗ Error: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
