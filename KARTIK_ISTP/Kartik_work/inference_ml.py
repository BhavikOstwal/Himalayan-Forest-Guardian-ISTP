"""
Inference script for Chainsaw Detection using trained ML models
Load a trained model and make predictions on new audio files
"""

import os
import yaml
import numpy as np
import librosa
import pickle
import argparse
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


class ChainsawDetector:
    """Lightweight chainsaw detection using trained ML model"""
    
    def __init__(self, model_dir):
        """
        Initialize detector with trained model
        
        Args:
            model_dir: Path to directory containing best_model.pkl and scaler.pkl
        """
        self.model_dir = Path(model_dir)
        
        # Load model
        with open(self.model_dir / 'best_model.pkl', 'rb') as f:
            self.model = pickle.load(f)
        print(f"Loaded model from: {self.model_dir / 'best_model.pkl'}")
        
        # Load scaler
        with open(self.model_dir / 'scaler.pkl', 'rb') as f:
            self.scaler = pickle.load(f)
        print(f"Loaded scaler from: {self.model_dir / 'scaler.pkl'}")
        
        # Load config
        with open('config.yaml', 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.sample_rate = self.config['audio']['sample_rate']
        self.duration = self.config['audio']['duration']
        self.n_mels = 40
        self.n_mfcc = 13
        self.hop_length = 512
        self.n_fft = 2048
        
        print("Detector initialized!")
    
    def extract_features(self, audio_path):
        """Extract features from audio file"""
        try:
            # Load audio
            audio, sr = librosa.load(audio_path, sr=self.sample_rate, 
                                    duration=self.duration)
            
            features = []
            
            # 1. MFCCs
            mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=self.n_mfcc)
            features.extend(np.mean(mfccs, axis=1))
            features.extend(np.std(mfccs, axis=1))
            features.extend(np.max(mfccs, axis=1))
            features.extend(np.min(mfccs, axis=1))
            
            # 2. Spectral features
            spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)
            features.append(np.mean(spectral_centroid))
            features.append(np.std(spectral_centroid))
            
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)
            features.append(np.mean(spectral_bandwidth))
            features.append(np.std(spectral_bandwidth))
            
            spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)
            features.append(np.mean(spectral_rolloff))
            features.append(np.std(spectral_rolloff))
            
            spectral_contrast = librosa.feature.spectral_contrast(y=audio, sr=sr, n_bands=6)
            features.extend(np.mean(spectral_contrast, axis=1))
            
            # 3. Zero crossing rate
            zcr = librosa.feature.zero_crossing_rate(audio)
            features.append(np.mean(zcr))
            features.append(np.std(zcr))
            
            # 4. RMS Energy
            rms = librosa.feature.rms(y=audio)
            features.append(np.mean(rms))
            features.append(np.std(rms))
            features.append(np.max(rms))
            features.append(np.min(rms))
            
            # 5. Chroma features
            chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
            features.extend(np.mean(chroma, axis=1))
            
            # 6. Mel spectrogram statistics
            mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, 
                                                     n_mels=self.n_mels)
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            features.append(np.mean(mel_spec_db))
            features.append(np.std(mel_spec_db))
            features.append(np.max(mel_spec_db))
            features.append(np.min(mel_spec_db))
            
            # 7. Tempo
            tempo, _ = librosa.beat.beat_track(y=audio, sr=sr)
            features.append(float(tempo))
            
            # 8. Time-domain features
            features.append(np.mean(np.abs(audio)))
            features.append(np.std(audio))
            features.append(np.max(np.abs(audio)))
            
            return np.array(features).reshape(1, -1)
            
        except Exception as e:
            print(f"Error extracting features: {e}")
            return None
    
    def predict(self, audio_path, return_proba=False):
        """
        Make prediction on audio file
        
        Args:
            audio_path: Path to audio file
            return_proba: If True, return probability score instead of binary prediction
            
        Returns:
            prediction: 0 (non-chainsaw) or 1 (chainsaw), or probability if return_proba=True
        """
        # Extract features
        features = self.extract_features(audio_path)
        
        if features is None:
            return None
        
        # Normalize
        features_scaled = self.scaler.transform(features)
        
        # Predict
        if return_proba and hasattr(self.model, 'predict_proba'):
            proba = self.model.predict_proba(features_scaled)[0]
            return proba[1]  # Probability of chainsaw class
        else:
            prediction = self.model.predict(features_scaled)[0]
            return int(prediction)
    
    def predict_batch(self, audio_files):
        """
        Make predictions on multiple audio files
        
        Args:
            audio_files: List of audio file paths
            
        Returns:
            List of (filename, prediction, probability) tuples
        """
        results = []
        
        for audio_path in audio_files:
            pred = self.predict(audio_path)
            proba = self.predict(audio_path, return_proba=True) if hasattr(self.model, 'predict_proba') else None
            
            label = "CHAINSAW" if pred == 1 else "NON-CHAINSAW"
            results.append((os.path.basename(audio_path), pred, proba, label))
        
        return results


def main():
    parser = argparse.ArgumentParser(description='Chainsaw Detection Inference')
    parser.add_argument('--model_dir', type=str, required=True,
                       help='Directory containing trained model')
    parser.add_argument('--audio_file', type=str, default=None,
                       help='Single audio file to predict')
    parser.add_argument('--audio_dir', type=str, default=None,
                       help='Directory of audio files to predict')
    parser.add_argument('--output', type=str, default=None,
                       help='Output CSV file for batch predictions')
    
    args = parser.parse_args()
    
    # Initialize detector
    detector = ChainsawDetector(args.model_dir)
    
    if args.audio_file:
        # Single file prediction
        print(f"\nProcessing: {args.audio_file}")
        pred = detector.predict(args.audio_file)
        proba = detector.predict(args.audio_file, return_proba=True)
        
        label = "CHAINSAW" if pred == 1 else "NON-CHAINSAW"
        
        print("\n" + "="*50)
        print(f"Prediction: {label}")
        if proba is not None:
            print(f"Confidence: {proba:.2%}")
        print("="*50)
        
    elif args.audio_dir:
        # Batch prediction
        audio_files = []
        for ext in ['*.wav', '*.mp3', '*.flac', '*.ogg']:
            audio_files.extend(Path(args.audio_dir).glob(ext))
        
        print(f"\nFound {len(audio_files)} audio files")
        print("Processing...")
        
        results = detector.predict_batch(audio_files)
        
        # Display results
        print("\n" + "="*70)
        print(f"{'Filename':<40} {'Prediction':<15} {'Confidence':<10}")
        print("="*70)
        
        for filename, pred, proba, label in results:
            proba_str = f"{proba:.2%}" if proba is not None else "N/A"
            print(f"{filename:<40} {label:<15} {proba_str:<10}")
        
        print("="*70)
        
        # Count statistics
        chainsaw_count = sum(1 for _, pred, _, _ in results if pred == 1)
        print(f"\nSummary:")
        print(f"  Total files: {len(results)}")
        print(f"  Chainsaw detected: {chainsaw_count}")
        print(f"  Non-chainsaw: {len(results) - chainsaw_count}")
        
        # Save to CSV if requested
        if args.output:
            import pandas as pd
            df = pd.DataFrame(results, columns=['filename', 'prediction', 'probability', 'label'])
            df.to_csv(args.output, index=False)
            print(f"\nResults saved to: {args.output}")
    
    else:
        print("Please provide either --audio_file or --audio_dir")


if __name__ == "__main__":
    main()
