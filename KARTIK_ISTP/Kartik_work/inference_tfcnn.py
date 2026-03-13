"""
Inference script for TFCNN chainsaw detection
"""

import torch
import torch.nn as nn
import librosa
import numpy as np
import argparse
import yaml
from pathlib import Path

from train_tfcnn import TFCNN


class ChainsawDetectorTFCNN:
    """TFCNN-based chainsaw detector"""
    
    def __init__(self, model_path, config_path='config.yaml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"Loading TFCNN model from: {model_path}")
        checkpoint = torch.load(model_path, map_location=self.device)
        
        self.model = TFCNN(num_classes=2)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        self.sample_rate = self.config['audio']['sample_rate']
        self.duration = self.config['audio']['duration']
        self.n_mels = self.config['audio']['n_mels']
        self.hop_length = self.config['audio']['hop_length']
        self.n_fft = self.config['audio']['n_fft']
        
        print(f"✓ TFCNN model loaded on {self.device}")
    
    def preprocess_audio(self, audio_path):
        """Load and preprocess audio"""
        audio, sr = librosa.load(audio_path, sr=self.sample_rate, duration=self.duration)
        
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=sr,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            fmax=8000
        )
        
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        mel_spec_db = (mel_spec_db - mel_spec_db.mean()) / (mel_spec_db.std() + 1e-8)
        mel_spec_tensor = torch.FloatTensor(mel_spec_db).unsqueeze(0).unsqueeze(0)
        
        return mel_spec_tensor
    
    def predict(self, audio_path, return_probs=False):
        """Predict chainsaw presence"""
        inputs = self.preprocess_audio(audio_path)
        inputs = inputs.to(self.device)
        
        with torch.no_grad():
            outputs = self.model(inputs)
            probs = torch.softmax(outputs, dim=-1)[0]
            prediction = torch.argmax(outputs, dim=-1).item()
        
        if return_probs:
            return {
                'non_chainsaw': probs[0].item(),
                'chainsaw': probs[1].item()
            }
        else:
            return prediction
    
    def predict_batch(self, audio_paths):
        """Batch prediction"""
        results = []
        
        for audio_path in audio_paths:
            try:
                probs = self.predict(audio_path, return_probs=True)
                results.append({
                    'file': audio_path,
                    'prediction': 'chainsaw' if probs['chainsaw'] > 0.5 else 'non_chainsaw',
                    'confidence': max(probs['chainsaw'], probs['non_chainsaw']),
                    'probabilities': probs
                })
            except Exception as e:
                results.append({
                    'file': audio_path,
                    'prediction': 'error',
                    'error': str(e)
                })
        
        return results
    
    def predict_long_audio(self, audio_path, window_size=10, hop_size=5):
        """Sliding window prediction"""
        audio, sr = librosa.load(audio_path, sr=self.sample_rate)
        
        window_samples = window_size * self.sample_rate
        hop_samples = hop_size * self.sample_rate
        
        results = []
        
        for start in range(0, len(audio) - window_samples + 1, hop_samples):
            end = start + window_samples
            audio_segment = audio[start:end]
            
            mel_spec = librosa.feature.melspectrogram(
                y=audio_segment,
                sr=sr,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                n_mels=self.n_mels,
                fmax=8000
            )
            
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            mel_spec_db = (mel_spec_db - mel_spec_db.mean()) / (mel_spec_db.std() + 1e-8)
            
            inputs = torch.FloatTensor(mel_spec_db).unsqueeze(0).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(inputs)
                probs = torch.softmax(outputs, dim=-1)[0]
            
            results.append({
                'start_time': start / self.sample_rate,
                'end_time': end / self.sample_rate,
                'chainsaw_probability': probs[1].item(),
                'is_chainsaw': probs[1].item() > 0.5
            })
        
        return results


def main():
    parser = argparse.ArgumentParser(description='TFCNN inference')
    parser.add_argument('--model-path', type=str, required=True)
    parser.add_argument('--audio', type=str, help='Single audio file')
    parser.add_argument('--audio-dir', type=str, help='Directory of audio files')
    parser.add_argument('--output', type=str, help='Output CSV file')
    parser.add_argument('--long-audio', action='store_true')
    parser.add_argument('--threshold', type=float, default=0.5)
    
    args = parser.parse_args()
    
    detector = ChainsawDetectorTFCNN(args.model_path)
    
    if args.audio:
        if args.long_audio:
            print(f"\nAnalyzing long audio: {args.audio}")
            results = detector.predict_long_audio(args.audio)
            
            print("\n" + "=" * 70)
            print("RESULTS - SLIDING WINDOW ANALYSIS (TFCNN)")
            print("=" * 70)
            
            chainsaw_segments = [r for r in results if r['is_chainsaw']]
            
            if chainsaw_segments:
                print(f"⚠ CHAINSAW DETECTED in {len(chainsaw_segments)} segments!")
                for segment in chainsaw_segments:
                    print(f"  Time: {segment['start_time']:.1f}s - {segment['end_time']:.1f}s "
                          f"(confidence: {segment['chainsaw_probability']:.3f})")
            else:
                print("✓ No chainsaw detected")
            
            if args.output:
                import pandas as pd
                pd.DataFrame(results).to_csv(args.output, index=False)
                print(f"\n✓ Results saved to: {args.output}")
        
        else:
            print(f"\nAnalyzing: {args.audio}")
            probs = detector.predict(args.audio, return_probs=True)
            
            print("\n" + "=" * 70)
            print("RESULTS - TFCNN MODEL")
            print("=" * 70)
            print(f"Non-Chainsaw: {probs['non_chainsaw']:.3f}")
            print(f"Chainsaw:     {probs['chainsaw']:.3f}")
            
            if probs['chainsaw'] > args.threshold:
                print(f"\n⚠ CHAINSAW DETECTED (confidence: {probs['chainsaw']:.3f})")
            else:
                print(f"\n✓ No chainsaw detected")
    
    elif args.audio_dir:
        audio_files = []
        audio_dir = Path(args.audio_dir)
        
        for ext in ['.wav', '.mp3', '.flac', '.ogg']:
            audio_files.extend(audio_dir.glob(f'*{ext}'))
        
        print(f"\nProcessing {len(audio_files)} audio files with TFCNN...")
        results = detector.predict_batch([str(f) for f in audio_files])
        
        print("\n" + "=" * 70)
        print("BATCH RESULTS")
        print("=" * 70)
        
        chainsaw_count = sum(1 for r in results if r.get('prediction') == 'chainsaw')
        print(f"Total files: {len(results)}")
        print(f"Chainsaw detected: {chainsaw_count}")
        print(f"No chainsaw: {len(results) - chainsaw_count}")
        
        if args.output:
            import pandas as pd
            pd.DataFrame(results).to_csv(args.output, index=False)
            print(f"\n✓ Results saved to: {args.output}")


if __name__ == "__main__":
    main()
