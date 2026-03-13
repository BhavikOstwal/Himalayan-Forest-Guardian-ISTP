"""
Inference script for simple CNN chainsaw detection
"""

import torch
import torch.nn as nn
import librosa
import numpy as np
import argparse
import yaml
from pathlib import Path

class SimpleCNN(nn.Module):
    """Lightweight CNN matching the trained model"""
    
    def __init__(self, num_classes=2):
        super(SimpleCNN, self).__init__()
        
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.5)
        
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, num_classes)
        
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.pool(self.relu(self.bn3(self.conv3(x))))
        
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


class ChainsawDetector:
    """Chainsaw detection inference class"""
    
    def __init__(self, model_path, config_path='config.yaml'):
        # Load config
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model
        print(f"Loading model from: {model_path}")
        checkpoint = torch.load(model_path, map_location=self.device)
        
        self.model = SimpleCNN(num_classes=2)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        # Audio parameters
        self.sample_rate = self.config['audio']['sample_rate']
        self.duration = self.config['audio']['duration']
        self.n_mels = self.config['audio']['n_mels']
        self.hop_length = self.config['audio']['hop_length']
        self.n_fft = self.config['audio']['n_fft']
        
        print(f"✓ Model loaded on {self.device}")
    
    def preprocess_audio(self, audio_path):
        """Load and preprocess audio file"""
        # Load audio
        audio, sr = librosa.load(audio_path, sr=self.sample_rate, duration=self.duration)
        
        # Create mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=sr,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels
        )
        
        # Convert to log scale
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Normalize
        mel_spec_db = (mel_spec_db - mel_spec_db.mean()) / (mel_spec_db.std() + 1e-8)
        
        # Convert to tensor and add batch dimension
        mel_spec_tensor = torch.FloatTensor(mel_spec_db).unsqueeze(0).unsqueeze(0)
        
        return mel_spec_tensor
    
    def predict(self, audio_path, return_probs=False):
        """
        Predict if audio contains chainsaw sound
        
        Args:
            audio_path: Path to audio file
            return_probs: If True, return probabilities instead of class
        
        Returns:
            prediction (int): 0 for non-chainsaw, 1 for chainsaw
            OR
            probabilities (dict): {'non_chainsaw': prob, 'chainsaw': prob}
        """
        # Preprocess audio
        inputs = self.preprocess_audio(audio_path)
        inputs = inputs.to(self.device)
        
        # Predict
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
        """Predict for multiple audio files"""
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
                print(f"Error processing {audio_path}: {e}")
                results.append({
                    'file': audio_path,
                    'prediction': 'error',
                    'error': str(e)
                })
        
        return results
    
    def predict_long_audio(self, audio_path, window_size=10, hop_size=5):
        """
        Predict on long audio by sliding window
        
        Args:
            audio_path: Path to long audio file
            window_size: Window size in seconds
            hop_size: Hop size in seconds
        
        Returns:
            List of predictions for each window with timestamps
        """
        # Load full audio
        audio, sr = librosa.load(audio_path, sr=self.sample_rate)
        
        window_samples = window_size * self.sample_rate
        hop_samples = hop_size * self.sample_rate
        
        results = []
        
        for start in range(0, len(audio) - window_samples + 1, hop_samples):
            end = start + window_samples
            audio_segment = audio[start:end]
            
            # Create mel spectrogram
            mel_spec = librosa.feature.melspectrogram(
                y=audio_segment,
                sr=sr,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                n_mels=self.n_mels
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
    parser = argparse.ArgumentParser(description='Run inference on audio files')
    parser.add_argument('--model-path', type=str, required=True, help='Path to trained model')
    parser.add_argument('--audio', type=str, help='Path to single audio file')
    parser.add_argument('--audio-dir', type=str, help='Path to directory of audio files')
    parser.add_argument('--output', type=str, help='Output file for results')
    parser.add_argument('--long-audio', action='store_true', 
                        help='Process as long audio with sliding window')
    parser.add_argument('--threshold', type=float, default=0.5, 
                        help='Confidence threshold for chainsaw detection')
    
    args = parser.parse_args()
    
    # Initialize detector
    detector = ChainsawDetector(args.model_path)
    
    if args.audio:
        # Single file prediction
        if args.long_audio:
            print(f"\nAnalyzing long audio: {args.audio}")
            results = detector.predict_long_audio(args.audio)
            
            print("\n" + "=" * 60)
            print("RESULTS - SLIDING WINDOW ANALYSIS")
            print("=" * 60)
            
            chainsaw_segments = [r for r in results if r['is_chainsaw']]
            
            if chainsaw_segments:
                print(f"⚠ CHAINSAW DETECTED in {len(chainsaw_segments)} segments!")
                for segment in chainsaw_segments:
                    print(f"  Time: {segment['start_time']:.1f}s - {segment['end_time']:.1f}s "
                          f"(confidence: {segment['chainsaw_probability']:.3f})")
            else:
                print("✓ No chainsaw detected")
            
            # Save results if output specified
            if args.output:
                import pandas as pd
                pd.DataFrame(results).to_csv(args.output, index=False)
                print(f"\n✓ Results saved to: {args.output}")
        
        else:
            print(f"\nAnalyzing: {args.audio}")
            probs = detector.predict(args.audio, return_probs=True)
            
            print("\n" + "=" * 60)
            print("RESULTS")
            print("=" * 60)
            print(f"Non-Chainsaw: {probs['non_chainsaw']:.3f}")
            print(f"Chainsaw:     {probs['chainsaw']:.3f}")
            
            if probs['chainsaw'] > args.threshold:
                print(f"\n⚠ CHAINSAW DETECTED (confidence: {probs['chainsaw']:.3f})")
            else:
                print(f"\n✓ No chainsaw detected")
    
    elif args.audio_dir:
        # Batch prediction
        audio_files = []
        audio_dir = Path(args.audio_dir)
        
        for ext in ['.wav', '.mp3', '.flac', '.ogg']:
            audio_files.extend(audio_dir.glob(f'*{ext}'))
        
        print(f"\nProcessing {len(audio_files)} audio files...")
        results = detector.predict_batch([str(f) for f in audio_files])
        
        # Display results
        print("\n" + "=" * 60)
        print("BATCH RESULTS")
        print("=" * 60)
        
        chainsaw_count = sum(1 for r in results if r.get('prediction') == 'chainsaw')
        print(f"Total files: {len(results)}")
        print(f"Chainsaw detected: {chainsaw_count}")
        print(f"No chainsaw: {len(results) - chainsaw_count}")
        
        # Show individual results
        for result in results:
            if result.get('prediction') == 'chainsaw':
                print(f"\n⚠ {Path(result['file']).name}: CHAINSAW "
                      f"(confidence: {result['confidence']:.3f})")
        
        # Save results
        if args.output:
            import pandas as pd
            pd.DataFrame(results).to_csv(args.output, index=False)
            print(f"\n✓ Results saved to: {args.output}")
    
    else:
        print("Please specify either --audio or --audio-dir")


if __name__ == "__main__":
    main()
