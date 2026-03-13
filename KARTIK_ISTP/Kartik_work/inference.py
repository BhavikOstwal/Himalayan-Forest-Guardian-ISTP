"""
Inference script for chainsaw detection
"""

import torch
import librosa
import numpy as np
import argparse
from pathlib import Path
from transformers import ASTFeatureExtractor, ASTForAudioClassification


class ChainsawDetector:
    """Chainsaw detection inference class"""
    
    def __init__(self, model_path, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Load model and feature extractor
        print(f"Loading model from: {model_path}")
        self.model = ASTForAudioClassification.from_pretrained(model_path)
        self.feature_extractor = ASTFeatureExtractor.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
        
        self.sample_rate = 16000
        print(f"✓ Model loaded on {self.device}")
    
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
        # Load audio
        audio, sr = librosa.load(audio_path, sr=self.sample_rate)
        
        # Process audio
        inputs = self.feature_extractor(
            audio,
            sampling_rate=self.sample_rate,
            return_tensors="pt"
        )
        
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Predict
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)[0]
            prediction = torch.argmax(logits, dim=-1).item()
        
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
            
            # Process segment
            inputs = self.feature_extractor(
                audio_segment,
                sampling_rate=self.sample_rate,
                return_tensors="pt"
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1)[0]
            
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
