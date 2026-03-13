"""
Audio preprocessing and data preparation
"""

import os
import yaml
import numpy as np
import pandas as pd
import librosa
import soundfile as sf
from pathlib import Path
from tqdm import tqdm
import torch
import torchaudio
from sklearn.model_selection import train_test_split
import argparse
import json


class AudioPreprocessor:
    def __init__(self, config):
        self.config = config
        self.sample_rate = config['audio']['sample_rate']
        self.duration = config['audio']['duration']
        self.target_length = self.sample_rate * self.duration
        
    def load_audio(self, file_path):
        """Load audio file and resample if needed"""
        try:
            audio, sr = librosa.load(file_path, sr=self.sample_rate, duration=self.duration)
            return audio, sr
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return None, None
    
    def pad_or_trim(self, audio):
        """Pad or trim audio to target length"""
        if len(audio) < self.target_length:
            # Pad with zeros
            padding = self.target_length - len(audio)
            audio = np.pad(audio, (0, padding), mode='constant')
        else:
            # Trim
            audio = audio[:self.target_length]
        return audio
    
    def apply_augmentation(self, audio, apply=True):
        """Apply data augmentation"""
        if not apply:
            return audio
        
        aug_config = self.config['augmentation']
        
        # Time stretching
        if np.random.random() < aug_config['apply_prob']:
            rate = np.random.uniform(*aug_config['time_stretch_range'])
            audio = librosa.effects.time_stretch(audio, rate=rate)
            audio = self.pad_or_trim(audio)  # Ensure correct length
        
        # Pitch shifting
        if np.random.random() < aug_config['apply_prob']:
            n_steps = np.random.uniform(*aug_config['pitch_shift_range'])
            audio = librosa.effects.pitch_shift(
                audio, sr=self.sample_rate, n_steps=n_steps
            )
        
        # Add noise
        if np.random.random() < aug_config['apply_prob']:
            noise = np.random.randn(len(audio)) * aug_config['noise_factor']
            audio = audio + noise
        
        return audio
    
    def save_processed_audio(self, audio, output_path):
        """Save processed audio"""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        sf.write(output_path, audio, self.sample_rate)


def organize_dataset_from_raw(config):
    """
    Organize raw audio files into train/val/test splits
    Assumes you have organized your files with labels
    """
    print("\n=== Organizing Dataset ===")
    
    # Create metadata file
    metadata = []
    
    raw_dir = Path(config['data']['raw_data_dir'])
    
    # Example: scan for audio files and their labels
    # You would customize this based on your dataset structure
    
    audio_extensions = ['.wav', '.mp3', '.flac', '.ogg']
    
    for audio_file in raw_dir.rglob('*'):
        if audio_file.suffix.lower() in audio_extensions:
            # Determine label based on filename or directory
            # This is just an example - adjust based on your data
            if 'chainsaw' in str(audio_file).lower():
                label = 1
                label_name = 'chainsaw'
            else:
                label = 0
                label_name = 'non_chainsaw'
            
            metadata.append({
                'file_path': str(audio_file),
                'label': label,
                'label_name': label_name
            })
    
    if len(metadata) == 0:
        print("⚠ No audio files found. Please organize your raw data first.")
        print("Place chainsaw sounds in a 'chainsaw' folder and other sounds in 'non_chainsaw' folder")
        return None
    
    df = pd.DataFrame(metadata)
    
    # Split data
    train_df, temp_df = train_test_split(
        df, test_size=(1 - config['split']['train']), 
        random_state=config['split']['random_seed'],
        stratify=df['label']
    )
    
    val_size = config['split']['val'] / (config['split']['val'] + config['split']['test'])
    val_df, test_df = train_test_split(
        temp_df, test_size=(1 - val_size),
        random_state=config['split']['random_seed'],
        stratify=temp_df['label']
    )
    
    print(f"Train samples: {len(train_df)}")
    print(f"Val samples: {len(val_df)}")
    print(f"Test samples: {len(test_df)}")
    
    # Save splits
    processed_dir = Path(config['data']['processed_data_dir'])
    train_df.to_csv(processed_dir / 'train.csv', index=False)
    val_df.to_csv(processed_dir / 'val.csv', index=False)
    test_df.to_csv(processed_dir / 'test.csv', index=False)
    
    return train_df, val_df, test_df


def preprocess_split(df, preprocessor, output_dir, split_name, apply_augmentation=False):
    """Preprocess a data split"""
    print(f"\nProcessing {split_name} split...")
    
    processed_files = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing {split_name}"):
        # Load audio
        audio, sr = preprocessor.load_audio(row['file_path'])
        
        if audio is None:
            continue
        
        # Pad or trim
        audio = preprocessor.pad_or_trim(audio)
        
        # Apply augmentation (only for training)
        if apply_augmentation and split_name == 'train':
            audio = preprocessor.apply_augmentation(audio, apply=True)
        
        # Save processed audio
        output_path = os.path.join(
            output_dir, split_name, row['label_name'], f"{idx}.wav"
        )
        preprocessor.save_processed_audio(audio, output_path)
        
        processed_files.append({
            'file_path': output_path,
            'label': row['label'],
            'label_name': row['label_name']
        })
    
    # Save metadata
    pd.DataFrame(processed_files).to_csv(
        os.path.join(output_dir, f'{split_name}_processed.csv'), 
        index=False
    )
    
    return processed_files


def main():
    parser = argparse.ArgumentParser(description='Preprocess audio data')
    parser.add_argument('--config', type=str, default='config.yaml', help='Config file')
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    print("=" * 60)
    print("AUDIO PREPROCESSING")
    print("=" * 60)
    
    # Organize dataset
    splits = organize_dataset_from_raw(config)
    
    if splits is None:
        print("\nPlease organize your data first:")
        print("1. Place chainsaw audio files in: data/raw/chainsaw/")
        print("2. Place non-chainsaw audio files in: data/raw/non_chainsaw/")
        print("3. Run this script again")
        return
    
    train_df, val_df, test_df = splits
    
    # Initialize preprocessor
    preprocessor = AudioPreprocessor(config)
    
    # Process each split
    output_dir = config['data']['processed_data_dir']
    
    preprocess_split(train_df, preprocessor, output_dir, 'train', apply_augmentation=True)
    preprocess_split(val_df, preprocessor, output_dir, 'val', apply_augmentation=False)
    preprocess_split(test_df, preprocessor, output_dir, 'test', apply_augmentation=False)
    
    print("\n" + "=" * 60)
    print("✓ Preprocessing complete!")
    print(f"Processed data saved to: {output_dir}")
    print("\nNext step: python train.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
