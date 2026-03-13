"""
Temporal Frequency CNN with Attention Mechanism for Chainsaw Detection
Based on: "A Chainsaw-Sound Recognition Model for Detecting Illegal Logging Activities"
"""

import os
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse
from datetime import datetime

from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import librosa
import pandas as pd


class AttentionModule(nn.Module):
    """Attention mechanism to focus on important temporal-frequency features"""
    
    def __init__(self, channels):
        super(AttentionModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        # Shared MLP
        self.fc1 = nn.Conv2d(channels, channels // 8, 1, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv2d(channels // 8, channels, 1, bias=False)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        avg_out = self.fc2(self.relu(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out) * x


class TemporalFrequencyBlock(nn.Module):
    """Combined temporal and frequency convolution block"""
    
    def __init__(self, in_channels, out_channels):
        super(TemporalFrequencyBlock, self).__init__()
        
        # Temporal convolution (along time axis)
        self.temporal_conv = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 7), padding=(0, 3))
        self.temporal_bn = nn.BatchNorm2d(out_channels)
        
        # Frequency convolution (along frequency axis)
        self.frequency_conv = nn.Conv2d(in_channels, out_channels, kernel_size=(7, 1), padding=(3, 0))
        self.frequency_bn = nn.BatchNorm2d(out_channels)
        
        # Combined convolution
        self.combined_conv = nn.Conv2d(out_channels * 2, out_channels, kernel_size=3, padding=1)
        self.combined_bn = nn.BatchNorm2d(out_channels)
        
        # Attention
        self.attention = AttentionModule(out_channels)
        
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        
    def forward(self, x):
        # Process temporal and frequency separately
        t_out = self.relu(self.temporal_bn(self.temporal_conv(x)))
        f_out = self.relu(self.frequency_bn(self.frequency_conv(x)))
        
        # Concatenate temporal and frequency features
        combined = torch.cat([t_out, f_out], dim=1)
        
        # Combine and apply attention
        out = self.relu(self.combined_bn(self.combined_conv(combined)))
        out = self.attention(out)
        out = self.pool(out)
        
        return out


class TFCNN(nn.Module):
    """
    Temporal Frequency Convolutional Neural Network with Attention
    for Chainsaw Sound Recognition
    """
    
    def __init__(self, num_classes=2):
        super(TFCNN, self).__init__()
        
        # Initial convolution
        self.initial_conv = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.initial_bn = nn.BatchNorm2d(32)
        self.relu = nn.ReLU()
        
        # TF-CNN blocks
        self.tf_block1 = TemporalFrequencyBlock(32, 64)
        self.tf_block2 = TemporalFrequencyBlock(64, 128)
        self.tf_block3 = TemporalFrequencyBlock(128, 256)
        
        # Global pooling
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.global_max_pool = nn.AdaptiveMaxPool2d((1, 1))
        
        # Feature representation module
        self.fc1 = nn.Linear(256 * 2, 512)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 256)
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(256, num_classes)
        
    def forward(self, x):
        # Initial processing
        x = self.relu(self.initial_bn(self.initial_conv(x)))
        
        # TF-CNN blocks with attention
        x = self.tf_block1(x)
        x = self.tf_block2(x)
        x = self.tf_block3(x)
        
        # Global pooling (both avg and max)
        avg_pool = self.global_avg_pool(x).view(x.size(0), -1)
        max_pool = self.global_max_pool(x).view(x.size(0), -1)
        
        # Concatenate pooled features
        x = torch.cat([avg_pool, max_pool], dim=1)
        
        # Feature representation and classification
        x = self.dropout1(self.relu(self.fc1(x)))
        x = self.dropout2(self.relu(self.fc2(x)))
        x = self.fc3(x)
        
        return x


class AudioDataset(Dataset):
    """Dataset with enhanced mel-spectrogram features"""
    
    def __init__(self, metadata_file, config):
        self.df = pd.read_csv(metadata_file)
        self.sample_rate = config['audio']['sample_rate']
        self.duration = config['audio']['duration']
        self.n_mels = config['audio']['n_mels']
        self.hop_length = config['audio']['hop_length']
        self.n_fft = config['audio']['n_fft']
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Load audio using soundfile (faster, no numba issues)
        import soundfile as sf
        audio, sr = sf.read(row['file_path'])
        
        # Resample if needed
        if sr != self.sample_rate:
            import scipy.signal as sps
            number_of_samples = round(len(audio) * float(self.sample_rate) / sr)
            audio = sps.resample(audio, number_of_samples)
            sr = self.sample_rate
        
        # Pad or trim to duration
        target_length = int(self.sample_rate * self.duration)
        if len(audio) < target_length:
            audio = np.pad(audio, (0, target_length - len(audio)))
        else:
            audio = audio[:target_length]
        
        # Create mel spectrogram using torchaudio (avoids librosa/numba issue)
        import torchaudio.transforms as T
        audio_tensor = torch.FloatTensor(audio)
        
        mel_transform = T.MelSpectrogram(
            sample_rate=sr,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            f_max=8000
        )
        
        mel_spec = mel_transform(audio_tensor)
        
        # Convert to log scale
        mel_spec_db = torch.log(mel_spec + 1e-9)
        
        # Normalize
        mel_spec_db = (mel_spec_db - mel_spec_db.mean()) / (mel_spec_db.std() + 1e-8)
        
        # Add channel dimension
        mel_spec_db = mel_spec_db.unsqueeze(0)
        
        label = torch.LongTensor([row['label']])[0]
        
        return mel_spec_db, label


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    pbar = tqdm(dataloader, desc="Training")
    for inputs, labels in pbar:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        preds = outputs.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        pbar.set_postfix({'loss': loss.item()})
    
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    
    return avg_loss, accuracy


def validate(model, dataloader, criterion, device):
    """Validate model"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Validating"):
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='binary', zero_division=0
    )
    
    return avg_loss, accuracy, precision, recall, f1


def train_model(config):
    """Main training function"""
    
    # Setup paths
    processed_dir = Path(config['data']['processed_data_dir'])
    output_dir = Path(config['data']['output_dir']) / f"tfcnn_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Load datasets
    print("\n=== Loading Datasets ===")
    train_dataset = AudioDataset(
        processed_dir / 'train_processed.csv',
        config
    )
    val_dataset = AudioDataset(
        processed_dir / 'val_processed.csv',
        config
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=0
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=0
    )
    
    # Create model
    print("\n=== Creating TFCNN Model ===")
    model = TFCNN(num_classes=2)
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Calculate class weights
    train_labels = pd.read_csv(processed_dir / 'train_processed.csv')['label'].values
    class_counts = np.bincount(train_labels)
    class_weights = 1.0 / class_counts
    class_weights = class_weights / class_weights.sum() * len(class_counts)
    class_weights = torch.FloatTensor(class_weights).to(device)
    
    print(f"Class distribution: {class_counts}")
    print(f"Class weights: {class_weights.cpu().numpy()}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config['training']['learning_rate']),
        weight_decay=float(config['training']['weight_decay'])
    )
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3, verbose=True
    )
    
    # Training loop
    print("\n=== Starting Training (TFCNN with Attention) ===")
    best_f1 = 0
    patience_counter = 0
    
    for epoch in range(config['training']['num_epochs']):
        print(f"\nEpoch {epoch + 1}/{config['training']['num_epochs']}")
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        
        # Validate
        val_loss, val_acc, val_precision, val_recall, val_f1 = validate(
            model, val_loader, criterion, device
        )
        
        scheduler.step(val_f1)
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        print(f"Val Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, F1: {val_f1:.4f}")
        
        # Save best model
        if val_f1 > best_f1:
            best_f1 = val_f1
            patience_counter = 0
            
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_f1': val_f1,
                'config': config
            }
            torch.save(checkpoint, output_dir / 'best_model.pt')
            print(f"✓ Saved best model (F1: {val_f1:.4f})")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= config['training']['early_stopping_patience']:
            print(f"\nEarly stopping after {epoch + 1} epochs")
            break
    
    print(f"\n✓ Training complete!")
    print(f"Best F1 score: {best_f1:.4f}")
    print(f"Model saved to: {output_dir}")
    
    return model, output_dir


def main():
    parser = argparse.ArgumentParser(description='Train TFCNN for chainsaw detection')
    parser.add_argument('--config', type=str, default='config.yaml', help='Config file')
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    print("=" * 70)
    print("CHAINSAW DETECTION - TFCNN WITH ATTENTION MECHANISM")
    print("Based on: Simiyu et al. (2024)")
    print("=" * 70)
    
    # Check if processed data exists
    processed_dir = Path(config['data']['processed_data_dir'])
    if not (processed_dir / 'train_processed.csv').exists():
        print("\n⚠ Processed data not found!")
        print("Please run: python preprocess.py")
        return
    
    # Train model
    model, output_dir = train_model(config)
    
    print("\n" + "=" * 70)
    print("NEXT STEPS:")
    print("=" * 70)
    print(f"1. Evaluate: python evaluate_tfcnn.py --model-path {output_dir}/best_model.pt")
    print(f"2. Run inference: python inference_tfcnn.py --model-path {output_dir}/best_model.pt --audio <file>")
    print("=" * 70)


if __name__ == "__main__":
    main()
