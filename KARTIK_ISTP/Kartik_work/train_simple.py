"""
Simplified training script using a lighter model for CPU training
Uses a simple CNN architecture that trains much faster
"""

import os
import yaml
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse
from datetime import datetime

from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import librosa
import pandas as pd


class SimpleCNN(nn.Module):
    """Deeper CNN with stronger regularization for better generalization"""
    
    def __init__(self, num_classes=2):
        super(SimpleCNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout1 = nn.Dropout(0.4)
        self.dropout2 = nn.Dropout(0.5)
        
        # Fully connected layers
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)
        
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # x shape: [batch, 1, n_mels, time_steps]
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.dropout1(x)
        x = self.pool(self.relu(self.bn3(self.conv3(x))))
        x = self.pool(self.relu(self.bn4(self.conv4(x))))
        
        # Global average pooling
        x = self.global_pool(x)  # [batch, 256, 1, 1]
        x = x.view(x.size(0), -1)  # [batch, 256]
        
        x = self.dropout2(x)
        x = self.relu(self.fc1(x))
        x = self.dropout1(x)
        x = self.relu(self.fc2(x))
        x = self.dropout1(x)
        x = self.fc3(x)
        
        return x


class AudioDataset(Dataset):
    """Dataset with data augmentation for better generalization"""
    
    def __init__(self, metadata_file, config, augment=False):
        self.df = pd.read_csv(metadata_file)
        self.sample_rate = config['audio']['sample_rate']
        self.duration = config['audio']['duration']
        self.n_mels = config['audio']['n_mels']
        self.hop_length = config['audio']['hop_length']
        self.n_fft = config['audio']['n_fft']
        self.augment = augment
        
    def __len__(self):
        return len(self.df)
    
    def augment_audio(self, audio, sr):
        """Apply random augmentations to prevent overfitting"""
        # 1. Time stretching (speed up/slow down)
        if np.random.random() < 0.4:
            rate = np.random.uniform(0.85, 1.15)
            audio = librosa.effects.time_stretch(audio, rate=rate)
        
        # 2. Pitch shifting (different chainsaw RPMs)
        if np.random.random() < 0.4:
            steps = np.random.uniform(-3, 3)
            audio = librosa.effects.pitch_shift(audio, sr=sr, n_steps=steps)
        
        # 3. Add background noise
        if np.random.random() < 0.5:
            noise_level = np.random.uniform(0.002, 0.02)
            noise = np.random.randn(len(audio)) * noise_level
            audio = audio + noise
        
        # 4. Random volume scaling
        if np.random.random() < 0.4:
            gain = np.random.uniform(0.6, 1.4)
            audio = audio * gain
        
        # 5. Random time shift
        if np.random.random() < 0.3:
            shift = int(np.random.uniform(-0.1, 0.1) * len(audio))
            audio = np.roll(audio, shift)
        
        return audio
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        target_length = int(self.sample_rate * self.duration)
        expected_frames = 1 + target_length // self.hop_length
        
        try:
            # Load audio
            audio, sr = librosa.load(
                row['file_path'], 
                sr=self.sample_rate,
                duration=self.duration
            )
            
            # Ensure consistent audio length (pad or crop)
            if len(audio) < target_length:
                audio = np.pad(audio, (0, target_length - len(audio)), mode='constant')
            else:
                audio = audio[:target_length]
            
            # Apply augmentation during training only
            if self.augment:
                audio = self.augment_audio(audio, sr)
                # Re-pad/crop after augmentation
                if len(audio) < target_length:
                    audio = np.pad(audio, (0, target_length - len(audio)), mode='constant')
                else:
                    audio = audio[:target_length]
            
            # Create mel spectrogram
            mel_spec = librosa.feature.melspectrogram(
                y=audio,
                sr=sr,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                n_mels=self.n_mels
            )
            
            # Force consistent shape
            if mel_spec.shape[1] < expected_frames:
                mel_spec = np.pad(mel_spec, ((0, 0), (0, expected_frames - mel_spec.shape[1])), mode='constant')
            else:
                mel_spec = mel_spec[:, :expected_frames]
            
            # Convert to log scale
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            
            # Normalize
            mel_spec_db = (mel_spec_db - mel_spec_db.mean()) / (mel_spec_db.std() + 1e-8)
            
            # SpecAugment: random frequency/time masking
            if self.augment:
                # Frequency masking
                if np.random.random() < 0.5:
                    f_mask = np.random.randint(0, self.n_mels // 8)
                    f_start = np.random.randint(0, self.n_mels - f_mask)
                    mel_spec_db[f_start:f_start + f_mask, :] = 0
                # Time masking
                if np.random.random() < 0.5:
                    t_mask = np.random.randint(0, expected_frames // 8)
                    t_start = np.random.randint(0, expected_frames - t_mask)
                    mel_spec_db[:, t_start:t_start + t_mask] = 0
            
            # Add channel dimension
            mel_spec_db = torch.FloatTensor(mel_spec_db).unsqueeze(0)
            
        except Exception:
            mel_spec_db = torch.zeros(1, self.n_mels, expected_frames)
        
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
    output_dir = Path(config['data']['output_dir']) / f"cnn_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Load datasets
    print("\n=== Loading Datasets ===")
    train_dataset = AudioDataset(
        processed_dir / 'train_processed.csv',
        config,
        augment=True   # augmentation ON for training
    )
    val_dataset = AudioDataset(
        processed_dir / 'val_processed.csv',
        config,
        augment=False  # NO augmentation for validation
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=0  # Use 0 for Windows compatibility
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=0
    )
    
    # Create model
    print("\n=== Creating Model ===")
    model = SimpleCNN(num_classes=2)
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Calculate class weights for imbalanced dataset
    train_labels = pd.read_csv(processed_dir / 'train_processed.csv')['label'].values
    class_counts = np.bincount(train_labels)
    class_weights = 1.0 / class_counts
    class_weights = class_weights / class_weights.sum() * len(class_counts)
    class_weights = torch.FloatTensor(class_weights).to(device)
    
    print(f"Class distribution: {class_counts}")
    print(f"Class weights: {class_weights.cpu().numpy()}")
    
    # Loss and optimizer with class weights
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
        weight_decay=1e-3   # L2 regularization to reduce overfitting
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config['training']['num_epochs'], eta_min=1e-5
    )
    
    # Training loop
    print("\n=== Starting Training ===")
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
        
        scheduler.step()
        
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
    parser = argparse.ArgumentParser(description='Train simple CNN for chainsaw detection')
    parser.add_argument('--config', type=str, default='config.yaml', help='Config file')
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    print("=" * 60)
    print("CHAINSAW DETECTION - SIMPLE CNN TRAINING")
    print("=" * 60)
    
    # Check if processed data exists
    processed_dir = Path(config['data']['processed_data_dir'])
    if not (processed_dir / 'train_processed.csv').exists():
        print("\n⚠ Processed data not found!")
        print("Please run: python preprocess.py")
        return
    
    # Train model
    model, output_dir = train_model(config)
    
    print("\n" + "=" * 60)
    print("NEXT STEPS:")
    print("=" * 60)
    print(f"1. Evaluate: python evaluate_simple.py --model-path {output_dir}/best_model.pt")
    print(f"2. Run inference: python inference_simple.py --model-path {output_dir}/best_model.pt --audio <file>")
    print("=" * 60)


if __name__ == "__main__":
    main()
