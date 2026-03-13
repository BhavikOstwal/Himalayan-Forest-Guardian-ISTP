"""
Training script for chainsaw detection using AST
"""

import os
import yaml
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse
from datetime import datetime

from torch.utils.data import Dataset, DataLoader
from transformers import (
    ASTFeatureExtractor,
    ASTForAudioClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
import librosa
import pandas as pd


class ChainsawDataset(Dataset):
    """Custom dataset for chainsaw detection"""
    
    def __init__(self, metadata_file, feature_extractor, config):
        self.df = pd.read_csv(metadata_file)
        self.feature_extractor = feature_extractor
        self.sample_rate = config['audio']['sample_rate']
        self.duration = config['audio']['duration']
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Load audio
        audio, sr = librosa.load(
            row['file_path'], 
            sr=self.sample_rate,
            duration=self.duration
        )
        
        # Extract features
        inputs = self.feature_extractor(
            audio, 
            sampling_rate=self.sample_rate,
            return_tensors="pt"
        )
        
        return {
            'input_values': inputs['input_values'].squeeze(0),
            'labels': torch.tensor(row['label'], dtype=torch.long)
        }


def compute_metrics(eval_pred):
    """Compute evaluation metrics"""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    # Calculate metrics
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='binary'
    )
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def create_model(config):
    """Initialize AST model"""
    print("\n=== Loading AST Model ===")
    
    model = ASTForAudioClassification.from_pretrained(
        config['model']['name'],
        num_labels=config['model']['num_labels'],
        ignore_mismatched_sizes=True
    )
    
    # Optionally freeze feature extractor
    if config['model']['freeze_feature_extractor']:
        for param in model.audio_spectrogram_transformer.parameters():
            param.requires_grad = False
        print("✓ Feature extractor frozen")
    
    print(f"✓ Model loaded: {config['model']['name']}")
    return model


def train_model(config):
    """Main training function"""
    
    # Setup paths
    processed_dir = Path(config['data']['processed_data_dir'])
    output_dir = Path(config['data']['output_dir']) / f"model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize feature extractor
    feature_extractor = ASTFeatureExtractor.from_pretrained(config['model']['name'])
    
    # Load datasets
    print("\n=== Loading Datasets ===")
    train_dataset = ChainsawDataset(
        processed_dir / 'train_processed.csv',
        feature_extractor,
        config
    )
    val_dataset = ChainsawDataset(
        processed_dir / 'val_processed.csv',
        feature_extractor,
        config
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    # Create model
    model = create_model(config)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=config['training']['num_epochs'],
        per_device_train_batch_size=config['training']['batch_size'],
        per_device_eval_batch_size=config['training']['batch_size'],
        learning_rate=float(config['training']['learning_rate']),
        warmup_steps=config['training']['warmup_steps'],
        weight_decay=float(config['training']['weight_decay']),
        gradient_accumulation_steps=config['training']['gradient_accumulation_steps'],
        eval_strategy="steps",
        eval_steps=config['training']['eval_steps'],
        save_steps=config['training']['save_steps'],
        logging_steps=config['training']['logging_steps'],
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        save_total_limit=3,
        fp16=config['hardware']['mixed_precision'] and torch.cuda.is_available(),
        dataloader_num_workers=config['hardware']['num_workers'],
        remove_unused_columns=False,
        push_to_hub=False,
        report_to="tensorboard"
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(
            early_stopping_patience=config['training']['early_stopping_patience']
        )]
    )
    
    # Train
    print("\n=== Starting Training ===")
    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    
    trainer.train()
    
    # Save final model
    final_model_path = output_dir / "final_model"
    trainer.save_model(final_model_path)
    feature_extractor.save_pretrained(final_model_path)
    
    print(f"\n✓ Training complete!")
    print(f"Model saved to: {final_model_path}")
    
    # Evaluate on validation set
    print("\n=== Final Validation Results ===")
    eval_results = trainer.evaluate()
    print(eval_results)
    
    # Save results
    import json
    with open(output_dir / 'training_results.json', 'w') as f:
        json.dump(eval_results, f, indent=2)
    
    return trainer, model


def main():
    parser = argparse.ArgumentParser(description='Train chainsaw detection model')
    parser.add_argument('--config', type=str, default='config.yaml', help='Config file')
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    print("=" * 60)
    print("CHAINSAW DETECTION - MODEL TRAINING")
    print("=" * 60)
    
    # Check if processed data exists
    processed_dir = Path(config['data']['processed_data_dir'])
    if not (processed_dir / 'train_processed.csv').exists():
        print("\n⚠ Processed data not found!")
        print("Please run: python preprocess.py")
        return
    
    # Train model
    trainer, model = train_model(config)
    
    print("\n" + "=" * 60)
    print("NEXT STEPS:")
    print("1. Check tensorboard logs: tensorboard --logdir output/")
    print("2. Evaluate on test set: python evaluate.py")
    print("3. Run inference: python inference.py --audio <path_to_audio>")
    print("=" * 60)


if __name__ == "__main__":
    main()
