"""
Evaluation script for trained chainsaw detection model
"""

import os
import yaml
import torch
import numpy as np
import pandas as pd
from pathlib import Path
import argparse
from tqdm import tqdm

from transformers import ASTFeatureExtractor, ASTForAudioClassification
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
import librosa

from train import ChainsawDataset


def evaluate_model(model, dataloader, device):
    """Evaluate model on dataset"""
    model.eval()
    all_predictions = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            inputs = batch['input_values'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
            predictions = torch.argmax(logits, dim=-1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())  # Probability of chainsaw class
    
    return np.array(all_predictions), np.array(all_labels), np.array(all_probs)


def plot_confusion_matrix(y_true, y_pred, output_path):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Non-Chainsaw', 'Chainsaw'],
                yticklabels=['Non-Chainsaw', 'Chainsaw'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"✓ Confusion matrix saved to: {output_path}")


def plot_roc_curve(y_true, y_probs, output_path):
    """Plot ROC curve"""
    fpr, tpr, thresholds = roc_curve(y_true, y_probs)
    auc = roc_auc_score(y_true, y_probs)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"✓ ROC curve saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate chainsaw detection model')
    parser.add_argument('--config', type=str, default='config.yaml', help='Config file')
    parser.add_argument('--model-path', type=str, required=True, help='Path to trained model')
    parser.add_argument('--split', type=str, default='test', choices=['train', 'val', 'test'],
                        help='Which split to evaluate on')
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    print("=" * 60)
    print("MODEL EVALUATION")
    print("=" * 60)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    print(f"\nLoading model from: {args.model_path}")
    model = ASTForAudioClassification.from_pretrained(args.model_path)
    model.to(device)
    
    # Load feature extractor
    feature_extractor = ASTFeatureExtractor.from_pretrained(args.model_path)
    
    # Load dataset
    processed_dir = Path(config['data']['processed_data_dir'])
    metadata_file = processed_dir / f'{args.split}_processed.csv'
    
    if not metadata_file.exists():
        print(f"⚠ Metadata file not found: {metadata_file}")
        return
    
    dataset = ChainsawDataset(metadata_file, feature_extractor, config)
    dataloader = DataLoader(
        dataset,
        batch_size=config['training']['batch_size'],
        num_workers=config['hardware']['num_workers'],
        shuffle=False
    )
    
    print(f"Evaluating on {args.split} set: {len(dataset)} samples")
    
    # Evaluate
    predictions, labels, probs = evaluate_model(model, dataloader, device)
    
    # Calculate metrics
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='binary'
    )
    auc = roc_auc_score(labels, probs)
    
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"AUC-ROC:   {auc:.4f}")
    
    # Classification report
    print("\n" + "=" * 60)
    print("CLASSIFICATION REPORT")
    print("=" * 60)
    print(classification_report(
        labels, predictions,
        target_names=['Non-Chainsaw', 'Chainsaw']
    ))
    
    # Save results
    output_dir = Path(args.model_path) / 'evaluation'
    output_dir.mkdir(exist_ok=True)
    
    # Save metrics
    results = {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'auc': float(auc),
        'split': args.split,
        'num_samples': len(dataset)
    }
    
    import json
    with open(output_dir / f'{args.split}_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Plot confusion matrix
    plot_confusion_matrix(labels, predictions, output_dir / f'{args.split}_confusion_matrix.png')
    
    # Plot ROC curve
    plot_roc_curve(labels, probs, output_dir / f'{args.split}_roc_curve.png')
    
    # Save predictions
    results_df = pd.DataFrame({
        'true_label': labels,
        'predicted_label': predictions,
        'chainsaw_probability': probs
    })
    results_df.to_csv(output_dir / f'{args.split}_predictions.csv', index=False)
    
    print(f"\n✓ Evaluation results saved to: {output_dir}")


if __name__ == "__main__":
    main()
