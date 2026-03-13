"""
Evaluation script for TFCNN model
"""

import os
import yaml
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
import argparse
from tqdm import tqdm

from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns

from train_tfcnn import TFCNN, AudioDataset


def evaluate_model(model, dataloader, device):
    """Evaluate model on dataset"""
    model.eval()
    all_predictions = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Evaluating"):
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=-1)
            predictions = torch.argmax(outputs, dim=-1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())
    
    return np.array(all_predictions), np.array(all_labels), np.array(all_probs)


def plot_confusion_matrix(y_true, y_pred, output_path):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Non-Chainsaw', 'Chainsaw'],
                yticklabels=['Non-Chainsaw', 'Chainsaw'])
    plt.title('Confusion Matrix - TFCNN Model')
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
    plt.plot(fpr, tpr, label=f'TFCNN (AUC = {auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - TFCNN Model')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"✓ ROC curve saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate TFCNN model')
    parser.add_argument('--config', type=str, default='config.yaml', help='Config file')
    parser.add_argument('--model-path', type=str, required=True, help='Path to trained model')
    parser.add_argument('--split', type=str, default='test', choices=['train', 'val', 'test'])
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    print("=" * 70)
    print("MODEL EVALUATION - TFCNN WITH ATTENTION")
    print("=" * 70)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    print(f"\nLoading model from: {args.model_path}")
    checkpoint = torch.load(args.model_path, map_location=device)
    
    model = TFCNN(num_classes=2)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"✓ Model loaded (trained for {checkpoint['epoch'] + 1} epochs)")
    print(f"✓ Best validation F1: {checkpoint['val_f1']:.4f}")
    
    # Load dataset
    processed_dir = Path(config['data']['processed_data_dir'])
    metadata_file = processed_dir / f'{args.split}_processed.csv'
    
    dataset = AudioDataset(metadata_file, config)
    dataloader = DataLoader(
        dataset,
        batch_size=config['training']['batch_size'],
        num_workers=0,
        shuffle=False
    )
    
    print(f"\nEvaluating on {args.split} set: {len(dataset)} samples")
    
    # Evaluate
    predictions, labels, probs = evaluate_model(model, dataloader, device)
    
    # Calculate metrics
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='binary', zero_division=0
    )
    
    precision_per_class, recall_per_class, f1_per_class, _ = precision_recall_fscore_support(
        labels, predictions, average=None, zero_division=0
    )
    
    try:
        auc = roc_auc_score(labels, probs)
    except:
        auc = 0.0
    
    print("\n" + "=" * 70)
    print("OVERALL RESULTS")
    print("=" * 70)
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"AUC-ROC:   {auc:.4f}")
    
    print("\n" + "=" * 70)
    print("PER-CLASS METRICS")
    print("=" * 70)
    print(f"Non-Chainsaw - Precision: {precision_per_class[0]:.4f}, Recall: {recall_per_class[0]:.4f}, F1: {f1_per_class[0]:.4f}")
    print(f"Chainsaw     - Precision: {precision_per_class[1]:.4f}, Recall: {recall_per_class[1]:.4f}, F1: {f1_per_class[1]:.4f}")
    
    print("\n" + "=" * 70)
    print("CLASSIFICATION REPORT")
    print("=" * 70)
    print(classification_report(
        labels, predictions,
        target_names=['Non-Chainsaw', 'Chainsaw'],
        zero_division=0
    ))
    
    # Save results
    model_dir = Path(args.model_path).parent
    output_dir = model_dir / 'evaluation'
    output_dir.mkdir(exist_ok=True)
    
    results = {
        'model_type': 'TFCNN',
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
    
    plot_confusion_matrix(labels, predictions, output_dir / f'{args.split}_confusion_matrix.png')
    
    if auc > 0:
        plot_roc_curve(labels, probs, output_dir / f'{args.split}_roc_curve.png')
    
    results_df = pd.DataFrame({
        'true_label': labels,
        'predicted_label': predictions,
        'chainsaw_probability': probs
    })
    results_df.to_csv(output_dir / f'{args.split}_predictions.csv', index=False)
    
    print(f"\n✓ Evaluation results saved to: {output_dir}")


if __name__ == "__main__":
    main()
