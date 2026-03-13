"""
Lightweight ML Models for Chainsaw Detection
Trains basic ML classifiers (SVM, Random Forest, XGBoost) for audio classification
"""

import os
import yaml
import numpy as np
import pandas as pd
import librosa
import pickle
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
import json
import warnings
warnings.filterwarnings('ignore')

# ML Libraries
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, confusion_matrix, classification_report,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    print("XGBoost not available. Install with: pip install xgboost")
    XGBOOST_AVAILABLE = False


class LightweightFeatureExtractor:
    """Extract compact but informative features from audio"""
    
    def __init__(self, config):
        self.config = config
        self.sample_rate = config['audio']['sample_rate']
        self.duration = config['audio']['duration']
        self.n_mels = 40  # Reduced from 128 for efficiency
        self.n_mfcc = 13  # Standard number
        self.hop_length = 512
        self.n_fft = 2048
    
    def extract_features(self, file_path):
        """Extract compact feature set from audio file"""
        try:
            # Load audio
            audio, sr = librosa.load(file_path, sr=self.sample_rate, 
                                    duration=self.duration)
            
            features = []
            
            # 1. MFCCs (most informative for speech/audio classification)
            mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=self.n_mfcc)
            features.extend(np.mean(mfccs, axis=1))  # 13 features
            features.extend(np.std(mfccs, axis=1))   # 13 features
            features.extend(np.max(mfccs, axis=1))   # 13 features
            features.extend(np.min(mfccs, axis=1))   # 13 features
            
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
            features.extend(np.mean(spectral_contrast, axis=1))  # 7 features
            
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
            
            # 5. Chroma features (12 pitch classes)
            chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
            features.extend(np.mean(chroma, axis=1))  # 12 features
            
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
            
            # 8. Additional time-domain features
            features.append(np.mean(np.abs(audio)))  # Mean absolute value
            features.append(np.std(audio))  # Standard deviation
            features.append(np.max(np.abs(audio)))  # Peak amplitude
            
            # Total: ~95 features (compact and efficient)
            return np.array(features)
            
        except Exception as e:
            print(f"Error extracting features from {file_path}: {e}")
            return None


class MLModelTrainer:
    """Train and evaluate lightweight ML models"""
    
    def __init__(self, config_path='config.yaml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.feature_extractor = LightweightFeatureExtractor(self.config)
        self.scaler = StandardScaler()
        
        # Create output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = Path(f"output/ml_model_{timestamp}")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Output directory: {self.output_dir}")
    
    def load_or_extract_features(self, csv_path, cache_path, split_name):
        """Load features from cache or extract them"""
        
        if os.path.exists(cache_path):
            print(f"Loading cached {split_name} features from {cache_path}")
            data = np.load(cache_path)
            return data['features'], data['labels']
        
        print(f"Extracting {split_name} features...")
        df = pd.read_csv(csv_path)
        
        features_list = []
        labels_list = []
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing {split_name}"):
            file_path = row['file_path']
            
            # Handle path - try both relative and absolute
            if not os.path.exists(file_path):
                file_path = os.path.join('c:\\Users\\dhira\\Desktop\\ISTP', file_path)
            
            if os.path.exists(file_path):
                features = self.feature_extractor.extract_features(file_path)
                if features is not None:
                    features_list.append(features)
                    labels_list.append(row['label'])
            else:
                print(f"File not found: {file_path}")
        
        features = np.array(features_list)
        labels = np.array(labels_list)
        
        # Cache for future use
        np.savez(cache_path, features=features, labels=labels)
        print(f"Cached features to {cache_path}")
        
        return features, labels
    
    def prepare_data(self):
        """Load and prepare train, validation, and test data"""
        print("\n" + "="*70)
        print("PREPARING DATA")
        print("="*70)
        
        data_dir = Path(self.config['data']['processed_data_dir'])
        cache_dir = Path('data/features_cache')
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Load or extract features for each split
        self.X_train, self.y_train = self.load_or_extract_features(
            data_dir / 'train_processed.csv',
            cache_dir / 'train_features.npz',
            'train'
        )
        
        self.X_val, self.y_val = self.load_or_extract_features(
            data_dir / 'val_processed.csv',
            cache_dir / 'val_features.npz',
            'validation'
        )
        
        self.X_test, self.y_test = self.load_or_extract_features(
            data_dir / 'test_processed.csv',
            cache_dir / 'test_features.npz',
            'test'
        )
        
        print(f"\nData shapes:")
        print(f"  Train: {self.X_train.shape}, Features: {self.X_train.shape[1]}")
        print(f"  Val:   {self.X_val.shape}")
        print(f"  Test:  {self.X_test.shape}")
        
        # Normalize features
        print("\nNormalizing features...")
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_val = self.scaler.transform(self.X_val)
        self.X_test = self.scaler.transform(self.X_test)
        
        # Save scaler
        with open(self.output_dir / 'scaler.pkl', 'wb') as f:
            pickle.dump(self.scaler, f)
        
        print(f"Class distribution in training set:")
        unique, counts = np.unique(self.y_train, return_counts=True)
        for label, count in zip(unique, counts):
            print(f"  Class {label}: {count} ({count/len(self.y_train)*100:.1f}%)")
    
    def train_models(self):
        """Train multiple lightweight ML models"""
        print("\n" + "="*70)
        print("TRAINING MODELS")
        print("="*70)
        
        self.models = {}
        self.results = {}
        
        # 1. Support Vector Machine (RBF kernel)
        print("\n1. Training SVM (RBF kernel)...")
        svm_model = SVC(kernel='rbf', C=10, gamma='scale', 
                       probability=True, random_state=42)
        svm_model.fit(self.X_train, self.y_train)
        self.models['SVM'] = svm_model
        print("   ✓ SVM trained")
        
        # 2. Random Forest
        print("\n2. Training Random Forest...")
        rf_model = RandomForestClassifier(n_estimators=200, max_depth=20,
                                         min_samples_split=5, min_samples_leaf=2,
                                         random_state=42, n_jobs=-1)
        rf_model.fit(self.X_train, self.y_train)
        self.models['Random Forest'] = rf_model
        print("   ✓ Random Forest trained")
        
        # 3. Gradient Boosting
        print("\n3. Training Gradient Boosting...")
        gb_model = GradientBoostingClassifier(n_estimators=100, max_depth=5,
                                             learning_rate=0.1, random_state=42)
        gb_model.fit(self.X_train, self.y_train)
        self.models['Gradient Boosting'] = gb_model
        print("   ✓ Gradient Boosting trained")
        
        # 4. XGBoost (if available)
        if XGBOOST_AVAILABLE:
            print("\n4. Training XGBoost...")
            xgb_model = xgb.XGBClassifier(n_estimators=100, max_depth=5,
                                         learning_rate=0.1, random_state=42,
                                         eval_metric='logloss')
            xgb_model.fit(self.X_train, self.y_train)
            self.models['XGBoost'] = xgb_model
            print("   ✓ XGBoost trained")
        
        # 5. Logistic Regression (baseline)
        print("\n5. Training Logistic Regression...")
        lr_model = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
        lr_model.fit(self.X_train, self.y_train)
        self.models['Logistic Regression'] = lr_model
        print("   ✓ Logistic Regression trained")
        
        print(f"\n✓ Trained {len(self.models)} models successfully")
    
    def evaluate_models(self):
        """Evaluate all trained models"""
        print("\n" + "="*70)
        print("EVALUATING MODELS")
        print("="*70)
        
        for model_name, model in self.models.items():
            print(f"\n{model_name}:")
            print("-" * 50)
            
            # Predictions
            y_train_pred = model.predict(self.X_train)
            y_val_pred = model.predict(self.X_val)
            y_test_pred = model.predict(self.X_test)
            
            # Probabilities (if available)
            if hasattr(model, 'predict_proba'):
                y_test_proba = model.predict_proba(self.X_test)[:, 1]
            else:
                y_test_proba = y_test_pred
            
            # Metrics
            train_acc = accuracy_score(self.y_train, y_train_pred)
            val_acc = accuracy_score(self.y_val, y_val_pred)
            test_acc = accuracy_score(self.y_test, y_test_pred)
            
            test_precision = precision_score(self.y_test, y_test_pred)
            test_recall = recall_score(self.y_test, y_test_pred)
            test_f1 = f1_score(self.y_test, y_test_pred)
            
            try:
                test_auc = roc_auc_score(self.y_test, y_test_proba)
            except:
                test_auc = 0.0
            
            # Store results
            self.results[model_name] = {
                'train_accuracy': train_acc,
                'val_accuracy': val_acc,
                'test_accuracy': test_acc,
                'test_precision': test_precision,
                'test_recall': test_recall,
                'test_f1': test_f1,
                'test_auc': test_auc,
                'y_test_pred': y_test_pred,
                'y_test_proba': y_test_proba,
                'confusion_matrix': confusion_matrix(self.y_test, y_test_pred)
            }
            
            # Print results
            print(f"  Train Accuracy:  {train_acc:.4f}")
            print(f"  Val Accuracy:    {val_acc:.4f}")
            print(f"  Test Accuracy:   {test_acc:.4f}")
            print(f"  Test Precision:  {test_precision:.4f}")
            print(f"  Test Recall:     {test_recall:.4f}")
            print(f"  Test F1:         {test_f1:.4f}")
            if test_auc > 0:
                print(f"  Test AUC:        {test_auc:.4f}")
            
            print(f"\n  Confusion Matrix:")
            cm = self.results[model_name]['confusion_matrix']
            print(f"    TN: {cm[0,0]:<4} FP: {cm[0,1]:<4}")
            print(f"    FN: {cm[1,0]:<4} TP: {cm[1,1]:<4}")
        
        # Find best model
        best_model_name = max(self.results, key=lambda x: self.results[x]['test_f1'])
        self.best_model_name = best_model_name
        self.best_model = self.models[best_model_name]
        
        print("\n" + "="*70)
        print(f"BEST MODEL: {best_model_name}")
        print(f"Test F1 Score: {self.results[best_model_name]['test_f1']:.4f}")
        print(f"Test Accuracy: {self.results[best_model_name]['test_accuracy']:.4f}")
        print("="*70)
    
    def plot_results(self):
        """Create visualization of results"""
        print("\nCreating result visualizations...")
        
        # Prepare data for plotting
        model_names = list(self.results.keys())
        metrics = ['train_accuracy', 'val_accuracy', 'test_accuracy', 
                  'test_precision', 'test_recall', 'test_f1']
        
        # 1. Model comparison plot
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Accuracy comparison
        ax = axes[0, 0]
        x = np.arange(len(model_names))
        width = 0.25
        
        train_accs = [self.results[m]['train_accuracy'] for m in model_names]
        val_accs = [self.results[m]['val_accuracy'] for m in model_names]
        test_accs = [self.results[m]['test_accuracy'] for m in model_names]
        
        ax.bar(x - width, train_accs, width, label='Train', alpha=0.8)
        ax.bar(x, val_accs, width, label='Validation', alpha=0.8)
        ax.bar(x + width, test_accs, width, label='Test', alpha=0.8)
        
        ax.set_xlabel('Model', fontsize=12)
        ax.set_ylabel('Accuracy', fontsize=12)
        ax.set_title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim([0.5, 1.0])
        
        # Test metrics comparison
        ax = axes[0, 1]
        test_metrics = ['test_accuracy', 'test_precision', 'test_recall', 'test_f1']
        metric_labels = ['Accuracy', 'Precision', 'Recall', 'F1']
        
        x = np.arange(len(test_metrics))
        width = 0.15
        
        for i, model_name in enumerate(model_names):
            values = [self.results[model_name][m] for m in test_metrics]
            ax.bar(x + i*width, values, width, label=model_name, alpha=0.8)
        
        ax.set_xlabel('Metric', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title('Test Metrics Comparison', fontsize=14, fontweight='bold')
        ax.set_xticks(x + width * (len(model_names)-1) / 2)
        ax.set_xticklabels(metric_labels)
        ax.legend(fontsize=9)
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim([0.5, 1.0])
        
        # Confusion matrix for best model
        ax = axes[1, 0]
        cm = self.results[self.best_model_name]['confusion_matrix']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                   xticklabels=['Non-Chainsaw', 'Chainsaw'],
                   yticklabels=['Non-Chainsaw', 'Chainsaw'])
        ax.set_title(f'Confusion Matrix - {self.best_model_name}', 
                    fontsize=14, fontweight='bold')
        ax.set_ylabel('True Label', fontsize=12)
        ax.set_xlabel('Predicted Label', fontsize=12)
        
        # ROC curves (if available)
        ax = axes[1, 1]
        for model_name in model_names:
            if self.results[model_name]['test_auc'] > 0:
                y_proba = self.results[model_name]['y_test_proba']
                fpr, tpr, _ = roc_curve(self.y_test, y_proba)
                auc = self.results[model_name]['test_auc']
                ax.plot(fpr, tpr, label=f'{model_name} (AUC={auc:.3f})', 
                       linewidth=2)
        
        ax.plot([0, 1], [0, 1], 'k--', label='Random', linewidth=1)
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title('ROC Curves', fontsize=14, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'model_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {self.output_dir / 'model_comparison.png'}")
        
        # 2. Feature importance (for tree-based models)
        if 'Random Forest' in self.models:
            fig, axes = plt.subplots(1, 2, figsize=(16, 6))
            
            # Random Forest feature importance
            rf_model = self.models['Random Forest']
            feature_importance = rf_model.feature_importances_
            top_n = 20
            top_indices = np.argsort(feature_importance)[-top_n:]
            
            ax = axes[0]
            ax.barh(range(top_n), feature_importance[top_indices], alpha=0.8)
            ax.set_xlabel('Importance', fontsize=12)
            ax.set_ylabel('Feature Index', fontsize=12)
            ax.set_title(f'Top {top_n} Features - Random Forest', 
                        fontsize=14, fontweight='bold')
            ax.set_yticks(range(top_n))
            ax.set_yticklabels(top_indices)
            ax.grid(axis='x', alpha=0.3)
            
            # Gradient Boosting feature importance
            if 'Gradient Boosting' in self.models:
                gb_model = self.models['Gradient Boosting']
                feature_importance = gb_model.feature_importances_
                top_indices = np.argsort(feature_importance)[-top_n:]
                
                ax = axes[1]
                ax.barh(range(top_n), feature_importance[top_indices], alpha=0.8)
                ax.set_xlabel('Importance', fontsize=12)
                ax.set_ylabel('Feature Index', fontsize=12)
                ax.set_title(f'Top {top_n} Features - Gradient Boosting', 
                            fontsize=14, fontweight='bold')
                ax.set_yticks(range(top_n))
                ax.set_yticklabels(top_indices)
                ax.grid(axis='x', alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(self.output_dir / 'feature_importance.png', 
                       dpi=300, bbox_inches='tight')
            plt.close()
            print(f"  Saved: {self.output_dir / 'feature_importance.png'}")
    
    def save_models_and_results(self):
        """Save trained models and results"""
        print("\nSaving models and results...")
        
        # Save best model
        with open(self.output_dir / 'best_model.pkl', 'wb') as f:
            pickle.dump(self.best_model, f)
        print(f"  Saved best model: {self.best_model_name}")
        
        # Save all models
        models_dir = self.output_dir / 'all_models'
        models_dir.mkdir(exist_ok=True)
        
        for model_name, model in self.models.items():
            safe_name = model_name.replace(' ', '_').lower()
            with open(models_dir / f'{safe_name}.pkl', 'wb') as f:
                pickle.dump(model, f)
        
        # Save results as JSON
        results_json = {}
        for model_name, result in self.results.items():
            results_json[model_name] = {
                k: float(v) if isinstance(v, (np.floating, float)) else 
                   v.tolist() if isinstance(v, np.ndarray) else v
                for k, v in result.items()
                if k not in ['y_test_pred', 'y_test_proba']
            }
        
        with open(self.output_dir / 'results.json', 'w') as f:
            json.dump(results_json, f, indent=2)
        
        # Save detailed classification report for best model
        y_pred = self.results[self.best_model_name]['y_test_pred']
        report = classification_report(self.y_test, y_pred, 
                                      target_names=['Non-Chainsaw', 'Chainsaw'])
        
        with open(self.output_dir / 'classification_report.txt', 'w') as f:
            f.write(f"Best Model: {self.best_model_name}\n")
            f.write("="*70 + "\n\n")
            f.write(report)
            f.write("\n\nDetailed Results:\n")
            f.write("-"*70 + "\n")
            for metric, value in self.results[self.best_model_name].items():
                if metric not in ['y_test_pred', 'y_test_proba', 'confusion_matrix']:
                    f.write(f"{metric}: {value}\n")
        
        print(f"  Saved results to: {self.output_dir}")
        
        # Save a summary README
        with open(self.output_dir / 'README.txt', 'w') as f:
            f.write("Chainsaw Detection - ML Models\n")
            f.write("="*70 + "\n\n")
            f.write(f"Training Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Number of Features: {self.X_train.shape[1]}\n")
            f.write(f"Training Samples: {len(self.y_train)}\n")
            f.write(f"Validation Samples: {len(self.y_val)}\n")
            f.write(f"Test Samples: {len(self.y_test)}\n\n")
            f.write(f"Best Model: {self.best_model_name}\n")
            f.write(f"Test Accuracy: {self.results[self.best_model_name]['test_accuracy']:.4f}\n")
            f.write(f"Test F1 Score: {self.results[self.best_model_name]['test_f1']:.4f}\n\n")
            f.write("Files:\n")
            f.write("  - best_model.pkl: Best performing model\n")
            f.write("  - scaler.pkl: Feature scaler (StandardScaler)\n")
            f.write("  - results.json: All model results\n")
            f.write("  - classification_report.txt: Detailed metrics\n")
            f.write("  - model_comparison.png: Visual comparison\n")
            f.write("  - all_models/: All trained models\n")
    
    def run_complete_pipeline(self):
        """Run the complete training pipeline"""
        print("\n" + "="*70)
        print("LIGHTWEIGHT ML MODELS FOR CHAINSAW DETECTION")
        print("="*70)
        
        # Prepare data
        self.prepare_data()
        
        # Train models
        self.train_models()
        
        # Evaluate models
        self.evaluate_models()
        
        # Plot results
        self.plot_results()
        
        # Save everything
        self.save_models_and_results()
        
        print("\n" + "="*70)
        print("TRAINING COMPLETE!")
        print("="*70)
        print(f"\nOutput directory: {self.output_dir.absolute()}")
        print(f"Best model: {self.best_model_name}")
        print(f"Test Accuracy: {self.results[self.best_model_name]['test_accuracy']:.4f}")
        print(f"Test F1 Score: {self.results[self.best_model_name]['test_f1']:.4f}")


if __name__ == "__main__":
    trainer = MLModelTrainer()
    trainer.run_complete_pipeline()
