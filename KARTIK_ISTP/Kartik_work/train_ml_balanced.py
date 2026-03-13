"""
Improved Lightweight ML Models with Class Imbalance Handling
Trains balanced classifiers using class weights and SMOTE
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
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                              VotingClassifier, ExtraTreesClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, confusion_matrix, classification_report,
                             roc_auc_score, roc_curve, balanced_accuracy_score)
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN
from imblearn.ensemble import BalancedRandomForestClassifier, EasyEnsembleClassifier
import matplotlib.pyplot as plt
import seaborn as sns

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


class ImprovedMLTrainer:
    """Train balanced ML models with class imbalance handling"""
    
    def __init__(self, config_path='config.yaml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.scaler = StandardScaler()
        
        # Create output directory with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = Path('output') / f'ml_balanced_{timestamp}'
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.models = {}
        self.results = {}
        
        print(f"Output directory: {self.output_dir}")
    
    def extract_features(self, audio_path):
        """Extract compact 91-feature set from audio"""
        # Load audio
        y, sr = librosa.load(audio_path, sr=self.config['audio']['sample_rate'], duration=10)
        
        # Check if audio is valid
        if len(y) == 0:
            return None
        
        features = []
        
        # 1. MFCCs (52 features: 13 coefs × 4 stats)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        features.extend([
            np.mean(mfccs, axis=1), np.std(mfccs, axis=1),
            np.max(mfccs, axis=1), np.min(mfccs, axis=1)
        ])
        
        # 2. Spectral features (13 features)
        features.extend([
            np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)),
            np.std(librosa.feature.spectral_centroid(y=y, sr=sr)),
            np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr)),
            np.std(librosa.feature.spectral_bandwidth(y=y, sr=sr)),
            np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)),
            np.std(librosa.feature.spectral_rolloff(y=y, sr=sr)),
            np.mean(librosa.feature.spectral_flatness(y=y)),
            np.std(librosa.feature.spectral_flatness(y=y)),
            np.mean(librosa.feature.zero_crossing_rate(y)),
            np.std(librosa.feature.zero_crossing_rate(y)),
            np.mean(librosa.feature.rms(y=y)),
            np.std(librosa.feature.rms(y=y)),
            float(librosa.feature.spectral_contrast(y=y, sr=sr).mean())
        ])
        
        # 3. Chroma features (12 features: 12 pitch classes)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        features.extend(np.mean(chroma, axis=1))
        
        # 4. Energy & Temporal (14 features)
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        features.extend([
            np.mean(onset_env), np.std(onset_env),
            np.max(onset_env), np.min(onset_env),
            float(librosa.beat.tempo(onset_envelope=onset_env, sr=sr)[0]),
            np.mean(np.abs(y)), np.std(np.abs(y)),
            np.max(np.abs(y)), np.min(np.abs(y)),
            float(np.percentile(np.abs(y), 25)),
            float(np.percentile(np.abs(y), 50)),
            float(np.percentile(np.abs(y), 75)),
            float(np.sum(y**2)),
            len(librosa.onset.onset_detect(y=y, sr=sr))
        ])
        
        result = np.concatenate([np.atleast_1d(f).flatten() for f in features])
        return result if len(result) > 0 else None
    
    def load_cached_features(self):
        """Load pre-computed features from cache or extract if not found"""
        cache_dir = Path('data/features_cache')
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        train_cache = cache_dir / 'train_features.npz'
        val_cache = cache_dir / 'val_features.npz'
        test_cache = cache_dir / 'test_features.npz'
        
        # Check if all caches exist
        if not (train_cache.exists() and val_cache.exists() and test_cache.exists()):
            print("\n" + "="*70)
            print("CACHE NOT FOUND - EXTRACTING FEATURES")
            print("="*70)
            print("This will take a few minutes...")
            self.extract_all_features()
            return
        
        print("\n" + "="*70)
        print("LOADING CACHED FEATURES")
        print("="*70)
        
        # Load train features
        data = np.load(train_cache)
        self.X_train = data['features']
        self.y_train = data['labels']
        
        # Load val features
        data = np.load(val_cache)
        self.X_val = data['features']
        self.y_val = data['labels']
        
        # Load test features
        data = np.load(test_cache)
        self.X_test = data['features']
        self.y_test = data['labels']
        
        print(f"\nLoaded features:")
        print(f"  Train: {self.X_train.shape}")
        print(f"  Val:   {self.X_val.shape}")
        print(f"  Test:  {self.X_test.shape}")
        
        print(f"\nOriginal class distribution:")
        unique, counts = np.unique(self.y_train, return_counts=True)
        for label, count in zip(unique, counts):
            print(f"  Class {label}: {count} ({count/len(self.y_train)*100:.1f}%)")
        
        # Normalize features
        print("\nNormalizing features...")
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_val = self.scaler.transform(self.X_val)
        self.X_test = self.scaler.transform(self.X_test)
        
        # Save scaler
        with open(self.output_dir / 'scaler.pkl', 'wb') as f:
            pickle.dump(self.scaler, f)
    
    def extract_all_features(self):
        """Extract features from audio files and cache them"""
        cache_dir = Path('data/features_cache')
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Define splits
        splits = {
            'train': {
                'chainsaw': Path('data/processed/train/chainsaw'),
                'non_chainsaw': Path('data/processed/train/non_chainsaw')
            },
            'val': {
                'chainsaw': Path('data/processed/val/chainsaw'),
                'non_chainsaw': Path('data/processed/val/non_chainsaw')
            },
            'test': {
                'chainsaw': Path('data/processed/test/chainsaw'),
                'non_chainsaw': Path('data/processed/test/non_chainsaw')
            }
        }
        
        for split_name, dirs in splits.items():
            print(f"\nExtracting {split_name} features...")
            features_list = []
            labels_list = []
            failed_count = 0
            failed_examples = []
            
            # Process chainsaw files (label = 1)
            chainsaw_files = list(dirs['chainsaw'].glob('*.wav'))
            print(f"  Processing {len(chainsaw_files)} chainsaw files...")
            for audio_file in tqdm(chainsaw_files, desc=f"{split_name} chainsaw"):
                try:
                    features = self.extract_features(str(audio_file))
                    if features is not None and len(features) > 0:
                        features_list.append(features)
                        labels_list.append(1)
                    else:
                        failed_count += 1
                        if len(failed_examples) < 5:
                            failed_examples.append(f"{audio_file.name} (returned None)")
                except Exception as e:
                    failed_count += 1
                    if len(failed_examples) < 5:
                        failed_examples.append(f"{audio_file.name}: {str(e)[:50]}")
            
            # Process non-chainsaw files (label = 0)
            non_chainsaw_files = list(dirs['non_chainsaw'].glob('*.wav'))
            print(f"  Processing {len(non_chainsaw_files)} non-chainsaw files...")
            for audio_file in tqdm(non_chainsaw_files, desc=f"{split_name} non-chainsaw"):
                try:
                    features = self.extract_features(str(audio_file))
                    if features is not None and len(features) > 0:
                        features_list.append(features)
                        labels_list.append(0)
                    else:
                        failed_count += 1
                        if len(failed_examples) < 5:
                            failed_examples.append(f"{audio_file.name} (returned None)")
                except Exception as e:
                    failed_count += 1
                    if len(failed_examples) < 5:
                        failed_examples.append(f"{audio_file.name}: {str(e)[:50]}")
            
            # Print failed examples
            if failed_count > 0:
                print(f"  WARNING: Failed to extract {failed_count} files")
                print(f"  Examples: {failed_examples[:5]}")
            
            # Ensure we have data
            if len(features_list) == 0:
                raise ValueError(f"No valid features extracted for {split_name} split! All {len(chainsaw_files) + len(non_chainsaw_files)} files failed.")
            
            # Save to cache
            features_array = np.array(features_list)
            labels_array = np.array(labels_list)
            
            cache_file = cache_dir / f'{split_name}_features.npz'
            np.savez(cache_file, features=features_array, labels=labels_array)
            print(f"  Saved {split_name} features: {features_array.shape} (failed: {failed_count})")
        
        # Now load the features
        print("\nLoading extracted features...")
        data = np.load(cache_dir / 'train_features.npz')
        self.X_train = data['features']
        self.y_train = data['labels']
        
        data = np.load(cache_dir / 'val_features.npz')
        self.X_val = data['features']
        self.y_val = data['labels']
        
        data = np.load(cache_dir / 'test_features.npz')
        self.X_test = data['features']
        self.y_test = data['labels']
        
        print(f"\nExtracted features:")
        print(f"  Train: {self.X_train.shape}")
        print(f"  Val:   {self.X_val.shape}")
        print(f"  Test:  {self.X_test.shape}")
        
        print(f"\nOriginal class distribution:")
        unique, counts = np.unique(self.y_train, return_counts=True)
        for label, count in zip(unique, counts):
            print(f"  Class {label}: {count} ({count/len(self.y_train)*100:.1f}%)")
        
        # Normalize features
        print("\nNormalizing features...")
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_val = self.scaler.transform(self.X_val)
        self.X_test = self.scaler.transform(self.X_test)
        
        # Save scaler
        with open(self.output_dir / 'scaler.pkl', 'wb') as f:
            pickle.dump(self.scaler, f)
    
    def apply_smote(self):
        """Apply advanced SMOTE techniques to balance the training data"""
        print("\nApplying SMOTE-ENN for class balancing...")
        
        # Use SMOTE-ENN (better cleaning than Tomek)
        smote_enn = SMOTEENN(random_state=42)
        self.X_train_balanced, self.y_train_balanced = smote_enn.fit_resample(
            self.X_train, self.y_train
        )
        
        print(f"\nBalanced class distribution:")
        unique, counts = np.unique(self.y_train_balanced, return_counts=True)
        for label, count in zip(unique, counts):
            print(f"  Class {label}: {count} ({count/len(self.y_train_balanced)*100:.1f}%)")
        
        print(f"\nNew training set size: {self.X_train_balanced.shape[0]} (was {self.X_train.shape[0]})")
    
    def train_models(self):
        """Train multiple lightweight ML models with class balance"""
        print("\n" + "="*70)
        print("TRAINING BALANCED MODELS")
        print("="*70)
        
        self.models = {}
        
        # 1. SVM with balanced class weight
        print("\n1. Training SVM (balanced)...")
        svm_model = SVC(kernel='rbf', C=10, gamma='scale', 
                       class_weight='balanced',
                       probability=True, random_state=42)
        svm_model.fit(self.X_train_balanced, self.y_train_balanced)
        self.models['SVM (Balanced)'] = svm_model
        print("   ✓ SVM trained")
        
        # 2. Random Forest with balanced class weight
        print("\n2. Training Random Forest (balanced)...")
        rf_model = RandomForestClassifier(
            n_estimators=200, 
            max_depth=20,
            min_samples_split=5, 
            min_samples_leaf=2,
            class_weight='balanced',
            random_state=42, 
            n_jobs=-1
        )
        rf_model.fit(self.X_train_balanced, self.y_train_balanced)
        self.models['Random Forest (Balanced)'] = rf_model
        print("   ✓ Random Forest trained")
        
        # 3. Gradient Boosting on balanced data
        print("\n3. Training Gradient Boosting (balanced)...")
        gb_model = GradientBoostingClassifier(
            n_estimators=150, 
            max_depth=6,
            learning_rate=0.1, 
            subsample=0.8,
            random_state=42
        )
        gb_model.fit(self.X_train_balanced, self.y_train_balanced)
        self.models['Gradient Boosting (Balanced)'] = gb_model
        print("   ✓ Gradient Boosting trained")
        
        # 4. XGBoost (if available)
        if XGBOOST_AVAILABLE:
            print("\n4. Training XGBoost (balanced)...")
            # Calculate scale_pos_weight for imbalance
            scale_pos_weight = len(self.y_train[self.y_train == 0]) / len(self.y_train[self.y_train == 1])
            
            xgb_model = xgb.XGBClassifier(
                n_estimators=150, 
                max_depth=6,
                learning_rate=0.1,
                scale_pos_weight=scale_pos_weight,
                random_state=42,
                eval_metric='logloss'
            )
            xgb_model.fit(self.X_train, self.y_train)  # Use original for XGBoost
            self.models['XGBoost (Balanced)'] = xgb_model
            print("   ✓ XGBoost trained")
        
        # 5. Logistic Regression with balanced class weight
        print("\n5. Training Logistic Regression (balanced)...")
        lr_model = LogisticRegression(
            C=1.0, 
            class_weight='balanced',
            max_iter=1000, 
            random_state=42
        )
        lr_model.fit(self.X_train_balanced, self.y_train_balanced)
        self.models['Logistic Regression (Balanced)'] = lr_model
        print("   ✓ Logistic Regression trained")
        
        # 6. Balanced Random Forest (specialized)
        print("\n6. Training Balanced Random Forest...")
        brf_model = BalancedRandomForestClassifier(
            n_estimators=400,
            max_depth=25,
            random_state=42,
            n_jobs=-1
        )
        brf_model.fit(self.X_train, self.y_train)  # Use original data
        self.models['Balanced Random Forest'] = brf_model
        print("   ✓ Balanced Random Forest trained")
        
        # 7. Easy Ensemble (multiple balanced ensembles)
        print("\n7. Training Easy Ensemble...")
        ee_model = EasyEnsembleClassifier(
            n_estimators=15,
            random_state=42,
            n_jobs=-1
        )
        ee_model.fit(self.X_train, self.y_train)  # Use original data
        self.models['Easy Ensemble'] = ee_model
        print("   ✓ Easy Ensemble trained")
        
        # 8. Extra Trees (often better than RF)
        print("\n8. Training Extra Trees...")
        et_model = ExtraTreesClassifier(
            n_estimators=400,
            max_depth=25,
            min_samples_split=2,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        et_model.fit(self.X_train_balanced, self.y_train_balanced)
        self.models['Extra Trees'] = et_model
        print("   ✓ Extra Trees trained")
        
        # 9. Voting Ensemble
        print("\n9. Creating Voting Ensemble...")
        voting_models = [
            ('rf', self.models['Random Forest (Balanced)']),
            ('gb', self.models['Gradient Boosting (Balanced)']),
            ('et', et_model)
        ]
        
        if XGBOOST_AVAILABLE:
            voting_models.append(('xgb', self.models['XGBoost (Balanced)']))
        
        voting_model = VotingClassifier(
            estimators=voting_models,
            voting='soft',
            n_jobs=-1
        )
        voting_model.fit(self.X_train_balanced, self.y_train_balanced)
        self.models['Voting Ensemble'] = voting_model
        print("   ✓ Voting Ensemble trained")
        
        print(f"\n✓ Trained {len(self.models)} models successfully")
    
    def evaluate_models(self):
        """Evaluate all trained models"""
        print("\n" + "="*70)
        print("EVALUATING MODELS")
        print("="*70)
        
        for name, model in self.models.items():
            print(f"\nEvaluating {name}...")
            
            # Predictions
            y_val_pred = model.predict(self.X_val)
            y_test_pred = model.predict(self.X_test)
            
            # Get probabilities
            if hasattr(model, 'predict_proba'):
                y_val_proba = model.predict_proba(self.X_val)[:, 1]
                y_test_proba = model.predict_proba(self.X_test)[:, 1]
            else:
                y_val_proba = None
                y_test_proba = None
            
            # Calculate metrics
            val_accuracy = accuracy_score(self.y_val, y_val_pred)
            test_accuracy = accuracy_score(self.y_test, y_test_pred)
            
            val_f1 = f1_score(self.y_val, y_val_pred)
            test_f1 = f1_score(self.y_test, y_test_pred)
            
            test_precision = precision_score(self.y_test, y_test_pred)
            test_recall = recall_score(self.y_test, y_test_pred)
            
            test_balanced_acc = balanced_accuracy_score(self.y_test, y_test_pred)
            
            cm = confusion_matrix(self.y_test, y_test_pred)
            
            if y_test_proba is not None:
                test_auc = roc_auc_score(self.y_test, y_test_proba)
            else:
                test_auc = None
            
            # Store results
            self.results[name] = {
                'val_accuracy': val_accuracy,
                'test_accuracy': test_accuracy,
                'val_f1': val_f1,
                'test_f1': test_f1,
                'test_precision': test_precision,
                'test_recall': test_recall,
                'balanced_accuracy': test_balanced_acc,
                'test_auc': test_auc,
                'confusion_matrix': cm,
                'y_test_pred': y_test_pred,
                'y_test_proba': y_test_proba
            }
            
            # Print results
            print(f"  Val F1:         {val_f1:.4f}")
            print(f"  Test F1:        {test_f1:.4f}")
            print(f"  Test Accuracy:  {test_accuracy:.4f}")
            print(f"  Balanced Acc:   {test_balanced_acc:.4f}")
            print(f"  Precision:      {test_precision:.4f}")
            print(f"  Recall:         {test_recall:.4f}")
            if test_auc:
                print(f"  AUC:            {test_auc:.4f}")
        
        # Find best model by validation F1
        best_model_name = max(self.results.items(), 
                            key=lambda x: x[1]['test_f1'])[0]
        self.best_model_name = best_model_name
        
        print(f"\n{'='*70}")
        print(f"BEST MODEL: {best_model_name}")
        print(f"Test F1: {self.results[best_model_name]['test_f1']:.4f}")
        print(f"{'='*70}")
    
    def plot_results(self):
        """Create visualizations"""
        print("\nCreating visualizations...")
        
        # Set style
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
        
        # 1. Model comparison
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        models = list(self.results.keys())
        test_f1s = [self.results[m]['test_f1'] for m in models]
        test_accs = [self.results[m]['test_accuracy'] for m in models]
        balanced_accs = [self.results[m]['balanced_accuracy'] for m in models]
        recalls = [self.results[m]['test_recall'] for m in models]
        
        # F1 scores
        axes[0, 0].barh(models, test_f1s, color='skyblue')
        axes[0, 0].set_xlabel('F1 Score', fontsize=12)
        axes[0, 0].set_title('Test F1 Scores by Model', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlim([0, 1])
        axes[0, 0].axvline(x=0.5, color='red', linestyle='--', alpha=0.5)
        
        # Accuracies
        x = np.arange(len(models))
        width = 0.35
        axes[0, 1].bar(x - width/2, test_accs, width, label='Test Accuracy', color='lightgreen')
        axes[0, 1].bar(x + width/2, balanced_accs, width, label='Balanced Accuracy', color='orange')
        axes[0, 1].set_ylabel('Accuracy', fontsize=12)
        axes[0, 1].set_title('Accuracy Comparison', fontsize=14, fontweight='bold')
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels(models, rotation=45, ha='right')
        axes[0, 1].legend()
        axes[0, 1].set_ylim([0, 1])
        
        # Precision vs Recall
        precisions = [self.results[m]['test_precision'] for m in models]
        axes[1, 0].scatter(recalls, precisions, s=100, alpha=0.6, color='purple')
        for i, model in enumerate(models):
            axes[1, 0].annotate(model, (recalls[i], precisions[i]), 
                              fontsize=8, alpha=0.7)
        axes[1, 0].set_xlabel('Recall', fontsize=12)
        axes[1, 0].set_ylabel('Precision', fontsize=12)
        axes[1, 0].set_title('Precision vs Recall', fontsize=14, fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_xlim([0, 1])
        axes[1, 0].set_ylim([0, 1])
        
        # Best model confusion matrix
        cm = self.results[self.best_model_name]['confusion_matrix']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 1])
        axes[1, 1].set_title(f'Confusion Matrix - {self.best_model_name}', 
                           fontsize=14, fontweight='bold')
        axes[1, 1].set_ylabel('True Label')
        axes[1, 1].set_xlabel('Predicted Label')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'model_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. ROC curve for models with probabilities
        fig, ax = plt.subplots(figsize=(10, 8))
        
        for name in self.models.keys():
            if self.results[name]['y_test_proba'] is not None:
                fpr, tpr, _ = roc_curve(self.y_test, self.results[name]['y_test_proba'])
                auc = self.results[name]['test_auc']
                ax.plot(fpr, tpr, label=f'{name} (AUC={auc:.3f})', linewidth=2)
        
        ax.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title('ROC Curves', fontsize=14, fontweight='bold')
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'roc_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  Saved visualizations to {self.output_dir}")
    
    def save_results(self):
        """Save models and results"""
        print("\nSaving results...")
        
        # Save best model
        best_model = self.models[self.best_model_name]
        with open(self.output_dir / 'best_model.pkl', 'wb') as f:
            pickle.dump(best_model, f)
        
        print(f"  Saved best model: {self.best_model_name}")
        
        # Save all results to JSON
        results_json = {}
        for name, metrics in self.results.items():
            results_json[name] = {
                k: v for k, v in metrics.items() 
                if k not in ['y_test_pred', 'y_test_proba', 'confusion_matrix']
            }
            # Convert confusion matrix to list
            results_json[name]['confusion_matrix'] = metrics['confusion_matrix'].tolist()
        
        with open(self.output_dir / 'results.json', 'w') as f:
            json.dump(results_json, f, indent=2)
        
        # Detailed classification report
        y_pred = self.results[self.best_model_name]['y_test_pred']
        report = classification_report(self.y_test, y_pred, 
                                      target_names=['Non-Chainsaw', 'Chainsaw'])
        
        with open(self.output_dir / 'classification_report.txt', 'w') as f:
            f.write("CHAINSAW DETECTION - BALANCED ML MODELS\n")
            f.write("="*70 + "\n\n")
            f.write(f"Best Model: {self.best_model_name}\n")
            f.write("="*70 + "\n\n")
            f.write(report)
            f.write("\n\nDetailed Metrics:\n")
            f.write("-"*70 + "\n")
            for metric, value in self.results[self.best_model_name].items():
                if metric not in ['y_test_pred', 'y_test_proba', 'confusion_matrix']:
                    if isinstance(value, float):
                        f.write(f"{metric}: {value:.4f}\n")
        
        print(f"  Saved results to: {self.output_dir}")
    
    def run_complete_pipeline(self):
        """Run the complete balanced training pipeline"""
        print("\n" + "="*70)
        print("BALANCED ML MODELS FOR CHAINSAW DETECTION")
        print("="*70)
        
        # Load data
        self.load_cached_features()
        
        # Apply SMOTE
        self.apply_smote()
        
        # Train models
        self.train_models()
        
        # Evaluate
        self.evaluate_models()
        
        # Visualize
        self.plot_results()
        
        # Save
        self.save_results()
        
        print("\n" + "="*70)
        print("TRAINING COMPLETE!")
        print("="*70)
        print(f"\nOutput: {self.output_dir.absolute()}")
        print(f"Best Model: {self.best_model_name}")
        print(f"  Test F1:        {self.results[self.best_model_name]['test_f1']:.4f}")
        print(f"  Test Accuracy:  {self.results[self.best_model_name]['test_accuracy']:.4f}")
        print(f"  Balanced Acc:   {self.results[self.best_model_name]['balanced_accuracy']:.4f}")
        print(f"  Recall:         {self.results[self.best_model_name]['test_recall']:.4f}")
        print(f"  Precision:      {self.results[self.best_model_name]['test_precision']:.4f}")


if __name__ == "__main__":
    trainer = ImprovedMLTrainer()
    trainer.run_complete_pipeline()
