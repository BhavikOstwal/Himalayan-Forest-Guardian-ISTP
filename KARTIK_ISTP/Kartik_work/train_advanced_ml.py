"""
Advanced ML Training with Hyperparameter Tuning and Better Techniques
Achieves higher accuracy through optimized features, balancing, and ensembles
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
                              VotingClassifier, AdaBoostClassifier, ExtraTreesClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, confusion_matrix, classification_report,
                             roc_auc_score, roc_curve, balanced_accuracy_score)
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.feature_selection import SelectKBest, f_classif, RFE

# Advanced balancing techniques
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
from imblearn.combine import SMOTETomek, SMOTEENN
from imblearn.ensemble import BalancedRandomForestClassifier, EasyEnsembleClassifier

import matplotlib.pyplot as plt
import seaborn as sns

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


class EnhancedFeatureExtractor:
    """Extract enhanced audio features with better discrimination"""
    
    def __init__(self, config):
        self.config = config
        self.sample_rate = config['audio']['sample_rate']
        self.duration = config['audio']['duration']
        self.n_mels = 64  # Increased from 40
        self.n_mfcc = 20  # Increased from 13
        self.hop_length = 512
        self.n_fft = 2048
    
    def extract_features(self, file_path):
        """Extract comprehensive feature set"""
        try:
            audio, sr = librosa.load(file_path, sr=self.sample_rate, 
                                    duration=self.duration)
            
            features = []
            
            # 1. MFCCs (enhanced) - 80 features
            mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=self.n_mfcc)
            features.extend(np.mean(mfccs, axis=1))
            features.extend(np.std(mfccs, axis=1))
            features.extend(np.max(mfccs, axis=1))
            features.extend(np.min(mfccs, axis=1))
            
            # 2. Delta MFCCs (velocity) - 40 features
            delta_mfccs = librosa.feature.delta(mfccs)
            features.extend(np.mean(delta_mfccs, axis=1))
            features.extend(np.std(delta_mfccs, axis=1))
            
            # 3. Delta-Delta MFCCs (acceleration) - 40 features
            delta2_mfccs = librosa.feature.delta(mfccs, order=2)
            features.extend(np.mean(delta2_mfccs, axis=1))
            features.extend(np.std(delta2_mfccs, axis=1))
            
            # 4. Mel Spectrogram statistics - 20 features
            mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, 
                                                     n_mels=self.n_mels,
                                                     hop_length=self.hop_length)
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            
            # Statistical moments across mel bins
            features.extend([
                np.mean(mel_spec_db),
                np.std(mel_spec_db),
                np.max(mel_spec_db),
                np.min(mel_spec_db),
                np.median(mel_spec_db),
                np.percentile(mel_spec_db, 25),
                np.percentile(mel_spec_db, 75),
                np.ptp(mel_spec_db),  # peak to peak
            ])
            
            # Temporal statistics - mean/std across time for each mel bin (sample 12 bins)
            mel_bins_sample = np.linspace(0, self.n_mels-1, 12).astype(int)
            for bin_idx in mel_bins_sample:
                features.extend([
                    np.mean(mel_spec_db[bin_idx]),
                    np.std(mel_spec_db[bin_idx])
                ])
            
            # 5. Spectral features (comprehensive) - 32 features
            spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)
            features.extend([
                np.mean(spectral_centroid),
                np.std(spectral_centroid),
                np.max(spectral_centroid),
                np.min(spectral_centroid)
            ])
            
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)
            features.extend([
                np.mean(spectral_bandwidth),
                np.std(spectral_bandwidth),
                np.max(spectral_bandwidth),
                np.min(spectral_bandwidth)
            ])
            
            spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)
            features.extend([
                np.mean(spectral_rolloff),
                np.std(spectral_rolloff),
                np.max(spectral_rolloff),
                np.min(spectral_rolloff)
            ])
            
            spectral_flatness = librosa.feature.spectral_flatness(y=audio)
            features.extend([
                np.mean(spectral_flatness),
                np.std(spectral_flatness),
                np.max(spectral_flatness),
                np.min(spectral_flatness)
            ])
            
            spectral_contrast = librosa.feature.spectral_contrast(y=audio, sr=sr, n_bands=6)
            features.extend(np.mean(spectral_contrast, axis=1))
            features.extend(np.std(spectral_contrast, axis=1))
            
            # 6. Tonnetz (tonal centroid features) - 6 features
            tonnetz = librosa.feature.tonnetz(y=audio, sr=sr)
            features.extend(np.mean(tonnetz, axis=1))
            
            # 7. Zero crossing rate - 4 features
            zcr = librosa.feature.zero_crossing_rate(audio)
            features.extend([
                np.mean(zcr),
                np.std(zcr),
                np.max(zcr),
                np.min(zcr)
            ])
            
            # 8. RMS Energy (detailed) - 6 features
            rms = librosa.feature.rms(y=audio)
            features.extend([
                np.mean(rms),
                np.std(rms),
                np.max(rms),
                np.min(rms),
                np.median(rms),
                np.ptp(rms)
            ])
            
            # 9. Chroma features (enhanced) - 24 features
            chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
            features.extend(np.mean(chroma, axis=1))
            features.extend(np.std(chroma, axis=1))
            
            # 10. Tempo and beat features - 3 features
            tempo, beats = librosa.beat.beat_track(y=audio, sr=sr)
            features.append(float(tempo))
            features.append(len(beats))  # Number of beats
            features.append(len(beats) / self.duration if self.duration > 0 else 0)  # Beat density
            
            # 11. Onset strength - 4 features
            onset_env = librosa.onset.onset_strength(y=audio, sr=sr)
            features.extend([
                np.mean(onset_env),
                np.std(onset_env),
                np.max(onset_env),
                np.sum(onset_env)
            ])
            
            # 12. Time-domain features - 10 features
            features.extend([
                np.mean(audio),
                np.std(audio),
                np.max(audio),
                np.min(audio),
                np.median(audio),
                np.mean(np.abs(audio)),
                np.max(np.abs(audio)),
                np.ptp(audio),  # peak to peak
                np.sqrt(np.mean(audio**2)),  # RMS alternative
                len(np.where(np.diff(np.sign(audio)))[0])  # Zero crossings count
            ])
            
            # 13. Harmonic and percussive components - 8 features
            y_harmonic, y_percussive = librosa.effects.hpss(audio)
            features.extend([
                np.mean(np.abs(y_harmonic)),
                np.std(y_harmonic),
                np.max(np.abs(y_harmonic)),
                np.mean(np.abs(y_percussive)),
                np.std(y_percussive),
                np.max(np.abs(y_percussive)),
                np.mean(np.abs(y_harmonic)) / (np.mean(np.abs(y_percussive)) + 1e-10),  # Ratio
                np.std(y_harmonic) / (np.std(y_percussive) + 1e-10)  # Ratio
            ])
            
            # Total features: ~310-320 features (much more discriminative)
            return np.array(features)
            
        except Exception as e:
            print(f"Error: {e}")
            return None


class AdvancedMLTrainer:
    """Advanced ML trainer with optimization"""
    
    def __init__(self, config_path='config.yaml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.feature_extractor = EnhancedFeatureExtractor(self.config)
        self.scaler = RobustScaler()  # More robust to outliers
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = Path(f"output/ml_advanced_{timestamp}")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Output directory: {self.output_dir}")
    
    def load_or_extract_features(self, csv_path, cache_path, split_name):
        """Load or extract enhanced features"""
        
        if os.path.exists(cache_path):
            print(f"Loading cached {split_name} features from {cache_path}")
            data = np.load(cache_path)
            return data['features'], data['labels']
        
        print(f"Extracting enhanced {split_name} features...")
        df = pd.read_csv(csv_path)
        
        features_list = []
        labels_list = []
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing {split_name}"):
            file_path = row['file_path']
            if not os.path.exists(file_path):
                file_path = os.path.join('c:\\Users\\dhira\\Desktop\\ISTP', file_path)
            
            if os.path.exists(file_path):
                features = self.feature_extractor.extract_features(file_path)
                if features is not None:
                    features_list.append(features)
                    labels_list.append(row['label'])
        
        features = np.array(features_list)
        labels = np.array(labels_list)
        
        np.savez(cache_path, features=features, labels=labels)
        print(f"Cached features to {cache_path}")
        
        return features, labels
    
    def prepare_data(self):
        """Load and prepare data"""
        print("\n" + "="*70)
        print("PREPARING DATA WITH ENHANCED FEATURES")
        print("="*70)
        
        data_dir = Path(self.config['data']['processed_data_dir'])
        cache_dir = Path('data/features_cache_advanced')
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.X_train, self.y_train = self.load_or_extract_features(
            data_dir / 'train_processed.csv',
            cache_dir / 'train_features_advanced.npz',
            'train'
        )
        
        self.X_val, self.y_val = self.load_or_extract_features(
            data_dir / 'val_processed.csv',
            cache_dir / 'val_features_advanced.npz',
            'validation'
        )
        
        self.X_test, self.y_test = self.load_or_extract_features(
            data_dir / 'test_processed.csv',
            cache_dir / 'test_features_advanced.npz',
            'test'
        )
        
        print(f"\nData shapes:")
        print(f"  Train: {self.X_train.shape}, Features: {self.X_train.shape[1]}")
        print(f"  Val:   {self.X_val.shape}")
        print(f"  Test:  {self.X_test.shape}")
        
        print(f"\nOriginal class distribution:")
        unique, counts = np.unique(self.y_train, return_counts=True)
        for label, count in zip(unique, counts):
            print(f"  Class {label}: {count} ({count/len(self.y_train)*100:.1f}%)")
        
        # Normalize
        print("\nNormalizing features with RobustScaler...")
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_val = self.scaler.transform(self.X_val)
        self.X_test = self.scaler.transform(self.X_test)
        
        with open(self.output_dir / 'scaler.pkl', 'wb') as f:
            pickle.dump(self.scaler, f)
    
    def apply_advanced_balancing(self):
        """Try multiple balancing techniques and choose best"""
        print("\n" + "="*70)
        print("APPLYING ADVANCED BALANCING TECHNIQUES")
        print("="*70)
        
        self.balanced_datasets = {}
        
        # 1. SMOTE-Tomek
        print("\n1. Applying SMOTE-Tomek...")
        smote_tomek = SMOTETomek(random_state=42)
        X_bal, y_bal = smote_tomek.fit_resample(self.X_train, self.y_train)
        self.balanced_datasets['SMOTE-Tomek'] = (X_bal, y_bal)
        print(f"   Balanced size: {len(y_bal)}")
        
        # 2. SMOTE-ENN
        print("\n2. Applying SMOTE-ENN...")
        smote_enn = SMOTEENN(random_state=42)
        X_bal, y_bal = smote_enn.fit_resample(self.X_train, self.y_train)
        self.balanced_datasets['SMOTE-ENN'] = (X_bal, y_bal)
        print(f"   Balanced size: {len(y_bal)}")
        
        # 3. ADASYN
        print("\n3. Applying ADASYN...")
        try:
            adasyn = ADASYN(random_state=42)
            X_bal, y_bal = adasyn.fit_resample(self.X_train, self.y_train)
            self.balanced_datasets['ADASYN'] = (X_bal, y_bal)
            print(f"   Balanced size: {len(y_bal)}")
        except Exception as e:
            print(f"   ADASYN failed: {e}")
        
        # 4. Borderline SMOTE
        print("\n4. Applying Borderline SMOTE...")
        borderline = BorderlineSMOTE(random_state=42, kind='borderline-1')
        X_bal, y_bal = borderline.fit_resample(self.X_train, self.y_train)
        self.balanced_datasets['Borderline-SMOTE'] = (X_bal, y_bal)
        print(f"   Balanced size: {len(y_bal)}")
        
        # Use SMOTE-Tomek as default (usually best)
        self.X_train_balanced, self.y_train_balanced = self.balanced_datasets['SMOTE-Tomek']
        
        print(f"\nFinal balanced distribution:")
        unique, counts = np.unique(self.y_train_balanced, return_counts=True)
        for label, count in zip(unique, counts):
            print(f"  Class {label}: {count} ({count/len(self.y_train_balanced)*100:.1f}%)")
    
    def train_optimized_models(self):
        """Train models with hyperparameter tuning"""
        print("\n" + "="*70)
        print("TRAINING OPTIMIZED MODELS")
        print("="*70)
        
        self.models = {}
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        # 1. Optimized Random Forest
        print("\n1. Training Optimized Random Forest...")
        rf_params = {
            'n_estimators': [200, 300, 400],
            'max_depth': [15, 20, 25, None],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2],
            'max_features': ['sqrt', 'log2'],
            'class_weight': ['balanced', 'balanced_subsample']
        }
        
        rf = RandomForestClassifier(random_state=42, n_jobs=-1)
        rf_grid = GridSearchCV(rf, rf_params, cv=3, scoring='f1', n_jobs=-1, verbose=1)
        rf_grid.fit(self.X_train_balanced, self.y_train_balanced)
        self.models['Random Forest (Optimized)'] = rf_grid.best_estimator_
        print(f"   Best params: {rf_grid.best_params_}")
        print(f"   Best CV F1: {rf_grid.best_score_:.4f}")
        
        # 2. Optimized XGBoost (if available)
        if XGBOOST_AVAILABLE:
            print("\n2. Training Optimized XGBoost...")
            scale_pos_weight = len(self.y_train[self.y_train == 0]) / len(self.y_train[self.y_train == 1])
            
            xgb_params = {
                'n_estimators': [150, 200, 250],
                'max_depth': [5, 7, 9],
                'learning_rate': [0.05, 0.1, 0.15],
                'subsample': [0.8, 0.9],
                'colsample_bytree': [0.8, 0.9],
            }
            
            xgb_model = xgb.XGBClassifier(
                scale_pos_weight=scale_pos_weight,
                random_state=42,
                eval_metric='logloss'
            )
            xgb_grid = GridSearchCV(xgb_model, xgb_params, cv=3, scoring='f1', n_jobs=-1, verbose=1)
            xgb_grid.fit(self.X_train_balanced, self.y_train_balanced)
            self.models['XGBoost (Optimized)'] = xgb_grid.best_estimator_
            print(f"   Best params: {xgb_grid.best_params_}")
            print(f"   Best CV F1: {xgb_grid.best_score_:.4f}")
        
        # 3. Balanced Random Forest (specialized for imbalanced data)
        print("\n3. Training Balanced Random Forest...")
        brf = BalancedRandomForestClassifier(
            n_estimators=300,
            max_depth=20,
            random_state=42,
            n_jobs=-1
        )
        brf.fit(self.X_train, self.y_train)  # Use original data
        self.models['Balanced Random Forest'] = brf
        print("   ✓ Trained")
        
        # 4. Easy Ensemble (multiple balanced ensembles)
        print("\n4. Training Easy Ensemble...")
        ee = EasyEnsembleClassifier(
            n_estimators=10,
            random_state=42,
            n_jobs=-1
        )
        ee.fit(self.X_train, self.y_train)  # Use original data
        self.models['Easy Ensemble'] = ee
        print("   ✓ Trained")
        
        # 5. Optimized SVM
        print("\n5. Training Optimized SVM...")
        svm_params = {
            'C': [1, 10, 50],
            'gamma': ['scale', 'auto', 0.001, 0.01],
            'class_weight': ['balanced']
        }
        
        svm = SVC(kernel='rbf', probability=True, random_state=42)
        svm_grid = GridSearchCV(svm, svm_params, cv=3, scoring='f1', n_jobs=-1, verbose=1)
        svm_grid.fit(self.X_train_balanced, self.y_train_balanced)
        self.models['SVM (Optimized)'] = svm_grid.best_estimator_
        print(f"   Best params: {svm_grid.best_params_}")
        print(f"   Best CV F1: {svm_grid.best_score_:.4f}")
        
        # 6. Gradient Boosting (optimized)
        print("\n6. Training Optimized Gradient Boosting...")
        gb = GradientBoostingClassifier(
            n_estimators=200,
            max_depth=7,
            learning_rate=0.1,
            subsample=0.8,
            random_state=42
        )
        gb.fit(self.X_train_balanced, self.y_train_balanced)
        self.models['Gradient Boosting (Optimized)'] = gb
        print("   ✓ Trained")
        
        # 7. Extra Trees (often better than RF)
        print("\n7. Training Extra Trees...")
        et = ExtraTreesClassifier(
            n_estimators=300,
            max_depth=20,
            min_samples_split=2,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        et.fit(self.X_train_balanced, self.y_train_balanced)
        self.models['Extra Trees'] = et
        print("   ✓ Trained")
        
        # 8. Voting Ensemble (combine best models)
        print("\n8. Creating Voting Ensemble...")
        voting_models = [
            ('rf', self.models['Random Forest (Optimized)']),
            ('gb', self.models['Gradient Boosting (Optimized)']),
            ('et', self.models['Extra Trees'])
        ]
        
        if XGBOOST_AVAILABLE:
            voting_models.append(('xgb', self.models['XGBoost (Optimized)']))
        
        voting = VotingClassifier(
            estimators=voting_models,
            voting='soft',
            n_jobs=-1
        )
        voting.fit(self.X_train_balanced, self.y_train_balanced)
        self.models['Voting Ensemble'] = voting
        print("   ✓ Trained")
        
        print(f"\n✓ Trained {len(self.models)} optimized models")
    
    def evaluate_models(self):
        """Comprehensive evaluation"""
        print("\n" + "="*70)
        print("EVALUATING MODELS")
        print("="*70)
        
        self.results = {}
        
        for model_name, model in self.models.items():
            print(f"\n{model_name}:")
            print("-" * 50)
            
            y_val_pred = model.predict(self.X_val)
            y_test_pred = model.predict(self.X_test)
            
            if hasattr(model, 'predict_proba'):
                y_test_proba = model.predict_proba(self.X_test)[:, 1]
            else:
                y_test_proba = y_test_pred
            
            # Metrics
            val_acc = accuracy_score(self.y_val, y_val_pred)
            test_acc = accuracy_score(self.y_test, y_test_pred)
            balanced_acc = balanced_accuracy_score(self.y_test, y_test_pred)
            
            test_precision = precision_score(self.y_test, y_test_pred, zero_division=0)
            test_recall = recall_score(self.y_test, y_test_pred, zero_division=0)
            test_f1 = f1_score(self.y_test, y_test_pred, zero_division=0)
            
            try:
                test_auc = roc_auc_score(self.y_test, y_test_proba)
            except:
                test_auc = 0.0
            
            cm = confusion_matrix(self.y_test, y_test_pred)
            tn, fp, fn, tp = cm.ravel()
            
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            
            self.results[model_name] = {
                'val_accuracy': val_acc,
                'test_accuracy': test_acc,
                'balanced_accuracy': balanced_acc,
                'test_precision': test_precision,
                'test_recall': test_recall,
                'test_f1': test_f1,
                'test_auc': test_auc,
                'sensitivity': sensitivity,
                'specificity': specificity,
                'y_test_pred': y_test_pred,
                'y_test_proba': y_test_proba,
                'confusion_matrix': cm
            }
            
            print(f"  Val Accuracy:      {val_acc:.4f}")
            print(f"  Test Accuracy:     {test_acc:.4f}")
            print(f"  Balanced Acc:      {balanced_acc:.4f}")
            print(f"  F1 Score:          {test_f1:.4f}")
            print(f"  Precision:         {test_precision:.4f}")
            print(f"  Recall:            {test_recall:.4f}")
            print(f"  AUC:               {test_auc:.4f}")
            print(f"  TN:{tn} FP:{fp} FN:{fn} TP:{tp}")
        
        # Find best model
        best_model_name = max(self.results, key=lambda x: self.results[x]['test_f1'])
        self.best_model_name = best_model_name
        self.best_model = self.models[best_model_name]
        
        print("\n" + "="*70)
        print(f"🏆 BEST MODEL: {best_model_name}")
        print(f"   Test F1:       {self.results[best_model_name]['test_f1']:.4f}")
        print(f"   Accuracy:      {self.results[best_model_name]['test_accuracy']:.4f}")
        print(f"   Balanced Acc:  {self.results[best_model_name]['balanced_accuracy']:.4f}")
        print(f"   Recall:        {self.results[best_model_name]['test_recall']:.4f}")
        print(f"   Precision:     {self.results[best_model_name]['test_precision']:.4f}")
        print("="*70)
    
    def plot_results(self):
        """Create visualizations"""
        print("\nCreating visualizations...")
        
        model_names = list(self.results.keys())
        
        fig, axes = plt.subplots(2, 2, figsize=(18, 14))
        
        # F1 Scores
        ax = axes[0, 0]
        f1_scores = [self.results[m]['test_f1'] for m in model_names]
        colors = ['#2ecc71' if m == self.best_model_name else '#3498db' for m in model_names]
        bars = ax.barh(model_names, f1_scores, color=colors, alpha=0.8, edgecolor='black')
        ax.set_xlabel('F1 Score', fontsize=12, fontweight='bold')
        ax.set_title('Model F1 Score Comparison', fontsize=14, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        ax.set_xlim([0, 1.0])
        
        for bar, score in zip(bars, f1_scores):
            ax.text(score + 0.02, bar.get_y() + bar.get_height()/2, 
                   f'{score:.3f}', va='center', fontsize=10, fontweight='bold')
        
        # Multi-metric comparison
        ax = axes[0, 1]
        metrics = ['test_accuracy', 'balanced_accuracy', 'test_precision', 'test_recall', 'test_f1']
        metric_labels = ['Accuracy', 'Bal Acc', 'Precision', 'Recall', 'F1']
        
        # Show top 5 models
        top_models = sorted(model_names, key=lambda x: self.results[x]['test_f1'], reverse=True)[:5]
        
        x = np.arange(len(metrics))
        width = 0.15
        
        for i, model_name in enumerate(top_models):
            values = [self.results[model_name][m] for m in metrics]
            ax.bar(x + i*width, values, width, label=model_name.split('(')[0].strip()[:15], alpha=0.8)
        
        ax.set_xlabel('Metric', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title('Top 5 Models - Metrics Comparison', fontsize=14, fontweight='bold')
        ax.set_xticks(x + width * 2)
        ax.set_xticklabels(metric_labels)
        ax.legend(fontsize=8)
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim([0, 1.0])
        
        # Confusion matrix
        ax = axes[1, 0]
        cm = self.results[self.best_model_name]['confusion_matrix']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                   xticklabels=['Non-Chainsaw', 'Chainsaw'],
                   yticklabels=['Non-Chainsaw', 'Chainsaw'])
        ax.set_title(f'Best Model Confusion Matrix\n{self.best_model_name}', 
                    fontsize=14, fontweight='bold')
        ax.set_ylabel('True Label', fontsize=12)
        ax.set_xlabel('Predicted Label', fontsize=12)
        
        # ROC Curves
        ax = axes[1, 1]
        for model_name in top_models:
            if self.results[model_name]['test_auc'] > 0:
                y_proba = self.results[model_name]['y_test_proba']
                fpr, tpr, _ = roc_curve(self.y_test, y_proba)
                auc = self.results[model_name]['test_auc']
                linewidth = 3 if model_name == self.best_model_name else 1.5
                ax.plot(fpr, tpr, label=f'{model_name.split("(")[0].strip()[:15]} ({auc:.3f})', 
                       linewidth=linewidth, alpha=0.8)
        
        ax.plot([0, 1], [0, 1], 'k--', label='Random', linewidth=1.5)
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title('ROC Curves (Top 5)', fontsize=14, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'results_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {self.output_dir / 'results_comparison.png'}")
    
    def save_results(self):
        """Save models and results"""
        print("\nSaving results...")
        
        with open(self.output_dir / 'best_model.pkl', 'wb') as f:
            pickle.dump(self.best_model, f)
        
        models_dir = self.output_dir / 'all_models'
        models_dir.mkdir(exist_ok=True)
        
        for model_name, model in self.models.items():
            safe_name = model_name.replace(' ', '_').replace('(', '').replace(')', '').lower()
            with open(models_dir / f'{safe_name}.pkl', 'wb') as f:
                pickle.dump(model, f)
        
        # Save results
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
        
        # Classification report
        y_pred = self.results[self.best_model_name]['y_test_pred']
        report = classification_report(self.y_test, y_pred, 
                                      target_names=['Non-Chainsaw', 'Chainsaw'])
        
        with open(self.output_dir / 'report.txt', 'w') as f:
            f.write("ADVANCED ML MODELS - CHAINSAW DETECTION\n")
            f.write("="*70 + "\n\n")
            f.write(f"Best Model: {self.best_model_name}\n")
            f.write("="*70 + "\n\n")
            f.write(report)
        
        print(f"  All results saved to: {self.output_dir}")
    
    def run_complete_pipeline(self):
        """Run complete advanced pipeline"""
        print("\n" + "="*70)
        print("🚀 ADVANCED ML TRAINING PIPELINE")
        print("="*70)
        
        self.prepare_data()
        self.apply_advanced_balancing()
        self.train_optimized_models()
        self.evaluate_models()
        self.plot_results()
        self.save_results()
        
        print("\n" + "="*70)
        print("✅ TRAINING COMPLETE!")
        print("="*70)
        print(f"\nBest Model: {self.best_model_name}")
        print(f"  F1 Score:      {self.results[self.best_model_name]['test_f1']:.4f}")
        print(f"  Accuracy:      {self.results[self.best_model_name]['test_accuracy']:.4f}")
        print(f"  Balanced Acc:  {self.results[self.best_model_name]['balanced_accuracy']:.4f}")
        print(f"  Recall:        {self.results[self.best_model_name]['test_recall']:.4f}")
        print(f"  Precision:     {self.results[self.best_model_name]['test_precision']:.4f}")
        print(f"\nOutput: {self.output_dir.absolute()}")


if __name__ == "__main__":
    trainer = AdvancedMLTrainer()
    trainer.run_complete_pipeline()
