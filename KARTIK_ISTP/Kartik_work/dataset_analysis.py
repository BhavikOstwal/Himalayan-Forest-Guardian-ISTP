"""
Comprehensive Dataset Analysis for Chainsaw Detection
This script performs detailed analysis with multiple visualizations including:
- Class distribution bar charts
- PCA analysis
- t-SNE visualization
- Feature distributions
- Spectrograms
- Waveform analysis
"""

import os
import yaml
import numpy as np
import pandas as pd
import librosa
import librosa.display
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


class DatasetAnalyzer:
    def __init__(self, config_path='config.yaml'):
        """Initialize the analyzer with configuration"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.sample_rate = self.config['audio']['sample_rate']
        self.duration = self.config['audio']['duration']
        self.n_mels = self.config['audio']['n_mels']
        self.hop_length = self.config['audio']['hop_length']
        self.n_fft = self.config['audio']['n_fft']
        
        # Create output directory for plots
        self.output_dir = Path('analysis_plots')
        self.output_dir.mkdir(exist_ok=True)
        
        print(f"Analysis plots will be saved to: {self.output_dir}")
    
    def load_datasets(self):
        """Load train, validation, and test datasets"""
        data_dir = Path(self.config['data']['processed_data_dir'])
        
        self.train_df = pd.read_csv(data_dir / 'train_processed.csv')
        self.val_df = pd.read_csv(data_dir / 'val_processed.csv')
        self.test_df = pd.read_csv(data_dir / 'test_processed.csv')
        
        print(f"Train samples: {len(self.train_df)}")
        print(f"Validation samples: {len(self.val_df)}")
        print(f"Test samples: {len(self.test_df)}")
        
        # Combine all for overall statistics
        self.all_df = pd.concat([self.train_df, self.val_df, self.test_df], 
                                 ignore_index=True)
        self.all_df['split'] = (['train'] * len(self.train_df) + 
                                ['val'] * len(self.val_df) + 
                                ['test'] * len(self.test_df))
        
        return self.train_df, self.val_df, self.test_df
    
    def plot_class_distribution(self):
        """Create bar charts for class distribution"""
        print("\n1. Creating class distribution plots...")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Overall distribution
        ax = axes[0, 0]
        class_counts = self.all_df['label_name'].value_counts()
        bars = ax.bar(class_counts.index, class_counts.values, 
                      color=['#ff7f0e', '#1f77b4'], alpha=0.7, edgecolor='black')
        ax.set_title('Overall Class Distribution', fontsize=14, fontweight='bold')
        ax.set_xlabel('Class', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.grid(axis='y', alpha=0.3)
        
        # Add count labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}',
                   ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        # Distribution by split
        ax = axes[0, 1]
        split_class = pd.crosstab(self.all_df['split'], self.all_df['label_name'])
        split_class.plot(kind='bar', ax=ax, color=['#ff7f0e', '#1f77b4'], 
                        alpha=0.7, edgecolor='black')
        ax.set_title('Class Distribution by Split', fontsize=14, fontweight='bold')
        ax.set_xlabel('Split', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.legend(title='Class', fontsize=10)
        ax.grid(axis='y', alpha=0.3)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=0)
        
        # Percentage distribution
        ax = axes[1, 0]
        class_percentages = (self.all_df['label_name'].value_counts() / 
                           len(self.all_df) * 100)
        bars = ax.bar(class_percentages.index, class_percentages.values,
                     color=['#ff7f0e', '#1f77b4'], alpha=0.7, edgecolor='black')
        ax.set_title('Class Distribution (Percentage)', fontsize=14, fontweight='bold')
        ax.set_xlabel('Class', fontsize=12)
        ax.set_ylabel('Percentage (%)', fontsize=12)
        ax.grid(axis='y', alpha=0.3)
        
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}%',
                   ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        # Stacked bar chart
        ax = axes[1, 1]
        split_class_pct = split_class.div(split_class.sum(axis=1), axis=0) * 100
        split_class_pct.plot(kind='bar', stacked=True, ax=ax, 
                            color=['#ff7f0e', '#1f77b4'], 
                            alpha=0.7, edgecolor='black')
        ax.set_title('Class Distribution by Split (Percentage)', 
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('Split', fontsize=12)
        ax.set_ylabel('Percentage (%)', fontsize=12)
        ax.legend(title='Class', fontsize=10)
        ax.grid(axis='y', alpha=0.3)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=0)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / '1_class_distribution.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   Saved: {self.output_dir / '1_class_distribution.png'}")
    
    def extract_features_from_audio(self, file_path, n_samples=None):
        """Extract various audio features from a file"""
        try:
            # Load audio
            audio, sr = librosa.load(file_path, sr=self.sample_rate, 
                                    duration=self.duration)
            
            # Extract features
            features = {}
            
            # 1. Mel spectrogram
            mel_spec = librosa.feature.melspectrogram(
                y=audio, sr=sr, n_mels=self.n_mels, 
                hop_length=self.hop_length, n_fft=self.n_fft
            )
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            features['mel_spec'] = mel_spec_db
            features['mel_spec_mean'] = np.mean(mel_spec_db, axis=1)
            features['mel_spec_std'] = np.std(mel_spec_db, axis=1)
            
            # 2. MFCCs
            mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=20)
            features['mfccs'] = mfccs
            features['mfcc_mean'] = np.mean(mfccs, axis=1)
            features['mfcc_std'] = np.std(mfccs, axis=1)
            
            # 3. Spectral features
            features['spectral_centroid'] = np.mean(
                librosa.feature.spectral_centroid(y=audio, sr=sr)
            )
            features['spectral_bandwidth'] = np.mean(
                librosa.feature.spectral_bandwidth(y=audio, sr=sr)
            )
            features['spectral_rolloff'] = np.mean(
                librosa.feature.spectral_rolloff(y=audio, sr=sr)
            )
            
            # 4. Zero crossing rate
            features['zero_crossing_rate'] = np.mean(
                librosa.feature.zero_crossing_rate(audio)
            )
            
            # 5. RMS energy
            features['rms_energy'] = np.mean(
                librosa.feature.rms(y=audio)
            )
            
            # 6. Chroma features
            chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
            features['chroma_mean'] = np.mean(chroma, axis=1)
            
            # 7. Tempo
            tempo, _ = librosa.beat.beat_track(y=audio, sr=sr)
            features['tempo'] = float(tempo) if not isinstance(tempo, (list, np.ndarray)) else float(tempo[0]) if len(tempo) > 0 else 0.0
            
            # Store raw audio for waveform plotting
            features['audio'] = audio
            features['duration'] = len(audio) / sr
            
            return features
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            return None
    
    def extract_features_for_analysis(self, n_samples_per_class=50):
        """Extract features from a subset of samples for analysis"""
        print(f"\n2. Extracting features from {n_samples_per_class} samples per class...")
        
        features_list = []
        labels = []
        label_names = []
        
        # Sample from each class
        for label_name in self.train_df['label_name'].unique():
            class_samples = self.train_df[
                self.train_df['label_name'] == label_name
            ].sample(n=min(n_samples_per_class, 
                          len(self.train_df[self.train_df['label_name'] == label_name])))
            
            for _, row in tqdm(class_samples.iterrows(), 
                             desc=f"Processing {label_name}", 
                             total=len(class_samples)):
                file_path = row['file_path']
                if not os.path.exists(file_path):
                    file_path = os.path.join('c:\\Users\\dhira\\Desktop\\ISTP', file_path)
                
                if os.path.exists(file_path):
                    features = self.extract_features_from_audio(file_path)
                    if features is not None:
                        # Create feature vector
                        feature_vector = np.concatenate([
                            features['mel_spec_mean'],
                            features['mel_spec_std'],
                            features['mfcc_mean'],
                            features['mfcc_std'],
                            features['chroma_mean'],
                            [features['spectral_centroid']],
                            [features['spectral_bandwidth']],
                            [features['spectral_rolloff']],
                            [features['zero_crossing_rate']],
                            [features['rms_energy']],
                            [features['tempo']]
                        ])
                        
                        features_list.append(feature_vector)
                        labels.append(row['label'])
                        label_names.append(row['label_name'])
        
        self.features_array = np.array(features_list)
        self.labels_array = np.array(labels)
        self.label_names_array = np.array(label_names)
        
        print(f"   Extracted features shape: {self.features_array.shape}")
        
        return self.features_array, self.labels_array, self.label_names_array
    
    def plot_pca_analysis(self):
        """Perform and visualize PCA analysis"""
        print("\n3. Performing PCA analysis...")
        
        # Standardize features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(self.features_array)
        
        # Apply PCA
        pca = PCA()
        pca_features = pca.fit_transform(features_scaled)
        
        # Create figure with multiple subplots
        fig = plt.figure(figsize=(18, 12))
        
        # 1. Explained variance ratio
        ax1 = plt.subplot(2, 3, 1)
        plt.plot(range(1, len(pca.explained_variance_ratio_) + 1),
                np.cumsum(pca.explained_variance_ratio_), 'bo-', linewidth=2)
        plt.xlabel('Number of Components', fontsize=12)
        plt.ylabel('Cumulative Explained Variance', fontsize=12)
        plt.title('PCA: Cumulative Explained Variance', 
                 fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.axhline(y=0.95, color='r', linestyle='--', 
                   label='95% variance', linewidth=2)
        plt.axhline(y=0.99, color='g', linestyle='--', 
                   label='99% variance', linewidth=2)
        plt.legend()
        
        # 2. Scree plot
        ax2 = plt.subplot(2, 3, 2)
        plt.bar(range(1, min(21, len(pca.explained_variance_ratio_) + 1)),
               pca.explained_variance_ratio_[:20], 
               alpha=0.7, color='steelblue', edgecolor='black')
        plt.xlabel('Principal Component', fontsize=12)
        plt.ylabel('Explained Variance Ratio', fontsize=12)
        plt.title('PCA: Scree Plot (Top 20 Components)', 
                 fontsize=14, fontweight='bold')
        plt.grid(axis='y', alpha=0.3)
        
        # 3. 2D PCA projection
        ax3 = plt.subplot(2, 3, 3)
        colors = ['#ff7f0e' if label == 1 else '#1f77b4' 
                 for label in self.labels_array]
        scatter = plt.scatter(pca_features[:, 0], pca_features[:, 1], 
                            c=colors, alpha=0.6, s=50, edgecolors='black', 
                            linewidth=0.5)
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)', 
                  fontsize=12)
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)', 
                  fontsize=12)
        plt.title('PCA: 2D Projection (PC1 vs PC2)', 
                 fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='#ff7f0e', label='Chainsaw'),
                          Patch(facecolor='#1f77b4', label='Non-Chainsaw')]
        plt.legend(handles=legend_elements, loc='best')
        
        # 4. 3D PCA projection (from different angle)
        from mpl_toolkits.mplot3d import Axes3D
        ax4 = plt.subplot(2, 3, 4, projection='3d')
        ax4.scatter(pca_features[:, 0], pca_features[:, 1], pca_features[:, 2],
                   c=colors, alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
        ax4.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})', fontsize=10)
        ax4.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})', fontsize=10)
        ax4.set_zlabel(f'PC3 ({pca.explained_variance_ratio_[2]:.2%})', fontsize=10)
        ax4.set_title('PCA: 3D Projection', fontsize=14, fontweight='bold')
        
        # 5. PC2 vs PC3
        ax5 = plt.subplot(2, 3, 5)
        plt.scatter(pca_features[:, 1], pca_features[:, 2], 
                   c=colors, alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
        plt.xlabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)', 
                  fontsize=12)
        plt.ylabel(f'PC3 ({pca.explained_variance_ratio_[2]:.2%} variance)', 
                  fontsize=12)
        plt.title('PCA: 2D Projection (PC2 vs PC3)', 
                 fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.legend(handles=legend_elements, loc='best')
        
        # 6. Component importance table
        ax6 = plt.subplot(2, 3, 6)
        ax6.axis('off')
        
        # Create table data
        n_components = 10
        table_data = []
        for i in range(n_components):
            table_data.append([
                f'PC{i+1}',
                f'{pca.explained_variance_ratio_[i]:.4f}',
                f'{np.cumsum(pca.explained_variance_ratio_)[i]:.4f}'
            ])
        
        table = ax6.table(cellText=table_data,
                         colLabels=['Component', 'Variance', 'Cumulative'],
                         cellLoc='center',
                         loc='center',
                         colWidths=[0.3, 0.35, 0.35])
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)
        
        # Style header
        for i in range(3):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        ax6.set_title('Top 10 Principal Components', 
                     fontsize=14, fontweight='bold', pad=20)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / '2_pca_analysis.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   Saved: {self.output_dir / '2_pca_analysis.png'}")
        
        # Save PCA results
        print(f"   Top 5 components explain {np.cumsum(pca.explained_variance_ratio_)[4]:.2%} of variance")
    
    def plot_tsne_analysis(self):
        """Perform and visualize t-SNE analysis"""
        print("\n4. Performing t-SNE analysis...")
        
        # Standardize features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(self.features_array)
        
        # Reduce dimensionality with PCA first (speed up t-SNE)
        pca = PCA(n_components=50)
        features_pca = pca.fit_transform(features_scaled)
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 14))
        
        # Adjust perplexities based on sample size
        n_samples = features_pca.shape[0]
        max_perplexity = min(50, n_samples // 3)
        perplexities = [5, 15, 30, max_perplexity]
        
        for idx, perplexity in enumerate(perplexities):
            ax = axes[idx // 2, idx % 2]
            
            print(f"   Computing t-SNE with perplexity={perplexity}...")
            tsne = TSNE(n_components=2, perplexity=perplexity, 
                       random_state=42, n_iter=1000)
            tsne_features = tsne.fit_transform(features_pca)
            
            colors = ['#ff7f0e' if label == 1 else '#1f77b4' 
                     for label in self.labels_array]
            
            ax.scatter(tsne_features[:, 0], tsne_features[:, 1],
                      c=colors, alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
            ax.set_xlabel('t-SNE Component 1', fontsize=12)
            ax.set_ylabel('t-SNE Component 2', fontsize=12)
            ax.set_title(f't-SNE Visualization (perplexity={perplexity})', 
                        fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            
            # Add legend
            from matplotlib.patches import Patch
            legend_elements = [Patch(facecolor='#ff7f0e', label='Chainsaw'),
                              Patch(facecolor='#1f77b4', label='Non-Chainsaw')]
            ax.legend(handles=legend_elements, loc='best')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / '3_tsne_analysis.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   Saved: {self.output_dir / '3_tsne_analysis.png'}")
        
        # Create 3D t-SNE visualization
        print("\n   Creating 3D t-SNE visualization...")
        from mpl_toolkits.mplot3d import Axes3D
        
        fig = plt.figure(figsize=(18, 6))
        
        # Try multiple perplexities for 3D
        perplexities_3d = [15, 30, max_perplexity]
        colors = ['#ff7f0e' if label == 1 else '#1f77b4' 
                 for label in self.labels_array]
        
        for idx, perplexity in enumerate(perplexities_3d):
            print(f"   Computing 3D t-SNE with perplexity={perplexity}...")
            tsne_3d = TSNE(n_components=3, perplexity=perplexity, 
                          random_state=42, n_iter=1000)
            tsne_features_3d = tsne_3d.fit_transform(features_pca)
            
            ax = fig.add_subplot(1, 3, idx + 1, projection='3d')
            
            # Plot points
            scatter = ax.scatter(tsne_features_3d[:, 0], 
                                tsne_features_3d[:, 1], 
                                tsne_features_3d[:, 2],
                                c=colors, alpha=0.6, s=50, 
                                edgecolors='black', linewidth=0.5)
            
            ax.set_xlabel('t-SNE Dim 1', fontsize=10)
            ax.set_ylabel('t-SNE Dim 2', fontsize=10)
            ax.set_zlabel('t-SNE Dim 3', fontsize=10)
            ax.set_title(f'3D t-SNE (perplexity={perplexity})', 
                        fontsize=12, fontweight='bold')
            
            # Add legend
            from matplotlib.patches import Patch
            legend_elements = [Patch(facecolor='#ff7f0e', label='Chainsaw'),
                              Patch(facecolor='#1f77b4', label='Non-Chainsaw')]
            ax.legend(handles=legend_elements, loc='upper right', fontsize=9)
            
            # Adjust viewing angle for better visualization
            ax.view_init(elev=25, azim=45)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / '3_tsne_3d_analysis.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   Saved: {self.output_dir / '3_tsne_3d_analysis.png'}")
    
    def plot_feature_distributions(self):
        """Plot distributions of various audio features"""
        print("\n5. Creating feature distribution plots...")
        
        # Extract a subset of samples
        n_samples = 30
        chainsaw_samples = self.train_df[self.train_df['label'] == 1].sample(
            n=min(n_samples, len(self.train_df[self.train_df['label'] == 1]))
        )
        non_chainsaw_samples = self.train_df[self.train_df['label'] == 0].sample(
            n=min(n_samples, len(self.train_df[self.train_df['label'] == 0]))
        )
        
        # Extract features
        chainsaw_features = []
        non_chainsaw_features = []
        
        for _, row in tqdm(chainsaw_samples.iterrows(), 
                          desc="Extracting chainsaw features", 
                          total=len(chainsaw_samples)):
            file_path = row['file_path']
            if not os.path.exists(file_path):
                file_path = os.path.join('c:\\Users\\dhira\\Desktop\\ISTP', file_path)
            if os.path.exists(file_path):
                features = self.extract_features_from_audio(file_path)
                if features is not None:
                    chainsaw_features.append(features)
        
        for _, row in tqdm(non_chainsaw_samples.iterrows(), 
                          desc="Extracting non-chainsaw features", 
                          total=len(non_chainsaw_samples)):
            file_path = row['file_path']
            if not os.path.exists(file_path):
                file_path = os.path.join('c:\\Users\\dhira\\Desktop\\ISTP', file_path)
            if os.path.exists(file_path):
                features = self.extract_features_from_audio(file_path)
                if features is not None:
                    non_chainsaw_features.append(features)
        
        # Create plots
        fig, axes = plt.subplots(3, 3, figsize=(18, 14))
        
        feature_names = [
            ('spectral_centroid', 'Spectral Centroid'),
            ('spectral_bandwidth', 'Spectral Bandwidth'),
            ('spectral_rolloff', 'Spectral Rolloff'),
            ('zero_crossing_rate', 'Zero Crossing Rate'),
            ('rms_energy', 'RMS Energy'),
            ('tempo', 'Tempo (BPM)'),
        ]
        
        for idx, (feature_key, feature_name) in enumerate(feature_names):
            ax = axes[idx // 3, idx % 3]
            
            chainsaw_values = [f[feature_key] for f in chainsaw_features]
            non_chainsaw_values = [f[feature_key] for f in non_chainsaw_features]
            
            ax.hist(non_chainsaw_values, bins=20, alpha=0.6, 
                   label='Non-Chainsaw', color='#1f77b4', edgecolor='black')
            ax.hist(chainsaw_values, bins=20, alpha=0.6, 
                   label='Chainsaw', color='#ff7f0e', edgecolor='black')
            
            ax.set_xlabel(feature_name, fontsize=11)
            ax.set_ylabel('Frequency', fontsize=11)
            ax.set_title(f'{feature_name} Distribution', 
                        fontsize=12, fontweight='bold')
            ax.legend()
            ax.grid(axis='y', alpha=0.3)
        
        # MFCC comparison
        ax = axes[2, 0]
        chainsaw_mfccs = np.mean([f['mfcc_mean'] for f in chainsaw_features], axis=0)
        non_chainsaw_mfccs = np.mean([f['mfcc_mean'] for f in non_chainsaw_features], axis=0)
        
        x = np.arange(len(chainsaw_mfccs))
        width = 0.35
        ax.bar(x - width/2, non_chainsaw_mfccs, width, 
              label='Non-Chainsaw', alpha=0.7, color='#1f77b4', edgecolor='black')
        ax.bar(x + width/2, chainsaw_mfccs, width, 
              label='Chainsaw', alpha=0.7, color='#ff7f0e', edgecolor='black')
        ax.set_xlabel('MFCC Coefficient', fontsize=11)
        ax.set_ylabel('Mean Value', fontsize=11)
        ax.set_title('Mean MFCC Coefficients Comparison', 
                    fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        # Chroma features
        ax = axes[2, 1]
        chainsaw_chroma = np.mean([f['chroma_mean'] for f in chainsaw_features], axis=0)
        non_chainsaw_chroma = np.mean([f['chroma_mean'] for f in non_chainsaw_features], axis=0)
        
        x = np.arange(len(chainsaw_chroma))
        ax.bar(x - width/2, non_chainsaw_chroma, width, 
              label='Non-Chainsaw', alpha=0.7, color='#1f77b4', edgecolor='black')
        ax.bar(x + width/2, chainsaw_chroma, width, 
              label='Chainsaw', alpha=0.7, color='#ff7f0e', edgecolor='black')
        ax.set_xlabel('Chroma Bin', fontsize=11)
        ax.set_ylabel('Mean Value', fontsize=11)
        ax.set_title('Mean Chroma Features Comparison', 
                    fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        # Summary statistics
        ax = axes[2, 2]
        ax.axis('off')
        
        summary_text = f"""
        Feature Statistics Summary
        
        Samples Analyzed:
        • Chainsaw: {len(chainsaw_features)}
        • Non-Chainsaw: {len(non_chainsaw_features)}
        
        Mean Spectral Centroid:
        • Chainsaw: {np.mean(chainsaw_values):.2f}
        • Non-Chainsaw: {np.mean(non_chainsaw_values):.2f}
        
        Mean RMS Energy:
        • Chainsaw: {np.mean([f['rms_energy'] for f in chainsaw_features]):.4f}
        • Non-Chainsaw: {np.mean([f['rms_energy'] for f in non_chainsaw_features]):.4f}
        
        Mean Tempo:
        • Chainsaw: {np.mean([f['tempo'] for f in chainsaw_features]):.2f} BPM
        • Non-Chainsaw: {np.mean([f['tempo'] for f in non_chainsaw_features]):.2f} BPM
        """
        
        ax.text(0.1, 0.5, summary_text, fontsize=10, 
               verticalalignment='center', family='monospace',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
        plt.tight_layout()
        plt.savefig(self.output_dir / '4_feature_distributions.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   Saved: {self.output_dir / '4_feature_distributions.png'}")
        
        return chainsaw_features, non_chainsaw_features
    
    def plot_spectrograms(self, chainsaw_features, non_chainsaw_features):
        """Plot spectrograms for sample audio files"""
        print("\n6. Creating spectrogram comparisons...")
        
        fig, axes = plt.subplots(4, 4, figsize=(20, 16))
        
        # Plot chainsaw spectrograms
        for i in range(4):
            if i < len(chainsaw_features):
                # Mel spectrogram
                ax = axes[i, 0]
                mel_spec = chainsaw_features[i]['mel_spec']
                img = librosa.display.specshow(mel_spec, x_axis='time', y_axis='mel',
                                              sr=self.sample_rate, 
                                              hop_length=self.hop_length,
                                              ax=ax, cmap='viridis')
                ax.set_title(f'Chainsaw Mel Spectrogram {i+1}', fontweight='bold')
                ax.set_xlabel('Time (s)')
                ax.set_ylabel('Mel Frequency')
                plt.colorbar(img, ax=ax, format='%+2.0f dB')
                
                # Waveform
                ax = axes[i, 1]
                audio = chainsaw_features[i]['audio']
                time = np.linspace(0, len(audio) / self.sample_rate, len(audio))
                ax.plot(time, audio, color='#ff7f0e', linewidth=0.5)
                ax.set_title(f'Chainsaw Waveform {i+1}', fontweight='bold')
                ax.set_xlabel('Time (s)')
                ax.set_ylabel('Amplitude')
                ax.grid(True, alpha=0.3)
        
        # Plot non-chainsaw spectrograms
        for i in range(4):
            if i < len(non_chainsaw_features):
                # Mel spectrogram
                ax = axes[i, 2]
                mel_spec = non_chainsaw_features[i]['mel_spec']
                img = librosa.display.specshow(mel_spec, x_axis='time', y_axis='mel',
                                              sr=self.sample_rate, 
                                              hop_length=self.hop_length,
                                              ax=ax, cmap='viridis')
                ax.set_title(f'Non-Chainsaw Mel Spectrogram {i+1}', fontweight='bold')
                ax.set_xlabel('Time (s)')
                ax.set_ylabel('Mel Frequency')
                plt.colorbar(img, ax=ax, format='%+2.0f dB')
                
                # Waveform
                ax = axes[i, 3]
                audio = non_chainsaw_features[i]['audio']
                time = np.linspace(0, len(audio) / self.sample_rate, len(audio))
                ax.plot(time, audio, color='#1f77b4', linewidth=0.5)
                ax.set_title(f'Non-Chainsaw Waveform {i+1}', fontweight='bold')
                ax.set_xlabel('Time (s)')
                ax.set_ylabel('Amplitude')
                ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / '5_spectrograms_comparison.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   Saved: {self.output_dir / '5_spectrograms_comparison.png'}")
    
    def plot_correlation_matrix(self):
        """Plot correlation matrix of features"""
        print("\n7. Creating correlation matrix...")
        
        # Create feature names
        feature_names = (
            [f'MelMean_{i}' for i in range(self.n_mels)] +
            [f'MelStd_{i}' for i in range(self.n_mels)] +
            [f'MFCC_Mean_{i}' for i in range(20)] +
            [f'MFCC_Std_{i}' for i in range(20)] +
            [f'Chroma_{i}' for i in range(12)] +
            ['SpectralCentroid', 'SpectralBandwidth', 'SpectralRolloff',
             'ZeroCrossingRate', 'RMSEnergy', 'Tempo']
        )
        
        # Compute correlation matrix
        corr_matrix = np.corrcoef(self.features_array.T)
        
        # Plot full correlation matrix (may be large)
        fig, ax = plt.subplots(figsize=(20, 18))
        im = ax.imshow(corr_matrix, cmap='coolwarm', aspect='auto', 
                      vmin=-1, vmax=1)
        
        plt.colorbar(im, ax=ax, label='Correlation Coefficient')
        ax.set_title('Feature Correlation Matrix', 
                    fontsize=16, fontweight='bold', pad=20)
        
        # Only show a subset of tick labels to avoid clutter
        tick_spacing = max(1, len(feature_names) // 20)
        ax.set_xticks(np.arange(0, len(feature_names), tick_spacing))
        ax.set_yticks(np.arange(0, len(feature_names), tick_spacing))
        ax.set_xticklabels([feature_names[i] for i in range(0, len(feature_names), tick_spacing)],
                          rotation=90, fontsize=8)
        ax.set_yticklabels([feature_names[i] for i in range(0, len(feature_names), tick_spacing)],
                          fontsize=8)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / '6_correlation_matrix_full.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot correlation of key features only
        key_feature_indices = list(range(self.n_mels * 2, self.n_mels * 2 + 20)) + \
                             list(range(-6, 0))  # MFCCs and spectral features
        key_features = self.features_array[:, key_feature_indices]
        key_feature_names = ([f'MFCC_Mean_{i}' for i in range(20)] +
                            ['SpectralCentroid', 'SpectralBandwidth', 
                             'SpectralRolloff', 'ZeroCrossingRate', 
                             'RMSEnergy', 'Tempo'])
        
        corr_matrix_key = np.corrcoef(key_features.T)
        
        fig, ax = plt.subplots(figsize=(14, 12))
        im = ax.imshow(corr_matrix_key, cmap='coolwarm', aspect='auto', 
                      vmin=-1, vmax=1)
        
        plt.colorbar(im, ax=ax, label='Correlation Coefficient')
        ax.set_title('Key Features Correlation Matrix', 
                    fontsize=16, fontweight='bold', pad=20)
        
        ax.set_xticks(np.arange(len(key_feature_names)))
        ax.set_yticks(np.arange(len(key_feature_names)))
        ax.set_xticklabels(key_feature_names, rotation=90, fontsize=9)
        ax.set_yticklabels(key_feature_names, fontsize=9)
        
        # Add correlation values for key features
        for i in range(len(key_feature_names)):
            for j in range(len(key_feature_names)):
                if abs(corr_matrix_key[i, j]) > 0.5 and i != j:
                    text = ax.text(j, i, f'{corr_matrix_key[i, j]:.2f}',
                                 ha="center", va="center", color="black", 
                                 fontsize=7, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / '6_correlation_matrix_key.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   Saved: {self.output_dir / '6_correlation_matrix_full.png'}")
        print(f"   Saved: {self.output_dir / '6_correlation_matrix_key.png'}")
    
    def plot_statistical_summary(self):
        """Create statistical summary plots"""
        print("\n8. Creating statistical summary...")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Box plot for class balance
        ax = axes[0, 0]
        split_counts = pd.DataFrame({
            'split': ['Train', 'Val', 'Test'],
            'chainsaw': [
                len(self.train_df[self.train_df['label'] == 1]),
                len(self.val_df[self.val_df['label'] == 1]),
                len(self.test_df[self.test_df['label'] == 1])
            ],
            'non_chainsaw': [
                len(self.train_df[self.train_df['label'] == 0]),
                len(self.val_df[self.val_df['label'] == 0]),
                len(self.test_df[self.test_df['label'] == 0])
            ]
        })
        
        x = np.arange(len(split_counts))
        width = 0.35
        ax.bar(x - width/2, split_counts['non_chainsaw'], width, 
              label='Non-Chainsaw', alpha=0.7, color='#1f77b4', edgecolor='black')
        ax.bar(x + width/2, split_counts['chainsaw'], width, 
              label='Chainsaw', alpha=0.7, color='#ff7f0e', edgecolor='black')
        
        ax.set_xlabel('Split', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.set_title('Sample Distribution Across Splits', 
                    fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(split_counts['split'])
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        # Pie chart for overall distribution
        ax = axes[0, 1]
        overall_counts = self.all_df['label_name'].value_counts()
        colors_pie = ['#ff7f0e', '#1f77b4']
        explode = (0.05, 0.05)
        
        wedges, texts, autotexts = ax.pie(overall_counts.values, 
                                          labels=overall_counts.index,
                                          autopct='%1.1f%%',
                                          colors=colors_pie,
                                          explode=explode,
                                          shadow=True,
                                          startangle=90,
                                          textprops={'fontsize': 12, 
                                                    'fontweight': 'bold'})
        ax.set_title('Overall Class Distribution', 
                    fontsize=14, fontweight='bold')
        
        # Dataset statistics table
        ax = axes[1, 0]
        ax.axis('off')
        
        stats_data = [
            ['Metric', 'Train', 'Val', 'Test', 'Total'],
            ['Total Samples', 
             str(len(self.train_df)), 
             str(len(self.val_df)), 
             str(len(self.test_df)),
             str(len(self.all_df))],
            ['Chainsaw Samples',
             str(len(self.train_df[self.train_df['label'] == 1])),
             str(len(self.val_df[self.val_df['label'] == 1])),
             str(len(self.test_df[self.test_df['label'] == 1])),
             str(len(self.all_df[self.all_df['label'] == 1]))],
            ['Non-Chainsaw Samples',
             str(len(self.train_df[self.train_df['label'] == 0])),
             str(len(self.val_df[self.val_df['label'] == 0])),
             str(len(self.test_df[self.test_df['label'] == 0])),
             str(len(self.all_df[self.all_df['label'] == 0]))],
            ['Class Balance (%)',
             f"{len(self.train_df[self.train_df['label'] == 1]) / len(self.train_df) * 100:.1f}",
             f"{len(self.val_df[self.val_df['label'] == 1]) / len(self.val_df) * 100:.1f}",
             f"{len(self.test_df[self.test_df['label'] == 1]) / len(self.test_df) * 100:.1f}",
             f"{len(self.all_df[self.all_df['label'] == 1]) / len(self.all_df) * 100:.1f}"]
        ]
        
        table = ax.table(cellText=stats_data, cellLoc='center',
                        loc='center', colWidths=[0.25, 0.15, 0.15, 0.15, 0.15])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2.5)
        
        # Style header row
        for i in range(5):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Style first column
        for i in range(1, 5):
            table[(i, 0)].set_facecolor('#E8F5E9')
            table[(i, 0)].set_text_props(weight='bold')
        
        ax.set_title('Dataset Statistics Summary', 
                    fontsize=14, fontweight='bold', pad=20)
        
        # Feature dimensionality info
        ax = axes[1, 1]
        ax.axis('off')
        
        info_text = f"""
        DATASET INFORMATION
        
        Audio Configuration:
        • Sample Rate: {self.sample_rate} Hz
        • Duration: {self.duration} seconds
        • Mel Bins: {self.n_mels}
        • N-FFT: {self.n_fft}
        • Hop Length: {self.hop_length}
        
        Feature Dimensions:
        • Total Features Extracted: {self.features_array.shape[1]}
        • Samples Analyzed: {self.features_array.shape[0]}
        
        Classes:
        • Chainsaw (Label 1)
        • Non-Chainsaw (Label 0)
        
        Analysis Output:
        • Location: {self.output_dir}/
        • Total Plots: 8 visualizations
        """
        
        ax.text(0.1, 0.5, info_text, fontsize=11,
               verticalalignment='center', family='monospace',
               bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
        
        plt.tight_layout()
        plt.savefig(self.output_dir / '7_statistical_summary.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   Saved: {self.output_dir / '7_statistical_summary.png'}")
    
    def run_complete_analysis(self):
        """Run all analysis steps"""
        print("="*70)
        print("COMPREHENSIVE DATASET ANALYSIS")
        print("="*70)
        
        # Load datasets
        self.load_datasets()
        
        # Generate plots
        self.plot_class_distribution()
        
        # Extract features for dimensionality reduction
        self.extract_features_for_analysis(n_samples_per_class=50)
        
        # Dimensionality reduction
        self.plot_pca_analysis()
        self.plot_tsne_analysis()
        
        # Feature analysis
        chainsaw_feats, non_chainsaw_feats = self.plot_feature_distributions()
        
        # Spectrograms
        self.plot_spectrograms(chainsaw_feats, non_chainsaw_feats)
        
        # Correlation analysis
        self.plot_correlation_matrix()
        
        # Statistical summary
        self.plot_statistical_summary()
        
        print("\n" + "="*70)
        print("ANALYSIS COMPLETE!")
        print("="*70)
        print(f"\nAll plots saved to: {self.output_dir.absolute()}")
        print("\nGenerated visualizations:")
        print("  1. Class distribution bar charts")
        print("  2. PCA analysis (2D, 3D, scree plots)")
        print("  3. t-SNE visualizations (2D - multiple perplexities)")
        print("  4. t-SNE 3D visualizations (3D - multiple perplexities)")
        print("  5. Feature distributions and comparisons")
        print("  6. Spectrograms and waveforms")
        print("  7. Correlation matrices")
        print("  8. Statistical summary")


if __name__ == "__main__":
    analyzer = DatasetAnalyzer()
    analyzer.run_complete_analysis()
