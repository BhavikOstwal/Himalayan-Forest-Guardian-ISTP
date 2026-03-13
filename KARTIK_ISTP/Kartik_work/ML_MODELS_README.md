# Lightweight ML Models for Chainsaw Detection

This directory contains trained lightweight machine learning models for binary classification of chainsaw sounds vs non-chainsaw sounds.

## Model Performance

### Best Model: Logistic Regression (Balanced)
- **Test F1 Score**: 0.3229
- **Test Accuracy**: 64.58%
- **Balanced Accuracy**: 57.70%
- **Recall (Sensitivity)**: 46.97% - Catches ~47% of chainsaw sounds
- **Precision**: 24.60%
- **Specificity**: 68.44% - Correctly identifies ~68% of non-chainsaw sounds

### Model Comparison
All models were trained with class imbalance handling using SMOTE:

| Model | F1 Score | Accuracy | Balanced Acc | Recall |
|-------|----------|----------|--------------|--------|
| **Logistic Regression** | **0.323** | **64.6%** | **57.7%** | **47.0%** |
| SVM (RBF) | 0.180 | 72.8% | 50.9% | 16.7% |
| Gradient Boosting | 0.168 | 73.0% | 50.4% | 15.2% |
| Random Forest | 0.136 | 72.2% | 48.8% | 12.1% |

## Features

The model uses **91 compact audio features** extracted from 10-second audio clips:

### Feature Categories:
1. **MFCCs (52 features)**: Mean, std, max, min of 13 MFCC coefficients
2. **Spectral Features (13 features)**: Centroid, bandwidth, rolloff, contrast
3. **Zero Crossing Rate (2 features)**: Mean and std
4. **RMS Energy (4 features)**: Mean, std, max, min
5. **Chroma Features (12 features)**: 12 pitch classes
6. **Mel Spectrogram Stats (4 features)**: Mean, std, max, min
7. **Tempo (1 feature)**: BPM
8. **Time Domain (3 features)**: Mean amplitude, std, peak

**Total: 91 features** - Lightweight and efficient for real-time processing

## Files in this Directory

```
ml_balanced_20260303_002150/
├── best_model.pkl              # Best performing model (Logistic Regression)
├── scaler.pkl                  # Feature scaler (StandardScaler)
├── results.json                # All model metrics
├── classification_report.txt   # Detailed performance report
├── model_comparison.png        # Visual comparison of all models
├── detailed_performance.png    # Detailed performance breakdown
└── all_models/                 # All trained models
    ├── svm_balanced.pkl
    ├── random_forest_balanced.pkl
    ├── gradient_boosting_balanced.pkl
    ├── logistic_regression_balanced.pkl
    └── random_forest_optimized.pkl
```

## Usage

### 1. Single File Prediction

```bash
python inference_ml.py \
    --model_dir output/ml_balanced_20260303_002150 \
    --audio_file path/to/audio.wav
```

**Output:**
```
Prediction: CHAINSAW
Confidence: 75.32%
```

### 2. Batch Prediction

```bash
python inference_ml.py \
    --model_dir output/ml_balanced_20260303_002150 \
    --audio_dir path/to/audio/folder \
    --output predictions.csv
```

### 3. Python API

```python
from inference_ml import ChainsawDetector

# Initialize detector
detector = ChainsawDetector('output/ml_balanced_20260303_002150')

# Single prediction
prediction = detector.predict('audio.wav')
probability = detector.predict('audio.wav', return_proba=True)

print(f"Prediction: {prediction}")  # 0 or 1
print(f"Chainsaw probability: {probability:.2%}")

# Batch predictions
audio_files = ['file1.wav', 'file2.wav', 'file3.wav']
results = detector.predict_batch(audio_files)

for filename, pred, proba, label in results:
    print(f"{filename}: {label} ({proba:.2%})")
```

## Training Details

### Dataset
- **Training samples**: 1,707 (82% non-chainsaw, 18% chainsaw)
- **Validation samples**: 366
- **Test samples**: 367
- **Audio**: 16kHz, 10 seconds duration

### Class Imbalance Handling
- **SMOTE (Synthetic Minority Over-sampling Technique)** applied to balance training data
- Balanced training set: 2,782 samples (50% each class)
- Class weights used in model training
- **SMOTETomek** for cleaning oversampled data

### Why Logistic Regression Performed Best?
1. **Better generalization**: Simple model avoids overfitting on small dataset
2. **Class imbalance handling**: Works well with balanced class weights
3. **Probabilistic output**: Provides meaningful confidence scores
4. **Computational efficiency**: Fast training and inference

## Model Limitations

1. **Class Imbalance**: Original dataset heavily skewed (82% non-chainsaw)
2. **Small chainsaw samples**: Only 66 chainsaw samples in test set
3. **Moderate F1 score**: 0.323 indicates room for improvement
4. **Precision-Recall tradeoff**: Higher recall (47%) with lower precision (25%)

## Recommendations for Production Use

### If False Negatives are Critical (Missing chainsaws is bad):
- Use **Logistic Regression** (current best model)
- Accept higher false positive rate
- Recall: 47% of chainsaws detected

### If False Positives are Critical (False alarms are bad):
- Use **Random Forest Optimized**
- Lower false alarms, but misses more chainsaws
- Specificity: 86% of non-chainsaws correctly identified

### Threshold Tuning
You can adjust the decision threshold based on your needs:

```python
# Get probability
proba = detector.predict('audio.wav', return_proba=True)

# Custom threshold
threshold = 0.3  # Lower = more sensitive to chainsaw
prediction = 1 if proba > threshold else 0
```

## Retraining the Model

To retrain with new data:

```bash
# Train with SMOTE balancing
python train_ml_balanced.py
```

This will:
1. Load cached features from `data/features_cache/`
2. Apply SMOTE for class balancing
3. Train 5+ models (SVM, RF, GB, LR)
4. Evaluate on validation and test sets
5. Save best model and results to `output/ml_balanced_<timestamp>/`

## Performance Visualizations

### 1. Model Comparison (`model_comparison.png`)
- Multi-metric comparison (accuracy, precision, recall, F1, AUC)
- F1 score ranking
- Confusion matrix for best model
- ROC curves for all models

### 2. Detailed Performance (`detailed_performance.png`)
- Sensitivity vs Specificity trade-off
- Precision vs Recall analysis
- Balanced accuracy comparison
- Summary table with key metrics

## Technical Requirements

```
Python >= 3.8
librosa >= 0.10.0
scikit-learn >= 1.0.0
imbalanced-learn >= 0.9.0
numpy >= 1.20.0
pandas >= 1.3.0
matplotlib >= 3.4.0
seaborn >= 0.11.0
```

## Future Improvements

1. **Collect more chainsaw samples**: Balance dataset naturally
2. **Feature engineering**: Add domain-specific audio features
3. **Ensemble methods**: Combine multiple models
4. **Time-series features**: Analyze temporal patterns
5. **Transfer learning**: Use pre-trained audio embeddings
6. **Data augmentation**: Time-stretching, pitch-shifting for training

## Contact & Support

For questions or issues, please refer to the main project README.

---

**Model trained on**: March 3, 2026
**Framework**: scikit-learn + imbalanced-learn
**Model size**: ~1.5 KB (best_model.pkl)
**Inference time**: ~0.5 seconds per 10-second audio clip
