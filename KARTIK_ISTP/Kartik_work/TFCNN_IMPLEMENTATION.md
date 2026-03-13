# TFCNN Implementation for Chainsaw Detection

## Overview
This implementation replicates the **Temporal Frequency Convolutional Neural Network (TFCNN)** architecture from the research paper:

**"A Chainsaw-Sound Recognition Model for Detecting Illegal Logging Activities in Forests"**  
*Daniel Simiyu, Allan Vikiru, Henry Muchiri, Fengshou Gu, and Julius Butime*

## Key Features of TFCNN

### 1. **Temporal-Frequency Decomposition**
Unlike traditional CNNs that treat spectrograms uniformly, TFCNN processes temporal and frequency information separately before combining them:

- **Temporal Convolution**: Captures patterns along the time axis (chainsaw operation sequences)
- **Frequency Convolution**: Captures patterns along the frequency axis (harmonic structures)
- **Combined Features**: Concatenates both to leverage complementary information

### 2. **Attention Mechanism**
The model includes channel-wise attention modules that:
- Focus on important features while suppressing irrelevant noise
- Use both average and max pooling for comprehensive feature extraction
- Apply learned weights to emphasize discriminative patterns

### 3. **Multi-Scale Feature Representation**
- Three TF-CNN blocks progressively extract features (64 → 128 → 256 channels)
- Each block includes batch normalization and max pooling for robust learning
- Global average and max pooling combine spatial information

### 4. **Enhanced Feature Representation Module**
The final classification layers include:
- Two fully-connected layers (512 → 256 neurons) for feature refinement
- Dropout regularization (50% and 30%) to prevent overfitting
- Designed to distinguish chainsaw sounds from environmental sounds

## Architecture Details

```
Input: Mel-Spectrogram [1 × 128 × Time]
    ↓
Initial Conv2D [32 channels]
    ↓
TF-Block 1 [64 channels]
    ├── Temporal Conv (1×7 kernel)
    ├── Frequency Conv (7×1 kernel)
    ├── Combine & Attention
    └── MaxPool 2×2
    ↓
TF-Block 2 [128 channels]
    ├── Temporal Conv (1×7 kernel)
    ├── Frequency Conv (7×1 kernel)
    ├── Combine & Attention
    └── MaxPool 2×2
    ↓
TF-Block 3 [256 channels]
    ├── Temporal Conv (1×7 kernel)
    ├── Frequency Conv (7×1 kernel)
    ├── Combine & Attention
    └── MaxPool 2×2
    ↓
Global Pooling (Avg + Max)
    ↓
FC 512 + Dropout(0.5)
    ↓
FC 256 + Dropout(0.3)
    ↓
FC 2 (Softmax)
```

**Total Parameters**: ~2.57M (4× larger than SimpleCNN but with better feature extraction)

## Key Differences from SimpleCNN

| Feature | SimpleCNN | TFCNN |
|---------|-----------|-------|
| **Architecture** | Standard CNN | Temporal-Frequency decomposition |
| **Attention** | None | Channel-wise attention |
| **Parameters** | 618K | 2.57M |
| **Feature Extraction** | General convolutions | Specialized T/F processing |
| **Robustness** | Basic | Enhanced via attention |

## Training Configuration

```yaml
Audio Processing:
- Sample Rate: 16,000 Hz
- Duration: 10 seconds
- Mel Bands: 128
- Max Frequency: 8,000 Hz (focus on chainsaw range)

Training:
- Optimizer: AdamW (better regularization)
- Learning Rate: 5e-5
- Weight Decay: 1e-4
- Scheduler: ReduceLROnPlateau
- Batch Size: 16
- Early Stopping: 5 epochs patience
- Gradient Clipping: Max norm 1.0
```

## Usage

### Training
```bash
python train_tfcnn.py
```

### Evaluation
```bash
python evaluate_tfcnn.py --model-path output/tfcnn_model_XXXXXX/best_model.pt --split test
```

### Inference

**Single file:**
```bash
python inference_tfcnn.py --model-path output/tfcnn_model_XXXXXX/best_model.pt --audio file.wav
```

**Long audio with sliding window:**
```bash
python inference_tfcnn.py --model-path output/tfcnn_model_XXXXXX/best_model.pt --audio long_recording.wav --long-audio
```

**Batch processing:**
```bash
python inference_tfcnn.py --model-path output/tfcnn_model_XXXXXX/best_model.pt --audio-dir path/to/files --output results.csv
```

## Expected Performance

With the current dataset (40 chainsaw samples):
- **Accuracy**: 50-60%
- **Chainsaw Recall**: 60-75% (better than SimpleCNN)
- **F1-Score**: 30-45%

With expanded dataset (200+ chainsaw samples):
- **Accuracy**: 75-85%
- **Chainsaw Recall**: 80-90%
- **F1-Score**: 70-80%

### Why TFCNN Should Perform Better:

1. **Temporal Patterns**: Chainsaws have distinct temporal patterns (engine cycles, cutting sequences) that temporal convolutions capture effectively

2. **Frequency Structure**: Chainsaw harmonics at specific frequencies are better extracted by frequency-specific convolutions

3. **Attention Focus**: The attention mechanism learns to focus on chainsaw-specific features while ignoring environmental noise

4. **Robustness**: Separate T/F processing makes the model more robust to variations in recording conditions

## Paper Recommendations Implemented

✅ **Temporal Frequency CNN**: Separate temporal and frequency convolutions  
✅ **Attention Mechanism**: Channel-wise attention for feature emphasis  
✅ **Feature Representation Module**: Multi-layer FC network for classification  
✅ **ESC-50 Dataset**: Using ESC-50 for environmental sound diversity  
⚠ **FSC22 Dataset**: Not yet integrated (forest-specific sounds)

## Next Steps for Production Deployment

1. **Data Collection**:
   - Download FSC22 dataset for forest-specific sounds
   - Collect more real-world chainsaw recordings (target: 200-500 samples)
   - Include various chainsaw types and recording conditions

2. **Model Optimization**:
   - Fine-tune attention mechanism parameters
   - Experiment with different TF-block configurations
   - Add spectrogram augmentation (time/frequency masking)

3. **IoT Integration** (as per paper):
   - Deploy on LoRa-enabled edge devices
   - Implement real-time sliding window inference
   - Set up cloud-based monitoring dashboard
   - Configure alert system for illegal logging detection

4. **Performance Tuning**:
   - Quantization for edge deployment
   - Model pruning to reduce size
   - ONNX export for cross-platform compatibility

## Comparison with State-of-the-Art

| Model | Parameters | Accuracy (Expected) | Deployment |
|-------|-----------|---------------------|------------|
| SimpleCNN | 618K | 47% (current) | ✅ Easy |
| TFCNN | 2.57M | 60-75% (expected) | ✅ Moderate |
| AST | 86M | 80-90% (expected) | ❌ GPU required |

## References

1. Simiyu, D. et al. "A Chainsaw-Sound Recognition Model for Detecting Illegal Logging Activities in Forests"
2. ESC-50: Dataset for Environmental Sound Classification
3. FSC22: FSD50K-Class Audio Events in Forest Soundscapes

---

**Implementation Status**: ✅ Training in Progress  
**Model Type**: TFCNN with Attention  
**Dataset**: ESC-50 (1707 train, 366 val, 367 test samples)  
**Expected Completion**: ~30-40 minutes (20 epochs on CPU)
