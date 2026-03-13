# Chainsaw Detection for Deforestation Monitoring

Audio-based chainsaw detection system using Audio Spectrogram Transformer (AST) for real-time deforestation monitoring.

## 🎯 Project Overview

This project uses deep learning to detect chainsaw sounds in audio recordings, enabling automated monitoring of illegal logging activities in forests.

**Model**: Audio Spectrogram Transformer (AST)  
**Task**: Binary audio classification (chainsaw vs non-chainsaw)  
**Expected Performance**: 92-96% accuracy

## 📋 Requirements

- Python 3.8+
- CUDA-capable GPU (recommended)
- 8GB+ RAM
- ~10GB disk space for datasets

## 🚀 Quick Start

### 1. Installation

```bash
# Clone or navigate to project directory
cd ISTP

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### 2. Setup Kaggle API (for dataset download)

1. Create Kaggle account at https://www.kaggle.com
2. Go to Account → API → Create New API Token
3. Download `kaggle.json`
4. Place it in: `C:\Users\<username>\.kaggle\kaggle.json` (Windows)

### 3. Download Datasets

```bash
python data_download.py
```

This will download:
- Rainforest Connection (RFCx) dataset
- ESC-50 dataset
- Instructions for AudioSet

**Manual organization needed:**
After download, organize your audio files:
```
data/raw/
├── chainsaw/          # Put chainsaw audio files here
│   ├── chainsaw_001.wav
│   ├── chainsaw_002.wav
│   └── ...
└── non_chainsaw/      # Put forest ambient/other sounds here
    ├── ambient_001.wav
    ├── birds_001.wav
    └── ...
```

### 4. Preprocess Data

```bash
python preprocess.py
```

This will:
- Resample audio to 16kHz
- Normalize lengths to 10 seconds
- Apply data augmentation (training only)
- Split into train/val/test sets (70/15/15)

### 5. Train Model

```bash
python train.py
```

Training takes ~2-4 hours on GPU (depends on dataset size).

Monitor training:
```bash
tensorboard --logdir output/
```

### 6. Evaluate Model

```bash
# Find your model path in output/ directory
python evaluate.py --model-path output/model_YYYYMMDD_HHMMSS/final_model --split test
```

This generates:
- Accuracy, precision, recall, F1-score
- Confusion matrix
- ROC curve
- Detailed predictions CSV

### 7. Run Inference

**Single file:**
```bash
python inference.py --model-path output/model_YYYYMMDD_HHMMSS/final_model --audio test_audio.wav
```

**Long audio with sliding window:**
```bash
python inference.py --model-path output/model_YYYYMMDD_HHMMSS/final_model --audio long_recording.wav --long-audio
```

**Batch processing:**
```bash
python inference.py --model-path output/model_YYYYMMDD_HHMMSS/final_model --audio-dir data/test_audio/ --output results.csv
```

## 📁 Project Structure

```
ISTP/
├── config.yaml              # Configuration file
├── requirements.txt         # Python dependencies
├── data_download.py         # Dataset download script
├── preprocess.py           # Audio preprocessing
├── train.py                # Model training
├── evaluate.py             # Model evaluation
├── inference.py            # Run predictions
├── README.md               # This file
├── data/
│   ├── raw/               # Raw audio files
│   └── processed/         # Preprocessed data
└── output/                # Trained models and results
```

## ⚙️ Configuration

Edit `config.yaml` to customize:

- **Audio settings**: Sample rate, duration, mel-spectrogram parameters
- **Data augmentation**: Time stretch, pitch shift, noise
- **Model**: AST variant, dropout, freezing layers
- **Training**: Batch size, learning rate, epochs, early stopping
- **Hardware**: GPU usage, workers, mixed precision

## 📊 Expected Results

With proper dataset (1000+ samples):
- **Accuracy**: 92-96%
- **Precision**: 90-95%
- **Recall**: 88-93%
- **F1-Score**: 89-94%
- **Inference time**: ~100ms per 10-second clip

## 🎓 Datasets Used

1. **Rainforest Connection (RFCx)** - Primary dataset with real forest recordings
2. **AudioSet** - Google's large-scale audio dataset (chainsaw class)
3. **ESC-50** - Environmental sound classification dataset

## 🔧 Troubleshooting

**CUDA out of memory:**
- Reduce batch size in `config.yaml`
- Set `mixed_precision: true`

**No audio files found:**
- Check data organization in `data/raw/chainsaw/` and `data/raw/non_chainsaw/`
- Ensure audio files are in supported formats (.wav, .mp3, .flac, .ogg)

**Low accuracy:**
- Increase dataset size (min 500 samples per class recommended)
- Check data quality and labels
- Increase training epochs
- Adjust data augmentation parameters

**Slow training:**
- Enable mixed precision: `mixed_precision: true`
- Reduce batch size or audio duration
- Use fewer augmentation techniques

## 📝 Notes

- First run will download AST model (~400MB)
- Recommended dataset size: 1000+ samples (500+ per class)
- For production deployment, consider ONNX export for faster inference
- Use sliding window approach for long audio files (>10 seconds)

## 🎯 Next Steps

1. **Collect more data** - More diverse chainsaw sounds improve accuracy
2. **Fine-tune hyperparameters** - Experiment with learning rate, batch size
3. **Deploy model** - Export to ONNX or TorchScript for edge devices
4. **Real-time monitoring** - Integrate with audio streaming pipeline
5. **Multi-class detection** - Extend to detect other logging sounds

## 📧 Support

For issues or questions:
1. Check troubleshooting section
2. Review config.yaml settings
3. Ensure data is properly organized

## 🏆 Performance Tips

- Use GPU for training (10-20x faster)
- Enable mixed precision training
- Use data augmentation to prevent overfitting
- Monitor validation metrics for early stopping
- Test on diverse audio conditions

---

**Good luck with your deforestation detection project! 🌳**
