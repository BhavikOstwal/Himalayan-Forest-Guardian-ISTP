# Dataset Comparison: What the Paper Says vs What I Used

## 📄 What the Paper Recommends

From the paper abstract:
> "Open-source datasets such as **ESC-50** and **FSC22** will be considered in model training but **the latter will be utilized more due to its overall focus on forest acoustics**."

### Paper's Dataset Strategy:
1. **Primary Dataset: FSC22** (Forest Soundscapes from FSD50K)
   - Focus: Forest acoustic events
   - Categories: Chainsaw, birds, rain, wind, thunder, insects, water streams
   - Advantage: Specifically designed for forest monitoring
   - Better environmental context for illegal logging detection

2. **Secondary Dataset: ESC-50** (Environmental Sound Classification)
   - Purpose: Additional environmental diversity
   - Categories: 50 general environmental sounds
   - Used to supplement FSC22

## ❌ What I Actually Used

### Current Implementation:
- **Only ESC-50 dataset**
- **40 chainsaw samples** from ESC-50
- **400 non-chainsaw samples** from ESC-50 general categories

### Why This is Insufficient:

| Aspect | Paper Recommendation (FSC22) | My Implementation (ESC-50 only) |
|--------|------------------------------|----------------------------------|
| **Forest Context** | ✅ Forest-specific sounds | ❌ General environmental sounds |
| **Chainsaw Samples** | 100-500+ forest chainsaws | ❌ Only 40 generic samples |
| **Background Noise** | ✅ Real forest acoustics | ❌ Mixed urban/indoor sounds |
| **Birds/Wildlife** | ✅ Forest bird species | ⚠️ Generic bird sounds |
| **Weather** | ✅ Forest rain/wind | ⚠️ Generic rain/wind |
| **Dataset Size** | ✅ Large, focused corpus | ❌ Small, generic subset |

## 🎯 What Should Be Done (Paper's Approach)

### 1. Download FSC22 Dataset
```bash
# FSC22 is part of FSD50K dataset
# Download from: https://zenodo.org/record/4060432
# OR Kaggle: https://www.kaggle.com/datasets/tonyarobertson/fsd50k
```

### 2. Filter Forest-Related Sounds
According to the paper's forest monitoring focus:

**Chainsaw Class:**
- Chainsaw sounds
- Power tools in forest
- Logging equipment

**Non-Chainsaw Class (Forest Context):**
- Bird vocalizations (forest species)
- Rain on leaves/forest canopy
- Wind through trees
- Streams/waterfalls
- Thunder in forest
- Insects (crickets, cicadas)
- Animal movements

### 3. Combine with ESC-50
Use ESC-50 as supplementary data:
- Add diversity
- Include edge cases
- Prevent overfitting to forest-only sounds

## 📊 Expected Performance Improvement

| Dataset | Accuracy | Precision | Recall | F1-Score |
|---------|----------|-----------|--------|----------|
| **ESC-50 only (current)** | 47% | 18% | 55% | 27% |
| **ESC-50 + small FSC22** | 65-70% | 40-50% | 70-75% | 50-60% |
| **FSC22 primary + ESC-50** | 75-85% | 60-70% | 80-85% | 70-80% |

## 🔧 How to Fix This

### Option 1: Download FSC22 (Recommended by Paper)
```bash
python download_fsc22.py
```

This will:
1. Guide you to download FSD50K dataset
2. Filter forest-related sounds
3. Create proper train/val/test splits
4. Combine with ESC-50

### Option 2: Use Alternative Forest Datasets

**RFCx (Rainforest Connection):**
- Real-world illegal logging audio
- Actual chainsaw sounds in rainforest context
- Available on Kaggle (requires competition acceptance)

**AudioSet (Filtered):**
- Large-scale dataset with forest categories
- Chainsaw, natural sounds, forest ambience
- Can download specific categories

**Custom Collection:**
- Record real chainsaw sounds in forest
- Collect from forest monitoring systems
- Most realistic but time-intensive

## 🎓 Why FSC22 Matters for This Application

### The Paper's Use Case: Forest Monitoring
The paper describes an **IoT-based architecture** for detecting illegal logging:

1. **Sensors in Forest**: Devices placed in actual forest environments
2. **Real-time Detection**: Must distinguish chainsaw from forest sounds
3. **Forest Acoustics**: Rain, wind, birds, insects are the actual background noise

### FSC22 Advantages:
- ✅ Trained on sounds sensors will actually hear
- ✅ Proper forest acoustic context
- ✅ Better generalization to deployment environment
- ✅ Reduces false positives from forest sounds

### ESC-50 Limitations:
- ❌ Includes urban sounds (keyboard, clock, vacuum)
- ❌ Indoor recordings (different acoustics)
- ❌ Limited forest context
- ❌ Generic chainsaw samples may not match logging chainsaws

## 📝 Summary

**What I should have done:**
1. Download FSC22 as primary dataset (**500-1000 samples**)
2. Filter for forest categories (chainsaw, birds, rain, wind, etc.)
3. Supplement with ESC-50 chainsaw samples (**40 samples**)
4. Total: **1000-2000 samples** with forest context

**What I actually did:**
1. Only ESC-50 (**440 samples total**)
2. No forest-specific context
3. Limited chainsaw diversity (**40 samples**)

**Impact:**
- Model learns generic environmental sounds, not forest acoustics
- Poor performance in actual forest deployment
- High false positives from forest sounds
- Doesn't match paper's methodology

---

**Next Step**: Run `python download_fsc22.py` to get the proper dataset as recommended in the paper.
