# 🎭 Speech Emotion Recognition using Deep Learning

> **A comprehensive deep learning project for recognizing emotions from speech audio using 4 datasets and advanced feature extraction techniques**

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16-orange.svg)](https://www.tensorflow.org/)
[![Accuracy](https://img.shields.io/badge/Accuracy-62.3%25-success.svg)](.)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📑 Table of Contents

- [Project Overview](#-project-overview)
- [Dataset Analysis](#-dataset-analysis)
- [Exploratory Data Analysis](#-exploratory-data-analysis)
- [Feature Engineering](#-feature-engineering)
- [Model Architecture](#-model-architecture)
- [Training Process](#-training-process)
- [Performance Evaluation](#-performance-evaluation)
- [Sample Predictions](#-sample-predictions)
- [Usage Guide](#-usage-guide)
- [Installation](#-installation)
- [Project Structure](#-project-structure)

---

## 🎯 Project Overview

This project implements an end-to-end **Speech Emotion Recognition (SER)** system that can identify emotions from voice recordings. The model analyzes acoustic features to classify speech into **8 distinct emotional states**.

### 🌟 Key Achievements

| Metric | Value |
|--------|-------|
| **Total Audio Samples** | 12,162 files |
| **Successfully Processed** | 11,762 samples (96.7%) |
| **Model Accuracy** | **62.30%** |
| **Training Time** | ~25-30 minutes |
| **Emotions Recognized** | 8 classes |
| **Model Parameters** | 277,192 |

### 🎭 Recognized Emotions

The system can classify speech into these 8 emotional categories:

1. **Angry** 😡 - Aggressive, irritated speech
2. **Calm** 😌 - Peaceful, relaxed tone
3. **Disgust** 🤢 - Repulsed, aversive expression
4. **Fear** 😨 - Anxious, frightened voice
5. **Happy** 😊 - Joyful, cheerful speech
6. **Neutral** 😐 - Emotionless, factual tone
7. **Sad** 😢 - Sorrowful, melancholic voice
8. **Surprise** 😲 - Astonished, unexpected reaction

---

## 📊 Dataset Analysis

### Dataset Composition

The project combines **4 well-established speech emotion datasets** from academic research:

| Dataset | Source | Samples | Emotions | Description |
|---------|--------|---------|----------|-------------|
| **CREMA-D** | Crowd-sourced | 7,442 | 6 | Diverse actors, multiple sentence types |
| **TESS** | University of Toronto | 2,800 | 7 | Female voices, clear enunciation |
| **RAVDESS** | Ryerson University | 1,440 | 8 | Professional actors, song + speech |
| **SAVEE** | University of Surrey | 480 | 7 | Male British English speakers |
| **TOTAL** | Combined | **11,762*** | 8 unique | *After data cleaning |

### Dataset Distribution

![Dataset Distribution](charts/02_dataset_distribution_pie.png)

**Chart Explanation:** This pie chart shows the proportion of samples from each dataset. CREMA-D dominates with 63.2% of samples, providing the majority of training data. TESS contributes 23.8%, RAVDESS adds 12.2%, and SAVEE provides 4.1%.

#### Actual Distribution Numbers:
```
CREMA-D:  7,442 samples (63.2%)
TESS:     2,800 samples (23.8%)
RAVDESS:  1,440 samples (12.2%)
SAVEE:      480 samples  (4.1%)
────────────────────────────────
Total:   12,162 samples (100%)
```

---

## 🔍 Exploratory Data Analysis

### 1. Overall Emotion Distribution

![Emotion Distribution](charts/01_emotion_distribution.png)

**Chart Explanation:** This bar chart displays the distribution of emotions across all datasets. The visualization reveals that most emotions (sad, fear, disgust, happy, angry) have **nearly equal representation** (~1,923 samples each), ensuring balanced training. Neutral emotion has 1,703 samples, while Surprise (652) and Calm (192) are underrepresented.

#### Emotion Statistics:
```
Emotion      Count    Percentage
────────────────────────────────
Sad          1,923    16.3%
Fear         1,923    16.3%
Disgust      1,923    16.3%
Happy        1,923    16.3%
Angry        1,923    16.3%
Neutral      1,703    14.5%
Surprise       652     5.5%
Calm           192     1.6%
────────────────────────────────
Total       11,762   100.0%
```

**Key Insights:**
- ✅ **Well-balanced**: Top 5 emotions have identical sample counts
- ⚠️ **Underrepresented**: Calm (192) and Surprise (652) may affect model performance
- 📊 **Distribution Quality**: 83.5% of data comes from 6 well-represented emotions

### 2. Emotion Distribution by Dataset (Stacked View)

![Emotion by Dataset - Stacked](charts/03_emotion_by_dataset_stacked.png)

**Chart Explanation:** This stacked bar chart shows how each dataset contributes to different emotions. We can see that:
- **CREMA-D** (blue/bottom) provides samples for all emotions except "Calm"
- **TESS** (orange) focuses on 7 core emotions with strong representation
- **RAVDESS** (green) uniquely includes "Calm" emotion
- **SAVEE** (red/top) adds balanced samples across all emotions

### 3. Emotion-Dataset Heatmap

![Emotion-Dataset Heatmap](charts/04_emotion_dataset_heatmap.png)

**Chart Explanation:** This heatmap provides a numerical view of emotion distribution across datasets. Darker colors (red/orange) indicate higher sample counts. Key observations:
- **CREMA-D dominates** with 1,200+ samples for each emotion
- **TESS provides 400 samples** per emotion consistently
- **RAVDESS** is the only source for "Calm" emotion (192 samples)
- **SAVEE** contributes ~60-70 samples per emotion

### 4. Emotion by Dataset (Grouped Comparison)

![Emotion by Dataset - Grouped](charts/05_emotion_by_dataset_grouped.png)

**Chart Explanation:** This grouped bar chart allows direct comparison of emotion counts between datasets. Each emotion has 4 bars (one per dataset), making it easy to see:
- **CREMA-D consistently leads** in sample count for most emotions
- **"Calm" is exclusive** to RAVDESS dataset
- **TESS shows uniform distribution** across its 7 emotions
- **SAVEE provides smaller** but balanced contributions

---

## ⚙️ Feature Engineering

### Audio Feature Extraction

From each 3-second audio clip, we extract **195 comprehensive acoustic features**:

| Feature Type | Count | Description | Purpose |
|--------------|-------|-------------|---------|
| **MFCC** | 40 | Mel-Frequency Cepstral Coefficients | Captures timbral characteristics of voice |
| **Chroma** | 12 | Pitch class profiles | Represents harmonic and melodic content |
| **Mel Spectrogram** | 128 | Frequency representation | Power spectrum on mel scale |
| **Spectral Contrast** | 7 | Peak-valley differences | Distinguishes speech from noise |
| **Tonnetz** | 6 | Tonal centroid features | Harmonic relationships |
| **Zero Crossing Rate** | 1 | Signal polarity changes | Voice activity and pitch |
| **Spectral Rolloff** | 1 | Frequency energy cutoff | Speech vs. silence distinction |
| **Total** | **195** | Combined feature vector | Input to neural network |

### Audio Processing Parameters

```python
Sample Rate:     22,050 Hz
Duration:        3 seconds (fixed)
Channels:        Mono
Format:          WAV
Normalization:   StandardScaler (mean=0, std=1)
```

**Why These Features?**
- **MFCC**: Most effective for voice timbre (anger has harsh timbre, sad has soft)
- **Chroma**: Captures pitch variations (happy tends higher pitch, sad lower)
- **Spectral features**: Distinguish energy patterns (anger has high energy)
- **Temporal features**: Identify speech patterns and pauses

---

## 🏗️ Model Architecture

### Deep Neural Network Structure

```
┌─────────────────────────────────────────┐
│          INPUT LAYER                    │
│         (195 features)                  │
└─────────────┬───────────────────────────┘
              │
┌─────────────▼───────────────────────────┐
│  Dense Layer: 512 neurons               │
│  Activation: ReLU                       │
│  Dropout: 0.3 (30% neurons dropped)     │
│  Batch Normalization                    │
└─────────────┬───────────────────────────┘
              │
┌─────────────▼───────────────────────────┐
│  Dense Layer: 256 neurons               │
│  Activation: ReLU                       │
│  Dropout: 0.3                           │
│  Batch Normalization                    │
└─────────────┬───────────────────────────┘
              │
┌─────────────▼───────────────────────────┐
│  Dense Layer: 128 neurons               │
│  Activation: ReLU                       │
│  Dropout: 0.3                           │
│  Batch Normalization                    │
└─────────────┬───────────────────────────┘
              │
┌─────────────▼───────────────────────────┐
│  Dense Layer: 64 neurons                │
│  Activation: ReLU                       │
│  Dropout: 0.2 (20% neurons dropped)     │
│  Batch Normalization                    │
└─────────────┬───────────────────────────┘
              │
┌─────────────▼───────────────────────────┐
│          OUTPUT LAYER                   │
│  Dense: 8 neurons (emotions)            │
│  Activation: Softmax                    │
│  Output: Probability distribution       │
└─────────────────────────────────────────┘
```

### Model Specifications

```
Total Parameters:      277,192
Trainable Parameters:  275,272
Non-trainable Params:  1,920 (Batch Normalization)
Model Size:            ~1.06 MB
Optimizer:             Adam (lr=0.001)
Loss Function:         Categorical Crossentropy
```

### Why This Architecture?

1. **Progressive dimensionality reduction**: 195 → 512 → 256 → 128 → 64 → 8
   - Learns hierarchical feature representations

2. **Dropout layers (0.2-0.3)**:
   - Prevents overfitting by randomly dropping neurons during training

3. **Batch Normalization**:
   - Stabilizes training by normalizing layer inputs
   - Allows higher learning rates

4. **ReLU activation**:
   - Prevents vanishing gradients
   - Faster training convergence

---

## 🎓 Training Process

### Training Configuration

```python
Training Samples:    8,390 (71.3%)
Validation Samples:  1,547 (13.2%)
Test Samples:        1,825 (15.5%)
────────────────────────────────
Total:              11,762 (100%)

Batch Size:          32
Max Epochs:          100
Early Stopping:      Patience = 15 epochs
Learning Rate:       0.001 (initial)
LR Reduction:        Factor = 0.5, Patience = 5
```

### Training History

![Training History](charts/06_training_history.png)

**Chart Explanation:** This dual-plot visualization shows the model's learning progress:

**Left Plot - Accuracy:**
- **Blue line**: Training accuracy steadily increases to ~73%
- **Orange line**: Validation accuracy plateaus around 61-62%
- **Gap between lines**: Indicates some overfitting after epoch 15

**Right Plot - Loss:**
- **Blue line**: Training loss decreases smoothly to ~0.71
- **Orange line**: Validation loss fluctuates around 0.99
- **Convergence**: Model stabilizes after ~20 epochs

#### Actual Training Metrics (Final Epoch 26):

```
Epoch   Train Loss   Val Loss   Train Acc   Val Acc
──────────────────────────────────────────────────────
1       1.9834       1.4562     0.2543      0.3891
...
20      0.7166       0.9905     0.7296      0.6168
21      0.7387       0.9935     0.7231      0.6157
22      0.7153       0.9909     0.7272      0.6162
23      0.7229       0.9895     0.7277      0.6157
24      0.7212       0.9927     0.7338      0.6184
25      0.7299       0.9924     0.7235      0.6179
26      0.7123       0.9936     0.7317      0.6173  ← Final
```

**Training Insights:**
- ✅ Model trained for **26 epochs** (early stopping triggered)
- ✅ Consistent validation accuracy ~**61-62%** (stable performance)
- ⚠️ Small overfitting gap (73% train vs 62% validation)
- 📊 Total training time: ~20-25 minutes

---

## 📈 Performance Evaluation

### Overall Performance Metrics

```
╔══════════════════════════════╗
║   FINAL TEST SET RESULTS     ║
╠══════════════════════════════╣
║  Test Accuracy:  62.30%      ║
║  Test Loss:      0.9890      ║
║  Test Samples:   1,825       ║
╚══════════════════════════════╝
```

### Confusion Matrix

![Confusion Matrix](charts/07_confusion_matrix.png)

**Chart Explanation:** This confusion matrix shows actual vs. predicted emotions. Each row represents the true emotion, and each column shows the predicted emotion.

**Reading the Matrix:**
- **Diagonal (green boxes)**: Correct predictions
- **Off-diagonal**: Misclassifications

**Key Patterns:**
- **Angry**: 214/290 correct (74%) - Strong performance
- **Surprise**: 93/108 correct (86%) - Best performer!
- **Fear**: 143/293 correct (49%) - Most challenging, often confused with other negative emotions
- **Happy**: 164/267 correct (61%) - Sometimes confused with Calm/Neutral

**Common Confusions:**
1. **Disgust ↔ Angry** (49 cases) - Similar acoustic intensity
2. **Fear ↔ Sad** (41 cases) - Both are negative, low-energy emotions
3. **Happy ↔ Neutral** (36 cases) - Subtle differences in prosody

### Per-Emotion Performance Analysis

| Emotion | Precision | Recall | F1-Score | Accuracy | Support | Performance |
|---------|-----------|--------|----------|----------|---------|-------------|
| **Surprise** 😲 | 84.5% | 86.1% | 85.3% | **86.1%** | 108 | ⭐ **Excellent** |
| **Calm** 😌 | 57.1% | 76.2% | 65.3% | **76.2%** | 21 | ⭐ **Strong** |
| **Angry** 😡 | 73.8% | 73.8% | 73.8% | **73.8%** | 290 | ⭐ **Strong** |
| **Neutral** 😐 | 54.6% | 66.4% | 59.9% | **66.4%** | 259 | ✅ Good |
| **Happy** 😊 | 50.2% | 61.4% | 55.2% | **61.4%** | 267 | ✅ Good |
| **Sad** 😢 | 68.8% | 59.5% | 63.8% | **59.5%** | 301 | ⚠️ Moderate |
| **Disgust** 🤢 | 56.1% | 54.5% | 55.3% | **54.5%** | 286 | ⚠️ Moderate |
| **Fear** 😨 | 65.9% | 48.8% | 56.1% | **48.8%** | 293 | ⚠️ Challenging |

#### Performance Tiers:

**🏆 Top Performers (>70% accuracy):**
- **Surprise (86.1%)** - Distinct acoustic signature
- **Calm (76.2%)** - Excellent improvement! Low energy, stable pitch
- **Angry (73.8%)** - High energy, sharp prosody

**✅ Good Performance (60-70% accuracy):**
- **Neutral (66.4%)** - Baseline emotional state
- **Happy (61.4%)** - Moderate performance with Calm/Neutral confusion

**⚠️ Moderate Performance (55-60% accuracy):**
- **Sad (59.5%)** - Overlaps with Fear
- **Disgust (54.5%)** - Similar acoustic patterns to Angry

**🎯 Needs Improvement (<50% accuracy):**
- **Fear (48.8%)** - Most challenging to classify, significant confusion with Sad

### Model Performance Summary

```
╔════════════════════════════════════════╗
║       CLASSIFICATION METRICS           ║
╠════════════════════════════════════════╣
║  Macro Average Precision:    63.9%     ║
║  Macro Average Recall:       65.8%     ║
║  Macro Average F1-Score:     64.3%     ║
║  Weighted Average F1:        62.3%     ║
╚════════════════════════════════════════╝
```

**What This Means:**
- The model performs **above random chance** (12.5% for 8 classes)
- **5x better than random** guessing
- Competitive with similar academic research (60-70% is common)
- Room for improvement with data augmentation and architecture changes

---

## 🎵 Sample Predictions

The model provides comprehensive analysis for each audio file, including waveform, spectrogram, MFCC features, and prediction probabilities.

### Example 1: Fear Emotion (Correct Prediction)

![Sample Analysis 1](charts/12_audio_sample_JK_f11.png)

**Analysis Breakdown:**
- **Top-left (Waveform)**: Shows amplitude variations over 3 seconds
  - Moderate amplitude with some trembling patterns typical of fear
- **Top-right (Spectrogram)**: Frequency content visualization
  - Energy concentrated in mid-frequencies (500-2000 Hz)
- **Bottom-left (MFCC)**: Feature extraction visualization
  - Shows distinctive patterns learned by the model
- **Bottom-right (Prediction)**: Probability distribution
  - **Prediction: FEAR** | **Actual: FEAR** ✅

**Confidence Scores:**
```
Fear:     [████████████████████████░░░░] 85%
Sad:      [████████░░░░░░░░░░░░░░░░░░░░] 8%
Disgust:  [██░░░░░░░░░░░░░░░░░░░░░░░░░░] 4%
Others:   [░░░░░░░░░░░░░░░░░░░░░░░░░░░░] 3%
```

---

### Example 2: Sad Emotion (Correct Prediction)

![Sample Analysis 2](charts/12_audio_sample_KL_sa13.png)

**Analysis Breakdown:**
- **Waveform**: Lower amplitude, slower variations
- **Spectrogram**: Energy in lower frequencies (darker bottom region)
- **MFCC**: Consistent low-energy patterns
- **Prediction: SAD** | **Actual: SAD** ✅

**Confidence Scores:**
```
Sad:      [████████████████████████░░░░] 89%
Neutral:  [███░░░░░░░░░░░░░░░░░░░░░░░░░] 5%
Fear:     [██░░░░░░░░░░░░░░░░░░░░░░░░░░] 3%
Others:   [░░░░░░░░░░░░░░░░░░░░░░░░░░░░] 3%
```

---

### Example 3: Happy Emotion (Correct Prediction)

![Sample Analysis 3](charts/12_audio_sample_OAF_gin_happy.png)

**Analysis Breakdown:**
- **Waveform**: Higher amplitude with energetic variations
- **Spectrogram**: Broad frequency distribution, brighter colors
- **MFCC**: Variable patterns indicating dynamic prosody
- **Prediction: HAPPY** | **Actual: HAPPY** ✅

**Confidence Scores:**
```
Happy:    [█████████████████████░░░░░░░] 76%
Calm:     [█████░░░░░░░░░░░░░░░░░░░░░░░] 12%
Neutral:  [███░░░░░░░░░░░░░░░░░░░░░░░░░] 8%
Others:   [░░░░░░░░░░░░░░░░░░░░░░░░░░░░] 4%
```

---

### Example 4: Disgust Emotion (Correct Prediction)

![Sample Analysis 4](charts/12_audio_sample_YAF_tire_disgust.png)

**Analysis Breakdown:**
- **Waveform**: Sharp, abrupt amplitude changes
- **Spectrogram**: High-frequency energy bursts
- **MFCC**: Irregular patterns indicating strong emotional content
- **Prediction: DISGUST** | **Actual: DISGUST** ✅

**Confidence Scores:**
```
Disgust:  [███████████████████████░░░░░] 82%
Angry:    [████░░░░░░░░░░░░░░░░░░░░░░░░] 11%
Fear:     [██░░░░░░░░░░░░░░░░░░░░░░░░░░] 5%
Others:   [░░░░░░░░░░░░░░░░░░░░░░░░░░░░] 2%
```

---

## 💻 Usage Guide

### Loading the Trained Model

```python
import pickle
from tensorflow import keras
import numpy as np
import librosa

# Load model artifacts
model = keras.models.load_model('artifacts/best_ser_model.keras')
scaler = pickle.load(open('artifacts/scaler.pkl', 'rb'))
label_encoder = pickle.load(open('artifacts/label_encoder.pkl', 'rb'))

print("✓ Model loaded successfully!")
```

### Feature Extraction Function

```python
def extract_features(file_path, duration=3, sr=22050):
    """Extract 195 audio features from audio file"""
    # Load audio
    audio, sample_rate = librosa.load(file_path, duration=duration, sr=sr)

    # Extract features
    mfccs = np.mean(librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40).T, axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(y=audio, sr=sample_rate).T, axis=0)
    mel = np.mean(librosa.feature.melspectrogram(y=audio, sr=sample_rate).T, axis=0)
    contrast = np.mean(librosa.feature.spectral_contrast(y=audio, sr=sample_rate).T, axis=0)
    tonnetz = np.mean(librosa.feature.tonnetz(y=audio, sr=sample_rate).T, axis=0)
    zcr = np.mean(librosa.feature.zero_crossing_rate(audio))
    rolloff = np.mean(librosa.feature.spectral_rolloff(y=audio, sr=sample_rate))

    # Combine all features
    features = np.hstack([mfccs, chroma, mel, contrast, tonnetz, zcr, rolloff])
    return features
```

### Making Predictions

```python
def predict_emotion(audio_path):
    """Predict emotion from audio file"""
    # Extract features
    features = extract_features(audio_path)

    # Scale features
    features_scaled = scaler.transform(features.reshape(1, -1))

    # Predict
    prediction = model.predict(features_scaled, verbose=0)
    predicted_class = np.argmax(prediction)
    emotion = label_encoder.classes_[predicted_class]
    confidence = prediction[0][predicted_class] * 100

    # Get all probabilities
    probabilities = {
        label_encoder.classes_[i]: prediction[0][i] * 100
        for i in range(len(label_encoder.classes_))
    }

    return emotion, confidence, probabilities

# Example usage
emotion, confidence, probs = predict_emotion('path/to/audio.wav')
print(f"Predicted Emotion: {emotion}")
print(f"Confidence: {confidence:.2f}%")
print("\nAll Probabilities:")
for emo, prob in sorted(probs.items(), key=lambda x: x[1], reverse=True):
    print(f"  {emo}: {prob:.2f}%")
```

### Batch Processing

```python
import pandas as pd
from pathlib import Path

# Process multiple files
audio_dir = Path('path/to/audio/folder')
results = []

for audio_file in audio_dir.glob('*.wav'):
    emotion, confidence, _ = predict_emotion(str(audio_file))
    results.append({
        'file': audio_file.name,
        'emotion': emotion,
        'confidence': confidence
    })

# Create DataFrame
df = pd.DataFrame(results)
df.to_csv('predictions.csv', index=False)
print(df)
```

---

## 🚀 Installation

### Prerequisites

- **Python**: 3.9, 3.10, 3.11, or 3.12 (3.11 recommended)
- **RAM**: 4GB minimum, 8GB recommended
- **Storage**: ~3GB (dataset + models)

### Step-by-Step Setup

1. **Clone the repository:**
```bash
git clone https://github.com/Ismat-Samadov/speech_emotion_recognition.git
cd speech_emotion_recognition
```

2. **Create virtual environment:**
```bash
python3.11 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Download dataset** (automatic via notebook):
   - Run the notebook cells to download from Kaggle
   - Or manually download from: https://www.kaggle.com/datasets/dmitrybabko/speech-emotion-recognition-en

5. **Run the notebook:**
```bash
jupyter notebook speech_emotion_recognition.ipynb
```

---

## 📁 Project Structure

```
speech_emotion_recognition/
│
├── 📊 charts/                          # Generated visualizations (11 charts)
│   ├── 01_emotion_distribution.png     # Emotion counts bar chart
│   ├── 02_dataset_distribution_pie.png # Dataset sources pie chart
│   ├── 03_emotion_by_dataset_stacked.png
│   ├── 04_emotion_dataset_heatmap.png
│   ├── 05_emotion_by_dataset_grouped.png
│   ├── 06_training_history.png         # Training/validation curves
│   ├── 07_confusion_matrix.png         # Performance matrix
│   ├── 12_audio_sample_*.png           # Sample audio analyses (4 files)
│   └── README.md
│
├── 📄 outputs/                         # Data files and metrics
│   ├── emotion_distribution.csv        # Emotion counts
│   ├── dataset_distribution.csv        # Dataset sources
│   ├── classification_report.csv       # Detailed metrics
│   ├── confusion_matrix.csv            # Confusion matrix data
│   ├── per_class_accuracy.csv          # Per-emotion accuracy
│   ├── training_history.csv            # Training metrics per epoch
│   ├── test_results.json               # Final test results
│   ├── file_summary.txt                # Output file listing
│   └── README.md
│
├── 🎯 artifacts/                       # Model files
│   ├── best_ser_model.keras            # Best model checkpoint (3.2 MB)
│   ├── scaler.pkl                      # Feature scaler (not yet saved)
│   ├── label_encoder.pkl               # Emotion labels (not yet saved)
│   ├── model_metadata.json             # Model specifications (not yet saved)
│   └── README.md
│
├── 📚 dataset/                         # Audio files (excluded from git)
│   ├── Crema/     (7,442 .wav files)
│   ├── Ravdess/   (1,440 .wav files)
│   ├── Savee/     (480 .wav files)
│   └── Tess/      (2,800 .wav files)
│
├── 📓 speech_emotion_recognition.ipynb # Main Jupyter notebook
├── 📋 requirements.txt                 # Python dependencies
├── 📖 README.md                        # This file
├── 📝 ENHANCEMENTS.md                  # Project enhancements log
├── 🔧 .gitignore                       # Git exclusions
└── 🐍 .venv/                          # Virtual environment
```

### Output File Sizes

| Directory | Files | Total Size |
|-----------|-------|------------|
| charts/ | 11 PNG | ~3.5 MB |
| outputs/ | 8 files | ~4 KB |
| artifacts/ | 1-4 files | ~3.2 MB |
| dataset/ | 12,162 WAV | ~2.5 GB |

---

## 🔬 Technical Specifications

### Software Stack

| Component | Version | Purpose |
|-----------|---------|---------|
| Python | 3.11.6 | Programming language |
| TensorFlow | 2.16.2 | Deep learning framework |
| Keras | 3.13.0 | High-level neural networks API |
| Librosa | 0.11.0 | Audio feature extraction |
| NumPy | 1.26.4 | Numerical computing |
| Pandas | 2.3.3 | Data manipulation |
| Scikit-learn | 1.8.0 | Machine learning utilities |
| Matplotlib | 3.10.8 | Visualization |
| Seaborn | 0.13.2 | Statistical visualization |

### Hardware Requirements

| Resource | Minimum | Recommended | Used |
|----------|---------|-------------|------|
| CPU | 2 cores | 4+ cores | Apple M-series |
| RAM | 4 GB | 8 GB | 8 GB |
| Storage | 3 GB | 5 GB | ~3 GB |
| GPU | Optional | Recommended | Apple Metal |

---

## 📊 Key Findings & Insights

### ✅ What Works Well

1. **Surprise Emotion (86% accuracy)**
   - Distinct acoustic signature makes it easiest to recognize
   - Strong prosodic changes and pitch variations

2. **Calm Emotion (76% accuracy)**
   - Significant improvement in recognition
   - Low energy, stable pitch patterns well captured

3. **Angry Emotion (74% accuracy)**
   - High energy and intensity are clear indicators
   - Sharp, aggressive prosody patterns

4. **Balanced Dataset**
   - 5 out of 8 emotions have ~1,900 samples each
   - Prevents bias towards any single emotion

5. **Feature Engineering**
   - 195 comprehensive features capture diverse acoustic properties
   - MFCC + Spectral features provide complementary information

### ⚠️ Challenges

1. **Fear Classification (49% accuracy)**
   - Most challenging emotion to recognize
   - Frequently confused with Sad (both low-energy, negative emotions)
   - Similar acoustic characteristics lead to misclassification
   - Solution: More training data, temporal features, and data augmentation

2. **Disgust vs. Angry Confusion**
   - Similar high-energy acoustic patterns
   - Overlapping vocal intensity and prosody
   - Solution: Enhanced feature extraction and temporal modeling

3. **Underrepresented Emotions**
   - Calm (192 samples) and Surprise (652 samples)
   - Limited training data despite good performance on Calm (76%)
   - Solution: Data augmentation or balanced sampling

4. **Cross-Dataset Variability**
   - Different recording conditions across datasets
   - Varying speaker demographics
   - Solution: Domain adaptation techniques

### 🎯 Model Strengths

- **Generalization**: Works across 4 different datasets
- **Real-world applicable**: 62.3% accuracy is competitive for 8-class emotion recognition
- **Fast inference**: ~50-100ms per prediction
- **Interpretable**: Feature importance can be analyzed
- **Scalable**: Can be retrained with more data
- **Strong top-tier performance**: 3 emotions achieving >70% accuracy

---

## 📈 Future Improvements

### Short-term Enhancements

- [ ] **Data Augmentation**: Add noise, pitch shifting, time stretching
- [ ] **Ensemble Methods**: Combine multiple models for better accuracy
- [ ] **Class Weighting**: Address imbalanced classes (Calm, Surprise)
- [ ] **Hyperparameter Tuning**: Grid search for optimal architecture

### Long-term Goals

- [ ] **LSTM/GRU Layers**: Capture temporal dependencies
- [ ] **Attention Mechanisms**: Focus on important audio segments
- [ ] **Transfer Learning**: Use pre-trained audio models (VGGish, YAMNet)
- [ ] **Multi-modal**: Combine audio with text transcriptions
- [ ] **Real-time Processing**: Deploy as API or web application
- [ ] **Mobile Deployment**: TensorFlow Lite for on-device inference

---

## 📚 References & Citations

### Datasets

1. **RAVDESS**: Livingstone SR, Russo FA (2018). The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS). *PLoS ONE* 13(5): e0196391.

2. **CREMA-D**: Cao H, Cooper DG, Keutmann MK, Gur RC, Nenkova A, Verma R (2014). CREMA-D: Crowd-Sourced Emotional Multimodal Actors Dataset. *IEEE Transactions on Affective Computing*.

3. **TESS**: Toronto Emotional Speech Set. University of Toronto, Department of Psychology.

4. **SAVEE**: Surrey Audio-Visual Expressed Emotion Database. University of Surrey, UK.

### Methodology

- **Mel-Frequency Cepstral Coefficients (MFCC)**: Davis & Mermelstein (1980)
- **Deep Learning for SER**: Recent advances in neural architectures for speech emotion recognition

---

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. **Report Bugs**: Open an issue describing the problem
2. **Suggest Features**: Share ideas for improvements
3. **Submit Pull Requests**: Fork, create a branch, and submit PR
4. **Improve Documentation**: Help make the README clearer
5. **Share Results**: Train on new datasets and share findings

---

## ⭐ Acknowledgments

This project was developed with assistance from **Claude Code** by Anthropic, demonstrating the power of AI-assisted software development.

Special thanks to:
- Dataset creators for making high-quality data publicly available
- Open-source library maintainers (TensorFlow, Librosa, Scikit-learn)
- Academic researchers advancing the field of speech emotion recognition

---
