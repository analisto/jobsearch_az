# Speech Emotion Recognition - Setup Guide

## Python Version Requirement

This project requires **Python 3.9, 3.10, 3.11, or 3.12**. TensorFlow does not yet support Python 3.14.

## Setup Instructions

### Option 1: Using Python 3.11 (Recommended)

1. **Check if Python 3.11 is installed:**
   ```bash
   python3.11 --version
   ```

2. **If not installed, install Python 3.11:**
   - **macOS (using Homebrew):**
     ```bash
     brew install python@3.11
     ```
   - **macOS (using pyenv):**
     ```bash
     pyenv install 3.11.7
     ```

3. **Create a virtual environment:**
   ```bash
   python3.11 -m venv .venv
   ```

4. **Activate the virtual environment:**
   ```bash
   source .venv/bin/activate
   ```

5. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

6. **Launch Jupyter:**
   ```bash
   pip install jupyter
   jupyter notebook speech_emotion_recognition.ipynb
   ```

### Option 2: Using Conda

1. **Create a conda environment with Python 3.11:**
   ```bash
   conda create -n ser python=3.11
   ```

2. **Activate the environment:**
   ```bash
   conda activate ser
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Launch Jupyter:**
   ```bash
   conda install jupyter
   jupyter notebook speech_emotion_recognition.ipynb
   ```

## For macOS Apple Silicon Users

If you're on macOS with Apple Silicon (M1/M2/M3), the installation will automatically use `tensorflow-macos` which is optimized for ARM architecture.

## Kaggle Authentication

To download the dataset, you'll need to authenticate with Kaggle:

1. Create a Kaggle account at https://www.kaggle.com
2. Go to Account settings and create an API token
3. Download `kaggle.json` and place it in `~/.kaggle/`
4. Set permissions: `chmod 600 ~/.kaggle/kaggle.json`

Alternatively, `kagglehub` will prompt you to authenticate on first use.

## Verify Installation

After installation, verify everything works:

```python
import tensorflow as tf
import librosa
import kagglehub

print(f"TensorFlow version: {tf.__version__}")
print(f"Librosa version: {librosa.__version__}")
print("All dependencies loaded successfully!")
```

## Troubleshooting

### Issue: "No module named 'tensorflow'"
- Make sure you're using Python 3.9-3.12
- Verify your virtual environment is activated
- Try reinstalling: `pip install --upgrade tensorflow`

### Issue: "Could not find a version that satisfies the requirement tensorflow"
- Your Python version is too new or too old
- Use Python 3.11 (most stable)

### Issue: librosa audio loading errors
- Install soundfile: `pip install soundfile`
- On macOS, you may need: `brew install libsndfile`

## Project Structure

```
speech_emotion_recognition/
├── speech_emotion_recognition.ipynb  # Main notebook
├── requirements.txt                   # Dependencies
├── SETUP.md                          # This file
└── (generated files after training)
    ├── best_ser_model.keras          # Best model weights
    ├── speech_emotion_recognition_model.keras
    ├── scaler.pkl                    # Feature scaler
    └── label_encoder.pkl             # Label encoder
```

## Next Steps

Once your environment is set up:
1. Open `speech_emotion_recognition.ipynb`
2. Run the installation cell (cell 2)
3. Follow the notebook cells sequentially
4. The dataset will be automatically downloaded from Kaggle

Enjoy building your Speech Emotion Recognition model!
