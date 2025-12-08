# Audio Deepfake Detection – Hybrid Deep Learning Model

This repository contains a full pipeline for **audio deepfake detection** using a hybrid deep learning approach.  
The model extracts features from raw audio (e.g., spectrograms, wavelets) and trains deep neural networks to classify clips as **real** or **fake**.

## Features

- Audio loading and preprocessing with `librosa`
- Wavelet-based feature extraction using `pywt`
- Train/validation split with `scikit-learn`
- PyTorch `AudioDataset` and dataloaders
- Deep learning model training loop with `torch`
- Evaluation with accuracy, confusion matrix, ROC curve, and AUC
- Visualizations using `matplotlib` and `seaborn`

## Requirements

Install dependencies with:

```bash
pip install -r requirements.txt
