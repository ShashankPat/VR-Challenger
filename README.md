# Shashank Challenger

A PyTorch-based multi‐label image classification pipeline for the VRL Challenge dataset.

## Overview

This repository contains a single script, `shashank_challenger.py`, which:

1. Downloads and imports the VRL Challenge and serialized image datasets from Kaggle via KaggleHub.  
2. Loads and preprocesses multi‐label annotations.  
3. Splits data into training, validation, and test sets.  
4. Defines a flexible **multi‐head model** on top of various backbones (ResNet, ConvNeXt, EfficientNet, Swin, ViT).  
5. Trains with class‐balanced sampling, mixed‐precision, and early stopping.  
6. Evaluates per‐category and exact‐match accuracies.  
7. Serializes test‐time predictions to CSV and visualizes sample inferences.

---

## Requirements

- Python 3.7+  
- **PyTorch**, **torchvision**  
- **pandas**, **numpy**  
- **scikit-learn**  
- **Pillow**  
- **tqdm**  
- **kagglehub** (for data download)  

```bash
pip install torch torchvision pandas numpy scikit-learn pillow tqdm kagglehub
