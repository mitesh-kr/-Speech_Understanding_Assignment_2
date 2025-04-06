# Speech Understanding Assignment 2

This repository contains the implementation for two speech processing assignments:

## Project Structure
```
├── README.md                   # Main project documentation
├── QUESTION_1/                 # Speech Enhancement in Multi-Speaker Environment
│   └── README.md               # Detailed documentation for Question 1
└── QUESTION_2/                 # MFCC Feature Extraction for Indian Languages
    ├── README.md               # Detailed documentation for Question 2
A detailed explanation can be seen in the individual readme file.
```

## Questions Overview

### Question 1: Speech Enhancement
- Speaker verification with pre-trained models
- Fine-tuning WAVLm using LoRA and ArcFace loss
- Multi-speaker separation using SepFormer
- Novel pipeline for speaker separation and enhancement
- Evaluate and compare the matrices of separated speech using these trained models. 

### Question 2: MFCC Feature Extraction for Indian Languages
- Feature extraction from 10 Indian languages
- MFCC spectrogram visualization
- Statistical analysis of MFCC features
- Language classification with neural network (86.89% validation accuracy)

For detailed implementation and results, please refer to the README files in each question's directory.
