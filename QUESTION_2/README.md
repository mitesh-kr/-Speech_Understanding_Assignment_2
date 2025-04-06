# MFCC Feature Extraction and Comparative Analysis of Indian Languages

## Project Overview
This project focuses on extracting Mel-Frequency Cepstral Coefficients (MFCC) from audio samples of different Indian languages and using these features to build a language classification model. The implementation covers both feature extraction and the development of a neural network classifier that can identify languages based on their acoustic characteristics.

## Dataset
- Audio Dataset with 10 Indian Languages from Kaggle
- Extracted MFCCs are stored in `.npy` files for efficient processing

## Methodology

### Feature Extraction (Task A)
1. Downloaded the audio dataset containing 10 Indian languages
2. Implemented Python code to extract MFCC features from each audio sample
3. Generated and visualized MFCC spectrograms for representative samples from multiple languages
4. Performed comparative analysis of spectral patterns across different languages
5. Conducted statistical analysis (mean, variance) of MFCC coefficients to quantify language differences

### Classification Model (Task B)
1. Utilized the extracted MFCC features to build a language classifier
2. Implemented a 7-layer Artificial Neural Network (ANN) with the following characteristics:
   - ReLU activation functions
   - Dropout layers to prevent overfitting
   - Softmax output layer for multi-class classification

## Data Preprocessing
- Dataset split: 80% training, 20% validation/testing
- Feature scaling applied to normalize the MFCC values
- Training samples: 205,454
- Validation samples: 51,364

## Model Performance
- Best results achieved after 17 epochs of training
- **Training Loss**: 0.2872
- **Training Accuracy**: 86.65%
- **Validation Loss**: 0.2814
- **Validation Accuracy**: 86.89%

## Key Findings
- The model successfully differentiates between Indian languages with high accuracy
- MFCC features effectively capture the unique acoustic properties of each language
- Dropout regularization effectively prevented overfitting during training

## Challenges and Considerations
- Speaker variability can affect MFCC patterns
- Background noise in audio samples may impact feature quality
- Regional accents within the same language present classification challenges
- Model performance may vary based on audio quality and recording conditions

## Repository Structure
```
├── README.md                   # Project documentation
├── mfcc_extraction.py          # Code for MFCC feature extraction
├── visualize_spectrograms.py   # Code for generating and comparing MFCC spectrograms
├── language_classifier.py      # Implementation of the neural network model
├── data/                       # Directory for dataset (not included in repo)
├── features/                   # Extracted MFCC features stored as .npy files
└── results/                    # Model performance metrics and visualizations
```

## Requirements
- Python 3.8+
- NumPy
- Pandas
- Librosa
- TensorFlow/Keras
- Matplotlib
- Scikit-learn

## Usage
1. Download the Audio Dataset with 10 Indian Languages from Kaggle
2. Run the MFCC extraction script to process audio files
3. Use the visualization script to compare spectrograms
4. Train the neural network model on the extracted features
5. Evaluate model performance on test data

## Future Work
- Experiment with different feature extraction techniques
- Implement more complex neural network architectures
- Explore the impact of different preprocessing methods
- Address challenges related to speaker variability and background noise
