"""
Configuration parameters for speaker identification model.
"""
import os
import torch

# Paths
VOX2_WAV_DIR = '/path/to/VOX2'  # Update with your VoxCeleb2 path
VOX1_DIR = '/path/to/VOX1/vox1_test_wav/wav'  # Update with your VoxCeleb1 path
TRIAL_PAIRS_PATH = '/path/to/VOX1/vox1_test_wav/meta/veri_test2.txt'  # Update with your path
CHECKPOINT_DIR = 'checkpoints'
PLOTS_DIR = 'plots'

# Audio parameters
SAMPLE_RATE = 16000
FIXED_DURATION = 5.0  # seconds
FIXED_SAMPLES = int(SAMPLE_RATE * FIXED_DURATION)

# Training parameters
BATCH_SIZE = 8
NUM_EPOCHS = 20
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
GRAD_CLIP = 3.0

# Model parameters
EMBEDDING_SIZE = 512
NUM_SPEAKERS = 100  # Number of speakers for training

# LoRA parameters
LORA_RANK = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.1

# ArcFace parameters
ARCFACE_MARGIN = 0.3
ARCFACE_SCALE = 30

# Evaluation parameters
EER_THRESHOLD = 0.5

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Random seeds for reproducibility
RANDOM_SEED = 42

# Create directories if they don't exist
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)
