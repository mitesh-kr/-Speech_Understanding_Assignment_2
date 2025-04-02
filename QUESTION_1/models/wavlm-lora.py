"""
WavLM model with LoRA adaptation and ArcFace for speaker identification.
"""
import torch
import torch.nn as nn
from transformers import WavLMModel
from peft import LoraConfig, get_peft_model
from models.arcface import ArcFaceLayer
from config import (
    EMBEDDING_SIZE, 
    NUM_SPEAKERS, 
    LORA_RANK, 
    LORA_ALPHA, 
    LORA_DROPOUT
)

class WavLM_Lora_ArcModel(nn.Module):
    """
    Speaker identification model using WavLM with LoRA adaptation and ArcFace loss.
    
    Args:
        base_model: pre-trained WavLM model
        num_speakers: number of speakers for classification
        embedding_size: dimensionality of speaker embeddings
    """
    def __init__(self, base_model, num_speakers=NUM_SPEAKERS, embedding_size=EMBEDDING_SIZE):
        super().__init__()
        self.base_model = base_model
        
        # Projection layer from WavLM output to embedding space
        self.projection = nn.Sequential(
            nn.Linear(base_model.config.hidden_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, embedding_size)
        )
        
        # ArcFace layer for classification
        self.arcface = ArcFaceLayer(embedding_size, num_speakers)
    
    def forward(self, input_values, labels=None):
        """
        Forward pass for the combined model.
        
        Args:
            input_values: audio input
            labels: speaker labels (optional, for training)
            
        Returns:
            During training (with labels): (logits, embeddings)
            During inference (without labels): embeddings
        """
        # Get WavLM output
        outputs = self.base_model(input_values).last_hidden_state
        
        # Pool over time dimension to get fixed-size representation
        embeddings = torch.mean(outputs, dim=1)
        
        # Project to embedding space
        embeddings = self.projection(embeddings)
        
        # If labels are provided (training mode)
        if labels is not None:
            logits = self.arcface(embeddings, labels)
            return logits, embeddings
            
        # Inference mode
        return embeddings


def create_model(num_speakers=NUM_SPEAKERS, embedding_size=EMBEDDING_SIZE):
    """
    Creates a WavLM model with LoRA adaptation and ArcFace.
    
    Args:
        num_speakers: number of speakers for classification
        embedding_size: dimensionality of speaker embeddings
        
    Returns:
        Initialized model
    """
    # Load pre-trained WavLM model
    base_model = WavLMModel.from_pretrained("microsoft/wavlm-base-plus")
    
    # Freeze all parameters
    for param in base_model.parameters():
        param.requires_grad = False
    
    # Configure LoRA
    lora_config = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
    )
    
    # Apply LoRA to the model
    base_model = get_peft_model(base_model, lora_config)
    
    # Create the combined model
    model = WavLM_Lora_ArcModel(base_model, num_speakers, embedding_size)
    
    return model
