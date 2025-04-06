# #!/usr/bin/env python3
#!/usr/bin/env python3
"""
Advanced WavLM Speaker Classification Model with:
- LoRA Fine-Tuning
- ArcFace Loss
- Enhanced Feature Extraction
- Robust Training Strategies
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import WavLMModel, AutoFeatureExtractor
from peft import LoraConfig, get_peft_model

# Comprehensive Configuration
class SpeakerClassificationConfig:
    def __init__(
        self,
        SAMPLE_RATE=16000,
        FIXED_DURATION=5.0,
        EMBEDDING_SIZE=512,
        LORA_RANK=16,
        LORA_ALPHA=32,
        LORA_DROPOUT=0.2,
        ARCFACE_MARGIN=0.2,
        ARCFACE_SCALE=64,
    ):
        self.SAMPLE_RATE = SAMPLE_RATE
        self.FIXED_DURATION = FIXED_DURATION
        self.FIXED_SAMPLES = int(SAMPLE_RATE * FIXED_DURATION)
        self.EMBEDDING_SIZE = EMBEDDING_SIZE
        self.LORA_RANK = LORA_RANK
        self.LORA_ALPHA = LORA_ALPHA
        self.LORA_DROPOUT = LORA_DROPOUT
        self.ARCFACE_MARGIN = ARCFACE_MARGIN
        self.ARCFACE_SCALE = ARCFACE_SCALE

class EnhancedArcFaceLayer(nn.Module):
    """
    Advanced ArcFace Layer with:
    - Dynamic margin adjustment
    - Temperature scaling
    - Enhanced feature normalization
    """
    def __init__(
        self, 
        in_features, 
        out_features, 
        scale, 
        margin
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.scale = scale
        self.margin = margin
        
        # Learnable weight initialization
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        
        # Additional adaptive parameters
        self.temperature = nn.Parameter(torch.ones(1) * 1.0)
        
    def forward(self, embeddings, labels=None):
        # Advanced feature normalization
        embeddings = F.normalize(embeddings, p=2, dim=1)
        weights = F.normalize(self.weight, p=2, dim=1)
        
        # Cosine similarity computation
        cosine = F.linear(embeddings, weights)
        
        # Inference mode
        if labels is None:
            return cosine * self.scale
        
        # Training mode with angular margin
        theta = torch.acos(torch.clamp(cosine, -1.0 + 1e-7, 1.0 - 1e-7))
        
        # Dynamic margin computation
        dynamic_margin = self.margin * (1 + torch.tanh(embeddings.mean(dim=1)))
        
        # Compute phi with dynamic margin
        phi = torch.cos(theta + dynamic_margin)
        
        # Create one-hot encoded target
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1), 1)
        
        # Combine original and margin-adjusted logits
        output = one_hot * phi + (1.0 - one_hot) * cosine
        
        # Temperature-scaled scaling
        return output * self.scale * self.temperature

class WavLMSpeakerEmbedding(nn.Module):
    """
    Advanced embedding extraction with:
    - Multi-level feature aggregation
    - Attention-based pooling
    """
    def __init__(
        self, 
        base_model, 
        config):
        super().__init__()
        self.base_model = base_model
        
        # Multi-level projection
        self.projection = nn.Sequential(
            nn.Linear(base_model.config.hidden_size, 1024),
            nn.InstanceNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, config.EMBEDDING_SIZE)
        )
        
        # Self-attention pooling
        self.attention_pool = nn.MultiheadAttention(
            embed_dim=base_model.config.hidden_size, 
            num_heads=8, 
            dropout=0.2
        )
        
    def forward(self, input_values):
        # Base model feature extraction
        outputs = self.base_model(input_values).last_hidden_state
        
        # Attention-based pooling
        attention_output, _ = self.attention_pool(
            outputs.transpose(0, 1), 
            outputs.transpose(0, 1), 
            outputs.transpose(0, 1)
        )
        attention_pooled = attention_output.mean(dim=0)
        
        # Alternative: combine mean and attention pooling
        mean_pooled = outputs.mean(dim=1)
        combined_pooled = (mean_pooled + attention_pooled) / 2
        
        # Projection and normalization
        embeddings = self.projection(combined_pooled)
        return F.normalize(embeddings, p=2, dim=1)

class WavLMSpeakerClassifier(nn.Module):
    """
    Comprehensive Speaker Classification Model
    """
    def __init__(
        self, 
        num_classes=50,
        base_model_path="microsoft/wavlm-base-plus",
        config=SpeakerClassificationConfig
    ):
        super().__init__()
        
        # Base WavLM model
        self.base_model = WavLMModel.from_pretrained(base_model_path)
        self.processor = AutoFeatureExtractor.from_pretrained(base_model_path)
        self.embedding_dim = config.EMBEDDING_SIZE
        # Freeze most base model parameters
        for param in self.base_model.parameters():
            param.requires_grad = False
        
        # LoRA configuration
        lora_config = LoraConfig(
            r=config.LORA_RANK,
            lora_alpha=config.LORA_ALPHA,
            lora_dropout=config.LORA_DROPOUT,
            bias="none",
            target_modules=["q_proj", "k_proj", "v_proj", "out_proj"]
        )
        
        # Apply LoRA
        self.base_model = get_peft_model(self.base_model, lora_config)
        
        # Enhanced embedding extraction
        self.embedding_extractor = WavLMSpeakerEmbedding(
            self.base_model, 
            config
        )
        
        # ArcFace classification layer
        self.arcface = EnhancedArcFaceLayer(
            config.EMBEDDING_SIZE, 
            num_classes,
            scale=config.ARCFACE_SCALE,
            margin=config.ARCFACE_MARGIN
        )
        
        # Additional metrics tracking
        self.num_classes = num_classes
        self.embedding_size = config.EMBEDDING_SIZE
    
    def forward(self, x, labels=None):
        """
        Flexible forward pass supporting both training and inference
        """
        # Embedding extraction
        embeddings = self.embedding_extractor(x)
        
        # Classification or embedding return
        if labels is not None:
            logits = self.arcface(embeddings, labels)
            return logits, embeddings
        
        return embeddings
    
    def extract_embeddings(self, wavs):
        """Extract speaker embeddings"""
        with torch.no_grad():
            return self(wavs)
    
    def classify(self, wavs):
        """Perform classification without labels"""
        embeddings = self.extract_embeddings(wavs)
        return self.arcface(embeddings)
    
    def prepare_input(self, wav, target_sample_rate=16000, target_duration=5.0):
        """
        Robust input preparation with:
        - Resampling
        - Length normalization
        - Device handling
        """
        import torchaudio.functional as F
        
        # Ensure correct device
        device = next(self.parameters()).device
        wav = wav.to(device)
        
        # Input shape handling
        if len(wav.shape) == 1:
            wav = wav.unsqueeze(0)
        
        # Resample if needed
        if hasattr(wav, 'sample_rate') and wav.sample_rate != target_sample_rate:
            wav = F.resample(wav, wav.sample_rate, target_sample_rate)
        
        # Fixed duration handling
        target_samples = int(target_sample_rate * target_duration)
        
        if wav.shape[1] > target_samples:
            wav = wav[:, :target_samples]
        elif wav.shape[1] < target_samples:
            wav = F.pad(wav, (0, target_samples - wav.shape[1]))
        
        return wav

# Example usage and model initialization
def create_speaker_classifier(
    num_speakers=50, 
    base_model_path="microsoft/wavlm-base-plus"
):
    """
    Convenience function for model creation
    """
    model = WavLMSpeakerClassifier(
        num_classes=num_speakers, 
        base_model_path=base_model_path
    )
    return model

# Optional: Loss functions
def arcface_loss(logits, labels, device='cuda'):
    """
    Comprehensive ArcFace loss computation
    """
    return F.cross_entropy(logits, labels)

def generalized_center_loss(embeddings, labels, num_classes):
    """
    Enhanced center loss for embedding clustering
    """
    centers = torch.zeros(num_classes, embeddings.size(1), device=embeddings.device)
    centers.scatter_add_(0, labels.unsqueeze(1).repeat(1, embeddings.size(1)), embeddings)
    centers_count = torch.bincount(labels, minlength=num_classes).float()
    centers /= centers_count.unsqueeze(1)
    return torch.mean(torch.pow(embeddings - centers[labels], 2))


# print("CHEKING PRINT LORA STATEMENT")
# import math
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from transformers import WavLMModel, AutoFeatureExtractor
# from peft import LoraConfig, get_peft_model

# # Constants that should match the fine-tuning settings
# SAMPLE_RATE = 16000
# FIXED_DURATION = 5.0   # seconds
# FIXED_SAMPLES = int(SAMPLE_RATE * FIXED_DURATION)
# EMBEDDING_SIZE = 512
# LORA_RANK = 8
# LORA_ALPHA = 16
# LORA_DROPOUT = 0.1
# ARCFACE_MARGIN = 0.3
# ARCFACE_SCALE = 30


# @property
# def arcface(self):
#     return self.model.arcface

# class ArcFaceLayer(nn.Module):
#     """
#     ArcFace loss implementation for enhanced speaker verification.
#     Applies an angular margin penalty to improve feature discrimination.
#     """
#     def __init__(self, in_features, out_features, scale=ARCFACE_SCALE, margin=ARCFACE_MARGIN):
#         super().__init__()
#         self.in_features = in_features
#         self.out_features = out_features
#         self.scale = scale
#         self.margin = margin
#         self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
#         nn.init.xavier_uniform_(self.weight)

#     def forward(self, embeddings, labels=None):
#         # Normalize embeddings and weights
#         embeddings = F.normalize(embeddings, dim=1)
#         weights = F.normalize(self.weight, dim=1)
        
#         # Compute cosine similarities
#         cosine = F.linear(embeddings, weights)
        
#         # If no labels provided (inference mode), return scaled cosine similarities
#         if labels is None:
#             return cosine * self.scale
        
#         # Apply angular margin penalty
#         sine = torch.sqrt(1.0 - cosine**2)
#         phi = cosine * math.cos(self.margin) - sine * math.sin(self.margin)
        
#         # Apply penalty only to the target classes
#         one_hot = torch.zeros_like(cosine)
#         one_hot.scatter_(1, labels.view(-1, 1), 1)
#         output = one_hot * phi + (1 - one_hot) * cosine
        
#         # Apply scaling
#         return output * self.scale


# class WavLM_Lora_ArcModel(nn.Module):
#     def __init__(self, pretrained_model_name_or_path, num_speakers, embedding_size=EMBEDDING_SIZE):
#         super().__init__()
#         self.base_model = WavLMModel.from_pretrained(pretrained_model_name_or_path)
        

#         self.projection = nn.Sequential(
#             nn.Linear(self.base_model.config.hidden_size, 1024),
#             nn.ReLU(),
#             nn.Linear(1024, embedding_size)
#         )

#         self.arcface = ArcFaceLayer(embedding_size, num_speakers)

#     def forward(self, input_values, labels=None):
#         input_values = input_values.detach().clone()
#         outputs = self.base_model(input_values).last_hidden_state
#         embeddings = torch.mean(outputs, dim=1)
#         embeddings = self.projection(embeddings)

#         if labels is not None:
#             logits = self.arcface(embeddings, labels)
#             return logits, embeddings

#         return embeddings





# class WavLMSpeakerClassifier(nn.Module):
#     """
#     SpeechBrain-compatible WavLM Speaker Classifier with LoRA and ArcFace.
    
#     This model can be used as a SpeechBrain module for speaker verification
#     or identification tasks.
#     """
#     def __init__(
#         self,
#         num_classes=100,
#         base_model_path="microsoft/wavlm-base-plus",
#         pretrained_path=None,
#         lora_rank=LORA_RANK,
#         lora_alpha=LORA_ALPHA,
#         lora_dropout=LORA_DROPOUT,
#         arcface_s=ARCFACE_SCALE,
#         arcface_m=ARCFACE_MARGIN,
#         embedding_dim=EMBEDDING_SIZE,
#     ):
#         """
#         Initialize WavLM speaker classifier.
        
#         Args:
#             num_classes: Number of speakers to classify
#             base_model_path: HuggingFace model name or path to WavLM model
#             pretrained_path: Path to pretrained classifier weights (optional)
#             lora_rank: Rank for LoRA adaptation
#             lora_alpha: Scaling factor for LoRA
#             lora_dropout: Dropout rate for LoRA
#             arcface_s: Scale factor for ArcFace
#             arcface_m: Angular margin for ArcFace
#             embedding_dim: Dimension of speaker embeddings
#         """
#         super().__init__()
        
#         # Load WavLM base model
#         self.base_model = WavLMModel.from_pretrained(base_model_path)
#         self.processor = AutoFeatureExtractor.from_pretrained(base_model_path)
        
#         # Freeze base model parameters
#         for param in self.base_model.parameters():
#             param.requires_grad = False
        
#         # Configure LoRA for efficient fine-tuning
#         lora_config = LoraConfig(
#             r=lora_rank,
#             lora_alpha=lora_alpha,
#             lora_dropout=lora_dropout,
#             bias="none",
#             target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
#         )
        
#         # Apply LoRA to WavLM base model
#         self.base_model = get_peft_model(self.base_model, lora_config)
        
#         # Create classifier with WavLM, LoRA, and ArcFace
#         self.model = WavLM_Lora_ArcModel(
#             self.base_model, 
#             num_classes, 
#             embedding_size=embedding_dim
#         )
        
#         # Load pretrained weights if provided
#         if pretrained_path is not None:
#             checkpoint = torch.load(pretrained_path, map_location="cpu")
#             self.model.load_state_dict(checkpoint["model_state_dict"])
            
#         # Save parameters for reference
#         self.num_classes = num_classes
#         self.embedding_dim = embedding_dim
            
#     def forward(self, x, labels=None):
#         """
#         Forward pass through the classifier.
        
#         Args:
#             x: Audio input tensor [batch_size, sequence_length]
#             labels: Speaker labels [batch_size] (optional)
            
#         Returns:
#             If labels provided: (logits, embeddings)
#             If no labels: embeddings
#         """
#         return self.model(x, labels)
        
#     def extract_embeddings(self, wavs):
#         """
#         Extract speaker embeddings from waveforms.
        
#         Args:
#             wavs: Audio waveforms [batch_size, sequence_length]
            
#         Returns:
#             Speaker embeddings [batch_size, embedding_dim]
#         """
#         with torch.no_grad():
#             embeddings = self.model(wavs)
#         return embeddings
            
#     def classify_batch(self, wavs):
#         """
#         Classify a batch of waveforms.
        
#         Args:
#             wavs: Audio waveforms [batch_size, sequence_length]
            
#         Returns:
#             Classification logits [batch_size, num_classes]
#         """
#         embeddings = self.extract_embeddings(wavs)
#         return self.model.arcface(embeddings)
        
#     def prepare_sample(self, wav):
#         import torchaudio.functional as F
        
#         # Ensure wav is on the same device as model
#         device = next(self.parameters()).device
#         wav = wav.to(device)
        
#         # If 2D, assume [batch_size, sequence_length]
#         if len(wav.shape) == 2:
#             # Resample if needed
#             if hasattr(wav, 'sample_rate') and wav.sample_rate != SAMPLE_RATE:
#                 wav = F.resample(wav, wav.sample_rate, SAMPLE_RATE)
                
#             # Adjust length to fixed duration
#             if wav.shape[1] > FIXED_SAMPLES:
#                 wav = wav[:, :FIXED_SAMPLES]  # Truncate
#             elif wav.shape[1] < FIXED_SAMPLES:
#                 wav = F.pad(wav, (0, FIXED_SAMPLES - wav.shape[1]))  # Pad
        
#         # If 1D, assume [sequence_length] and add batch dimension
#         else:
#             # Resample if needed
#             if hasattr(wav, 'sample_rate') and wav.sample_rate != SAMPLE_RATE:
#                 wav = F.resample(wav, wav.sample_rate, SAMPLE_RATE)
                
#             # Adjust length to fixed duration
#             if wav.shape[0] > FIXED_SAMPLES:
#                 wav = wav[:FIXED_SAMPLES]  # Truncate
#             elif wav.shape[0] < FIXED_SAMPLES:
#                 wav = F.pad(wav, (0, FIXED_SAMPLES - wav.shape[0]))  # Pad
            
#             # Add batch dimension
#             wav = wav.unsqueeze(0)
            
#         return wav