"""
ArcFace loss implementation for speaker identification.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import ARCFACE_SCALE, ARCFACE_MARGIN

class ArcFaceLayer(nn.Module):
    """
    ArcFace: Additive Angular Margin Loss for Deep Face Recognition.
    
    Args:
        in_features: size of each input sample
        out_features: size of each output sample (number of speakers)
        scale: scale factor (s in the paper)
        margin: margin parameter (m in the paper)
    """
    def __init__(self, in_features, out_features, scale=ARCFACE_SCALE, margin=ARCFACE_MARGIN):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.scale = scale
        self.margin = margin
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, embeddings, labels=None):
        """
        Forward pass for ArcFace.
        
        Args:
            embeddings: normalized feature vectors
            labels: ground truth speaker labels (optional, for training)
            
        Returns:
            logits with the ArcFace margin if labels are provided,
            otherwise regular cosine similarity logits
        """
        # Normalize embeddings and weights
        embeddings = F.normalize(embeddings, dim=1)
        weights = F.normalize(self.weight, dim=1)
        
        # Compute cosine similarity (dot product of normalized vectors)
        cosine = F.linear(embeddings, weights)
        
        # If not in training mode or no labels provided, return cosine similarity
        if labels is None:
            return cosine * self.scale
            
        # Apply angular margin
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * math.cos(self.margin) - sine * math.sin(self.margin)
        
        # One-hot encoding for labels
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1), 1)
        
        # Apply margin only to the target class
        output = one_hot * phi + (1 - one_hot) * cosine
        
        # Scale and return
        return output * self.scale
