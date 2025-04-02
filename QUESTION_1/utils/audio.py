"""
Audio processing utilities.
"""
import torch
import torchaudio
import torch.nn.functional as F
from config import SAMPLE_RATE, FIXED_SAMPLES

def process_audio_fixed(waveform, sample_rate, target_sr=SAMPLE_RATE, max_samples=FIXED_SAMPLES):
    """
    Process audio to a fixed length for consistent input.
    
    Args:
        waveform: input audio waveform
        sample_rate: original sample rate
        target_sr: target sample rate
        max_samples: maximum number of samples
        
    Returns:
        Processed waveform with consistent length and sample rate
    """
    # Convert stereo to mono if needed
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    
    # Resample if necessary
    if sample_rate != target_sr:
        waveform = torchaudio.transforms.Resample(sample_rate, target_sr)(waveform)
    
    # Crop or pad to fixed length
    if waveform.shape[1] > max_samples:
        waveform = waveform[:, :max_samples]
    elif waveform.shape[1] < max_samples:
        pad_len = max_samples - waveform.shape[1]
        waveform = F.pad(waveform, (0, pad_len))
    
    return waveform


def augment_audio(waveform, sample_rate=SAMPLE_RATE):
    """
    Apply augmentations to audio for training robustness.
    
    Args:
        waveform: input audio waveform
        sample_rate: audio sample rate
        
    Returns:
        Augmented audio waveform
    """
    # Possible augmentations:
    # 1. Time stretching
    # 2. Adding noise
    # 3. Frequency masking
    # 4. Time masking
    
    # This is a placeholder - implement your desired augmentations
    return waveform
