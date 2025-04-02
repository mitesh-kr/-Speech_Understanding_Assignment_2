"""
Dataset loaders for VoxCeleb1.
"""
import os
import torch
import torchaudio
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from utils.audio import process_audio_fixed
from config import SAMPLE_RATE, FIXED_SAMPLES

class VoxCeleb1TrialDataset(Dataset):
    """
    Dataset for VoxCeleb1 verification trials.
    
    Args:
        trial_path: path to trial pairs file
        vox_dir: path to VoxCeleb1 audio files
        processor: feature extractor for audio preprocessing
    """
    def __init__(self, trial_path, vox_dir, processor):
        self.processor = processor
        self.vox_dir = vox_dir
        self.pairs = []
        
        with open(trial_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 3:
                    label = 1 if parts[0] == "1" else 0
                    file1 = os.path.join(vox_dir, parts[1])
                    file2 = os.path.join(vox_dir, parts[2])
                    if os.path.exists(file1) and os.path.exists(file2):
                        self.pairs.append((label, file1, file2))
        
        print(f"Loaded {len(self.pairs)} trial pairs from VoxCeleb1.")

    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        label, file1, file2 = self.pairs[idx]
        
        # Load and process audio files
        waveform1, sr1 = torchaudio.load(file1)
        waveform2, sr2 = torchaudio.load(file2)
        waveform1 = process_audio_fixed(waveform1, sr1)
        waveform2 = process_audio_fixed(waveform2, sr2)
        
        # Extract features
        inputs1 = self.processor(waveform1.squeeze(0).numpy(), sampling_rate=SAMPLE_RATE, return_tensors="pt")
        inputs2 = self.processor(waveform2.squeeze(0).numpy(), sampling_rate=SAMPLE_RATE, return_tensors="pt")
        
        return {
            "waveform1": inputs1.input_values.squeeze(0),
            "waveform2": inputs2.input_values.squeeze(0),
            "label": torch.tensor(label, dtype=torch.float32)
        }


class VoxCeleb1IdentificationDataset(Dataset):
    """
    Dataset for VoxCeleb1 speaker identification.
    
    Args:
        vox_dir: path to VoxCeleb1 audio files
        processor: feature extractor for audio preprocessing
    """
    def __init__(self, vox_dir, processor):
        self.processor = processor
        self.samples = []
        self.labels = []
        
        # Get all speaker IDs (folder names)
        self.speaker_ids = sorted([d for d in os.listdir(vox_dir) if os.path.isdir(os.path.join(vox_dir, d))])
        self.speaker2label = {spk: i for i, spk in enumerate(self.speaker_ids)}
        
        # Collect all audio files for each speaker
        for spk in self.speaker_ids:
            spk_dir = os.path.join(vox_dir, spk)
            for root, _, files in os.walk(spk_dir):
                for file in files:
                    if file.lower().endswith(".wav"):
                        self.samples.append(os.path.join(root, file))
                        self.labels.append(self.speaker2label[spk])
        
        print(f"Loaded {len(self.samples)} samples from {len(self.speaker_ids)} speakers for identification.")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        filepath = self.samples[idx]
        label = self.labels[idx]
        
        # Load and process audio
        waveform, sr = torchaudio.load(filepath)
        waveform = process_audio_fixed(waveform, sr)
        
        # Extract features
        inputs = self.processor(waveform.squeeze(0).numpy(), sampling_rate=SAMPLE_RATE, return_tensors="pt")
        
        return {
            "input_values": inputs.input_values.squeeze(0), 
            "label": label
        }


def collate_identification(batch):
    """
    Collate function for identification datasets.
    """
    input_values = [item["input_values"] for item in batch]
    labels = [item["label"] for item in batch]
    return {
        "input_values": torch.stack(input_values), 
        "label": torch.tensor(labels)
    }
