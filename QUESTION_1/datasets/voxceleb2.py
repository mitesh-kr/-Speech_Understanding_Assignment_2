"""
Dataset loaders for VoxCeleb2.
"""
import os
import random
import torch
import torchaudio
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from config import SAMPLE_RATE, FIXED_DURATION, FIXED_SAMPLES

class VoxCeleb2Dataset(Dataset):
    """
    Dataset for VoxCeleb2 with fixed-length audio segments.
    
    Args:
        root_dir: path to VoxCeleb2 audio files
        feature_extractor: feature extractor for audio preprocessing
        max_duration: maximum duration of audio segments in seconds
        split: "train" for the first 100 speakers, "val" for the next 18 speakers
    """
    def __init__(self, root_dir, feature_extractor, max_duration=FIXED_DURATION, split="train"):
        self.root_dir = root_dir
        self.feature_extractor = feature_extractor
        self.max_duration = max_duration
        self.sample_rate = SAMPLE_RATE
        self.max_samples = int(self.sample_rate * self.max_duration)
        self.split = split.lower()

        # Determine speakers based on split
        all_speakers = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
        if self.split == "train":
            self.speaker_ids = all_speakers[:100]  # First 100 speakers
        elif self.split == "val":
            self.speaker_ids = all_speakers[100:118]  # Next 18 speakers
        else:
            raise ValueError("split must be 'train' or 'val'.")

        print(f"{'Training' if self.split == 'train' else 'Validation'} on {len(self.speaker_ids)} speakers from VoxCeleb2.")

        self.id_to_label = {spk: i for i, spk in enumerate(self.speaker_ids)}
        self.samples = []
        
        # Collect all speaker samples
        for spk in self.speaker_ids:
            spk_dir = os.path.join(root_dir, spk)
            for root, _, files in os.walk(spk_dir):
                for file in files:
                    if file.lower().endswith(".wav"):
                        self.samples.append({
                            'path': os.path.join(root, file),
                            'speaker_id': spk,
                            'label': self.id_to_label[spk]
                        })
        
        print(f"Loaded {len(self.samples)} samples from VoxCeleb2 {self.split} set.")

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        waveform, sr = torchaudio.load(sample['path'])

        # Convert stereo to mono if needed
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # Resample if necessary
        if sr != self.sample_rate:
            waveform = torchaudio.transforms.Resample(sr, self.sample_rate)(waveform)

        # Crop or pad the waveform
        if waveform.shape[1] > self.max_samples:
            if self.split == "train":
                # Random crop for training
                start = random.randint(0, waveform.shape[1] - self.max_samples)
            else:
                # Center crop for validation
                start = (waveform.shape[1] - self.max_samples) // 2
            waveform = waveform[:, start:start+self.max_samples]
        elif waveform.shape[1] < self.max_samples:
            # Pad with zeros if too short
            pad_len = self.max_samples - waveform.shape[1]
            waveform = F.pad(waveform, (0, pad_len))

        # Extract features
        inputs = self.feature_extractor(
            waveform.squeeze(0).numpy(),
            sampling_rate=self.sample_rate,
            return_tensors="pt"
        )

        return {
            "input_values": inputs.input_values.squeeze(0),
            "label": torch.tensor(sample["label"], dtype=torch.long)
        }


def collate_fn(batch):
    """
    Collate function for VoxCeleb2 datasets.
    """
    input_values = [item["input_values"] for item in batch]
    labels = [item["label"] for item in batch]
    return {
        "input_values": torch.stack(input_values), 
        "label": torch.tensor(labels)
    }
