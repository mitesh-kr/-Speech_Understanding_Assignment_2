from datasets.voxceleb1 import VoxCeleb1TrialDataset, VoxCeleb1IdentificationDataset, collate_identification
from datasets.voxceleb2 import VoxCeleb2Dataset, collate_fn

__all__ = [
    'VoxCeleb1TrialDataset',
    'VoxCeleb1IdentificationDataset',
    'VoxCeleb2Dataset',
    'collate_identification',
    'collate_fn'
]
