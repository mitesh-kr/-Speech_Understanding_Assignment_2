# Speaker Identification with WavLM-LoRA and ArcFace

This repository contains code for speaker identification and verification using WavLM base model with LoRA (Low-Rank Adaptation) fine-tuning and ArcFace loss. The system is trained on VoxCeleb2 dataset and evaluated on both VoxCeleb1 and VoxCeleb2.

## Features

- WavLM base model with LoRA adaptation for efficient fine-tuning
- ArcFace loss for improved speaker embedding discrimination
- Speaker verification and identification evaluation
- Comprehensive metrics including EER, TAR@1%FAR, and accuracy

## Results

### Performance Comparison

| Metric | Pre-trained Model | Fine-tuned Model |
|--------|-------------------|------------------|
| **VoxCeleb1 Verification** |                   |                  |
| EER (Equal Error Rate) | 44.32% | 16.00% |
| TAR@1%FAR | 2.12% | 31.85% |
| Verification Accuracy | 49.99% | 62.17% |
| **VoxCeleb1 Identification** |                   |                  |
| Identification Accuracy | 55.72% | 83.46% |
| **VoxCeleb2 Identification** |                   |                  |
| Identification Accuracy | 67.62% | 88.57% |

### Model Evaluation Metrics

| Metric | SDR | SIR | SAR | PESQ |
|--------|-----|-----|-----|------|
| Value  | 0.16| 11.92| 6.01| 1.80 |

### Embedding Quality

| Model | Embedding Identification Accuracy | Classification Identification Accuracy |
|-------|-----------------------------------|----------------------------------------|
| Pre-trained WavLM | 17.73% | - |
| Fine-tuned WavLM | 38.50% | 6.83% |

## Installation

```bash
# Clone this repository
git clone https://github.com/yourusername/speaker-id-wavlm-lora.git
cd speaker-id-wavlm-lora

# Create a conda environment
conda create -n speaker-id python=3.8
conda activate speaker-id

# Install requirements
pip install -r requirements.txt
```

## Usage

### Data Preparation

1. Download VoxCeleb1 dataset: [VoxCeleb1](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox1.html)
2. Download VoxCeleb2 dataset: [VoxCeleb2](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox2.html)
3. Update the paths in `config.py` to point to your datasets

### Training

```bash
# Fine-tune WavLM with LoRA on VoxCeleb2
python train.py --batch_size 8 --epochs 20 --learning_rate 1e-4
```

### Evaluation

```bash
# Evaluate on VoxCeleb1
python evaluate.py --model_path path/to/checkpoint --dataset voxceleb1

# Evaluate on VoxCeleb2
python evaluate.py --model_path path/to/checkpoint --dataset voxceleb2

# Run the evaluation on test speakers (51-100)
python evaluate_separation.py --vox2_dir /path/to/vox2mixed --model_path checkpoints/best_model.pt --output_dir separation_results --num_mixtures 100 --train_eval test
```
## Project Structure

```
.
├── config.py                  # Configuration parameters
├── datasets/
│   ├── __init__.py
│   ├── voxceleb1.py           # VoxCeleb1 dataset loader
│   └── voxceleb2.py           # VoxCeleb2 dataset loader
├── models/
│   ├── __init__.py
│   ├── arcface.py             # ArcFace loss implementation
│   └── wavlm_lora.py          # WavLM with LoRA implementation
├── utils/
│   ├── __init__.py
│   ├── audio.py               # Audio processing utilities
│   ├── metrics.py             # Evaluation metrics
│   └── plotting.py            # Plotting utilities
├── train.py                   # Training script
├── evaluate.py
├── evaluate_separation.py                 # Evaluation script
└── requirements.txt           # Dependencies
```

## Citation

If you use this code for your research, please cite:

```
@inproceedings{your-paper,
  title={Your Paper Title},
  author={Your Name and Coauthors},
  booktitle={Proceedings of ...},
  year={2023}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- [HuggingFace Transformers](https://github.com/huggingface/transformers) for providing WavLM implementation
- [PEFT](https://github.com/huggingface/peft) for LoRA implementation
- VoxCeleb datasets
