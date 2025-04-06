# Speaker Identification with WavLM-LoRA and ArcFace

This repository contains code for speaker identification and verification using a WavLM base model with LoRA (Low-Rank Adaptation) fine-tuning and ArcFace loss. The system is trained on the VoxCeleb2 dataset and evaluated on both VoxCeleb1 and VoxCeleb2. Th second model is novel pipelin approach to combine the speaker identification modelHuBERT Base model along with the SepFromer model to perform speaker separation with the speaker identification model and speech enhancement with the SepFormer model. Finetune/Train this new pipeline on the train set of created multi-speaker scenario dataset create by mixing of VOX2 datapoints to perform speech enhancement of each speaker in the multi-speaker dataset. 
 

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
| Value  | 3.82| 19.34| 5.19|  1.22|

### Embedding Quality

| Model | Embedding Identification Accuracy | Classification Identification Accuracy |
|-------|-----------------------------------|----------------------------------------|
| Pre-trained WavLM | 17.73% | - |
| Fine-tuned WavLM | 38.50% | 6.83% |

## Installation

```bash
# Clone this repository
git clone https://github.com/mitesh-kr/Speech_Understanding_Assignment_2.git 
cd Speech_Understanding_Assignment_2

# Install requirements
pip install -r requirements.txt

For seperator + classifer
git clone https://github.com/speechbrain/speechbrain/
cd speechbrain
pip install -r requirements.txt
pip install -e .

cd recipes/WHAMandWHAMR/
# Download and replace the separator folder
# [Download Link](https://drive.google.com/drive/folders/1kOLrP6sj_MnJ1IKjxr9OxgVxOaMUhLXd?usp=sharing)
Nothe that all training, evaluation , wavlm classifer and config file should be in same directory.




```

## Usage

### Data Preparation

1. Download VoxCeleb1 dataset: [VoxCeleb1](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox1.html)
2. Download VoxCeleb2 dataset: [VoxCeleb2](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox2.html)
3. Update the paths in `config.py` to point to your datasets

### Training

```bash
# Fine-tune WavLM with LoRA on VoxCeleb2
python train_speaker_identifier.py --batch_size 8 --epochs 20 --learning_rate 1e-4

#Run trainng on sepeartor + classifer pipeline
python QUESTION_1/4/train_seperator_identifier.py hparams/sepformer-wavlm-speaker.yaml --data_folder /path/to/data --train_data /path/to/train.csv

```

### Audio Mixing


```bash
python create_metadata.py --vox2_dir /path/to/vox2 --out_dir /path/to/output

VoxCeleb2 Train
python create_mixtures.py --metadata_file /path/to/output/train_voxceleb2_metadata.csv --output_dir /path/to/train_mixtures --n_mix 1000

VoxCeleb2 Test
python create_mixtures.py --metadata_file /path/to/output/test_voxceleb2_metadata.csv --output_dir /path/to/test_mixtures --n_mix 500


```

### Evaluation

```bash
# Evaluate on VoxCeleb1
python evaluate_speaker_identifier.py --model_path path/to/checkpoint --dataset voxceleb1

# Evaluate on VoxCeleb2
python evaluate_speaker_identifier.py --model_path path/to/checkpoint --dataset voxceleb2

# Run the evaluation on test speakers (51-100)
python evaluate_separation.py --vox2_dir /path/to/vox2mixed --model_path checkpoints/best_model.pt --output_dir separation_results --num_mixtures 100 --train_eval test

#Run sepeartion of mixed audios on test data
python QUESTION_1/4/evaluation.py hparams/sepformer-wavlm-speaker.yaml --test_data /path/to/test.csv
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
├── train_speaker_identifier.py                   # Training script
├── evaluate_speaker_identifier.py
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
