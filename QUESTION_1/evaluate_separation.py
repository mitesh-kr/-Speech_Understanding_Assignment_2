"""
Script for mixing speakers, performing separation, and evaluating quality.
"""
import os
import argparse
import random
import numpy as np
import torch
import torchaudio
from speechbrain.inference.separation import SepformerSeparation
from transformers import AutoFeatureExtractor

from config import DEVICE, SAMPLE_RATE
from models import create_model
from utils.audio import process_audio_fixed
from utils.metrics import calculate_audio_quality_metrics
from datasets.voxceleb2 import VoxCeleb2Dataset, collate_fn

def parse_args():
    parser = argparse.ArgumentParser(description="Mix speakers, separate, and evaluate")
    parser.add_argument('--vox2_dir', type=str, required=True, help='Path to VoxCeleb2 dataset')
    parser.add_argument('--model_path', type=str, required=True, help='Path to speaker identification model')
    parser.add_argument('--output_dir', type=str, default='separation_results', help='Output directory')
    parser.add_argument('--num_mixtures', type=int, default=100, help='Number of mixtures to create')
    parser.add_argument('--train_eval', type=str, choices=['train', 'test'], default='test', 
                       help='Use training or testing speakers')
    return parser.parse_args()

def get_speaker_files(root_dir, start_idx=0, end_idx=50):
    """Get audio files for a range of speakers."""
    all_speakers = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
    selected_speakers = all_speakers[start_idx:end_idx]
    
    speaker_files = {}
    for speaker in selected_speakers:
        speaker_dir = os.path.join(root_dir, speaker)
        files = []
        for root, _, filenames in os.walk(speaker_dir):
            for filename in filenames:
                if filename.endswith('.wav') or filename.endswith('.m4a'):
                    files.append(os.path.join(root, filename))
        if files:
            speaker_files[speaker] = files
    
    return speaker_files

def create_mixture(file1, file2, target_sr=16000, duration=5.0):
    """Create a mixture of two audio files with equal energy."""
    # Load audio files
    try:
        waveform1, sr1 = torchaudio.load(file1)
        waveform2, sr2 = torchaudio.load(file2)
    except Exception as e:
        print(f"Error loading audio: {e}")
        return None, None, None
    
    # Convert to mono and resample if needed
    if waveform1.shape[0] > 1:
        waveform1 = torch.mean(waveform1, dim=0, keepdim=True)
    if waveform2.shape[0] > 1:
        waveform2 = torch.mean(waveform2, dim=0, keepdim=True)
    
    if sr1 != target_sr:
        waveform1 = torchaudio.transforms.Resample(sr1, target_sr)(waveform1)
    if sr2 != target_sr:
        waveform2 = torchaudio.transforms.Resample(sr2, target_sr)(waveform2)
    
    # Limit to specified duration
    max_samples = int(target_sr * duration)
    waveform1 = waveform1[:, :min(waveform1.shape[1], max_samples)]
    waveform2 = waveform2[:, :min(waveform2.shape[1], max_samples)]
    
    # Normalize energy
    waveform1 = waveform1 / torch.sqrt(torch.sum(waveform1**2))
    waveform2 = waveform2 / torch.sqrt(torch.sum(waveform2**2))
    
    # Create common length
    min_len = min(waveform1.shape[1], waveform2.shape[1])
    waveform1 = waveform1[:, :min_len]
    waveform2 = waveform2[:, :min_len]
    
    # Mix with equal energy
    mixture = waveform1 + waveform2
    
    return mixture, waveform1, waveform2

def identify_speaker(model, waveform, processor, speaker_embeddings, speaker_ids):
    """Identify speaker using the given model."""
    model.eval()
    
    # Process audio
    waveform = process_audio_fixed(waveform, SAMPLE_RATE)
    
    # Extract features
    inputs = processor(waveform.squeeze(0).numpy(), sampling_rate=SAMPLE_RATE, return_tensors="pt")
    input_values = inputs.input_values.to(DEVICE)
    
    # Get speaker embedding
    with torch.no_grad():
        embedding = model(input_values)
    
    # Compare with known speaker embeddings
    embedding = torch.nn.functional.normalize(embedding, dim=1)
    similarities = []
    
    for spk_emb in speaker_embeddings:
        spk_emb = torch.nn.functional.normalize(spk_emb, dim=0)
        similarity = torch.sum(embedding * spk_emb, dim=1).item()
        similarities.append(similarity)
    
    # Get the most similar speaker
    predicted_idx = np.argmax(similarities)
    return speaker_ids[predicted_idx], similarities[predicted_idx]

def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Determine speaker range
    start_idx = 0 if args.train_eval == 'train' else 50
    end_idx = 50 if args.train_eval == 'train' else 100
    
    # Get speaker files
    print(f"Getting files for speakers {start_idx}-{end_idx}...")
    speaker_files = get_speaker_files(args.vox2_dir, start_idx, end_idx)
    speaker_ids = list(speaker_files.keys())
    print(f"Found {len(speaker_ids)} speakers with audio files")
    
    # Initialize models
    print("Loading SepFormer model...")
    separator = SepformerSeparation.from_hparams(
        source="speechbrain/sepformer-whamr",
        savedir="pretrained_models/sepformer-whamr"
    ).to(DEVICE)
    
    print("Loading speaker identification model...")
    processor = AutoFeatureExtractor.from_pretrained("microsoft/wavlm-base-plus")
    speaker_model = create_model()
    
    # Load checkpoint
    checkpoint = torch.load(args.model_path, map_location=DEVICE)
    if 'model_state_dict' in checkpoint:
        speaker_model.load_state_dict(checkpoint['model_state_dict'])
    else:
        speaker_model.load_state_dict(checkpoint)
    
    speaker_model = speaker_model.to(DEVICE)
    
    # Compute speaker embeddings for all speakers
    print("Computing speaker embeddings...")
    speaker_embeddings = []
    for speaker in speaker_ids:
        # Use the first file for each speaker
        file_path = speaker_files[speaker][0]
        waveform, sr = torchaudio.load(file_path)
        waveform = process_audio_fixed(waveform, sr)
        
        # Extract features
        inputs = processor(waveform.squeeze(0).numpy(), sampling_rate=SAMPLE_RATE, return_tensors="pt")
        input_values = inputs.input_values.to(DEVICE)
        
        # Get speaker embedding
        speaker_model.eval()
        with torch.no_grad():
            embedding = speaker_model(input_values)
        
        speaker_embeddings.append(embedding.cpu())
    
    # Create mixtures and evaluate
    print(f"Creating {args.num_mixtures} mixtures and evaluating...")
    
    all_metrics = {
        "sdr": [],
        "sir": [],
        "sar": [],
        "pesq": [],
        "identification_correct": 0,
        "total_sources": 0
    }
    
    for i in range(args.num_mixtures):
        # Select two random speakers
        speaker1, speaker2 = random.sample(speaker_ids, 2)
        
        # Select random files for each speaker
        file1 = random.choice(speaker_files[speaker1])
        file2 = random.choice(speaker_files[speaker2])
        
        # Create mixture
        mixture, source1, source2 = create_mixture(file1, file2)
        if mixture is None:
            continue
        
        # Save mixture and sources
        mix_path = os.path.join(args.output_dir, f"mixture_{i:03d}.wav")
        src1_path = os.path.join(args.output_dir, f"source1_{i:03d}_{speaker1}.wav")
        src2_path = os.path.join(args.output_dir, f"source2_{i:03d}_{speaker2}.wav")
        
        torchaudio.save(mix_path, mixture, SAMPLE_RATE)
        torchaudio.save(src1_path, source1, SAMPLE_RATE)
        torchaudio.save(src2_path, source2, SAMPLE_RATE)
        
        # Separate using SepFormer
        with torch.no_grad():
            est_sources = separator.separate_file(path=mix_path)
        
        # Save separated sources
        sep1_path = os.path.join(args.output_dir, f"separated1_{i:03d}.wav")
        sep2_path = os.path.join(args.output_dir, f"separated2_{i:03d}.wav")
        
        sep1 = est_sources[:, :, 0]
        sep2 = est_sources[:, :, 1]
        
        torchaudio.save(sep1_path, sep1.cpu(), SAMPLE_RATE)
        torchaudio.save(sep2_path, sep2.cpu(), SAMPLE_RATE)
        
        # Calculate audio quality metrics
        metrics = calculate_audio_quality_metrics(
            [source1, source2],
            [sep1.cpu(), sep2.cpu()]
        )
        
        # Accumulate metrics
        for key in ["sdr", "sir", "sar", "pesq"]:
            all_metrics[key].append(metrics[key])
        
        # Identify speakers in separated sources
        pred_spk1, _ = identify_speaker(speaker_model, sep1.cpu(), processor, speaker_embeddings, speaker_ids)
        pred_spk2, _ = identify_speaker(speaker_model, sep2.cpu(), processor, speaker_embeddings, speaker_ids)
        
        # Check if predictions are correct (allowing for any order)
        true_speakers = {speaker1, speaker2}
        pred_speakers = {pred_spk1, pred_spk2}
        
        if pred_spk1 in true_speakers:
            all_metrics["identification_correct"] += 1
        
        if pred_spk2 in true_speakers:
            all_metrics["identification_correct"] += 1
        
        all_metrics["total_sources"] += 2
        
        print(f"Mixture {i+1}/{args.num_mixtures} - SDR: {metrics['sdr']:.2f}, SIR: {metrics['sir']:.2f}, " 
              f"SAR: {metrics['sar']:.2f}, PESQ: {metrics['pesq']:.2f}")
        print(f"True speakers: {speaker1}, {speaker2} - Predicted: {pred_spk1}, {pred_spk2}")
    
    # Calculate average metrics
    avg_metrics = {
        "sdr": np.mean(all_metrics["sdr"]),
        "sir": np.mean(all_metrics["sir"]),
        "sar": np.mean(all_metrics["sar"]),
        "pesq": np.mean(all_metrics["pesq"]),
        "identification_accuracy": all_metrics["identification_correct"] / all_metrics["total_sources"] * 100
    }
    
    # Print results
    print("\n=== Evaluation Results ===")
    print(f"SDR: {avg_metrics['sdr']:.2f}")
    print(f"SIR: {avg_metrics['sir']:.2f}")
    print(f"SAR: {avg_metrics['sar']:.2f}")
    print(f"PESQ: {avg_metrics['pesq']:.2f}")
    print(f"Speaker Identification Accuracy: {avg_metrics['identification_accuracy']:.2f}%")
    
    # Save results
    results_path = os.path.join(args.output_dir, "separation_results.txt")
    with open(results_path, "w") as f:
        f.write("=== Evaluation Results ===\n")
        f.write(f"Metric  Value\n")
        f.write(f"SDR     {avg_metrics['sdr']:.2f}\n")
        f.write(f"SIR     {avg_metrics['sir']:.2f}\n")
        f.write(f"SAR     {avg_metrics['sar']:.2f}\n")
        f.write(f"PESQ    {avg_metrics['pesq']:.2f}\n")
        f.write(f"Identification Accuracy  {avg_metrics['identification_accuracy']:.2f}%\n")

if __name__ == "__main__":
    args = parse_args()
    main()