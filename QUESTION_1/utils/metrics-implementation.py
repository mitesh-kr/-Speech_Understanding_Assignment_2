"""
Evaluation metrics for speaker verification and identification.
"""
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, roc_curve
from scipy.interpolate import interp1d
from tqdm import tqdm
from torch.utils.data import DataLoader
import os
import re
from collections import defaultdict
import torchaudio
import torchaudio.transforms as T
import mir_eval
from pesq import pesq


def calculate_eer(labels, scores):
    """
    Calculate Equal Error Rate (EER) and threshold.
    
    Args:
        labels: ground truth binary labels (1 for same speaker, 0 for different)
        scores: similarity scores between pairs
        
    Returns:
        eer: equal error rate
        threshold: EER threshold
    """
    fpr, tpr, thresholds = roc_curve(labels, scores)
    fnr = 1 - tpr
    
    # Find the threshold where FPR = FNR (EER)
    idx_eer = np.nanargmin(np.abs(fpr - fnr))
    eer = fpr[idx_eer]
    
    return eer * 100, thresholds[idx_eer]


def calculate_tar_at_far(labels, scores, far_value=0.01):
    """
    Calculate True Accept Rate at a specific False Accept Rate.
    
    Args:
        labels: ground truth binary labels
        scores: similarity scores
        far_value: target false accept rate (default 1%)
        
    Returns:
        tar: true accept rate at the specified FAR
    """
    fpr, tpr, _ = roc_curve(labels, scores)
    
    # Interpolate to get TAR at the specific FAR
    interp_tpr = interp1d(fpr, tpr, kind="linear", fill_value="extrapolate")
    tar = interp_tpr(far_value)
    
    return tar * 100


def evaluate_verification(model, dataset, device, threshold=None):
    """
    Evaluate speaker verification performance.
    
    Args:
        model: speaker model
        dataset: trial pairs dataset
        device: computation device
        threshold: decision threshold (if None, use EER threshold)
        
    Returns:
        metrics: dictionary containing EER, TAR@1%FAR, and accuracy
    """
    model.eval()
    loader = DataLoader(dataset, batch_size=16, shuffle=False)
    
    all_scores = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating verification"):
            wav1 = batch["waveform1"].to(device)
            wav2 = batch["waveform2"].to(device)
            labels = batch["label"].cpu().numpy()
            
            # Extract embeddings
            emb1 = model(wav1)
            emb2 = model(wav2)
            
            # If output is from WavLM, take mean over time dimension
            if hasattr(emb1, 'last_hidden_state'):
                emb1 = torch.mean(emb1.last_hidden_state, dim=1)
                emb2 = torch.mean(emb2.last_hidden_state, dim=1)
            
            # Compute cosine similarity
            emb1 = F.normalize(emb1, dim=1)
            emb2 = F.normalize(emb2, dim=1)
            scores = torch.sum(emb1 * emb2, dim=1).cpu().numpy()
            
            all_scores.extend(scores)
            all_labels.extend(labels)
    
    all_scores = np.array(all_scores)
    all_labels = np.array(all_labels)
    
    # Calculate metrics
    eer, eer_threshold = calculate_eer(all_labels, all_scores)
    tar_at_far = calculate_tar_at_far(all_labels, all_scores, 0.01)
    
    # Use EER threshold if no threshold is provided
    if threshold is None:
        threshold = eer_threshold
        
    # Calculate verification accuracy
    predictions = (all_scores > threshold).astype(int)
    accuracy = accuracy_score(all_labels, predictions)
    
    return {
        "eer": eer,
        "tar_at_far": tar_at_far,
        "accuracy": accuracy * 100,
        "threshold": threshold
    }


def evaluate_identification(model, dataset, device, collate_fn=None):
    """
    Evaluate speaker identification performance.
    
    Args:
        model: speaker model
        dataset: identification dataset
        device: computation device
        collate_fn: collate function for dataloader
        
    Returns:
        accuracy: identification accuracy
    """
    model.eval()
    loader = DataLoader(dataset, batch_size=16, shuffle=False, collate_fn=collate_fn)
    
    all_embeddings = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Computing embeddings"):
            input_values = batch['input_values'].to(device)
            labels = batch['label']
            
            # Extract embeddings
            emb = model(input_values)
            
            # If output is from WavLM, take mean over time dimension
            if hasattr(emb, 'last_hidden_state'):
                emb = torch.mean(emb.last_hidden_state, dim=1)
            
            all_embeddings.append(emb)
            all_labels.extend(labels.numpy())
    
    all_embeddings = torch.cat(all_embeddings, dim=0)
    all_labels = np.array(all_labels)
    
    # Compute embeddings for each speaker (centroids)
    unique_labels = np.unique(all_labels)
    centroids = {}
    
    for label in unique_labels:
        speaker_embeddings = all_embeddings[all_labels == label]
        centroids[label] = torch.mean(speaker_embeddings, dim=0)
    
    # Predict speaker for each embedding using cosine similarity
    preds = []
    for i in range(all_embeddings.shape[0]):
        emb = F.normalize(all_embeddings[i], dim=0)
        
        # Compute similarity to all centroids
        similarities = []
        for label in unique_labels:
            centroid = F.normalize(centroids[label], dim=0)
            similarity = torch.dot(emb, centroid).item()
            similarities.append(similarity)
        
        # Predict the speaker with highest similarity
        pred_label = unique_labels[np.argmax(similarities)]
        preds.append(pred_label)
    
    # Calculate accuracy
    acc = accuracy_score(all_labels, preds)
    
    return acc * 100


def compute_sdr_sir_sar(reference, estimated, interference):
    """
    Compute Source-to-Distortion Ratio (SDR), Source-to-Interference Ratio (SIR), 
    and Source-to-Artifact Ratio (SAR).
    
    Args:
        reference: reference audio signal
        estimated: estimated audio signal
        interference: interference audio signal
        
    Returns:
        sdr: Source-to-Distortion Ratio
        sir: Source-to-Interference Ratio
        sar: Source-to-Artifact Ratio
    """
    reference = reference.squeeze().cpu().numpy()
    estimated = estimated.squeeze().cpu().numpy()
    interference = interference.squeeze().cpu().numpy()
    sources = np.vstack([reference, interference])
    estimated_sources = np.vstack([estimated, interference])
    sdr, sir, sar, _ = mir_eval.separation.bss_eval_sources(sources, estimated_sources)
    return sdr[0], sir[0], sar[0]


def compute_pesq(reference, estimated, sample_rate=8000):
    """
    Compute Perceptual Evaluation of Speech Quality (PESQ).
    
    Args:
        reference: reference audio signal
        estimated: estimated audio signal
        sample_rate: audio sample rate (default: 8000)
        
    Returns:
        pesq_score: PESQ score
    """
    reference = reference.squeeze().cpu().numpy()
    estimated = estimated.squeeze().cpu().numpy()
    return pesq(sample_rate, reference, estimated, 'nb')


def resample_audio(audio, orig_sr, target_sr=8000):
    """
    Resample audio to target sample rate.
    
    Args:
        audio: audio tensor
        orig_sr: original sample rate
        target_sr: target sample rate (default: 8000)
        
    Returns:
        resampled_audio: audio at target sample rate
    """
    if orig_sr != target_sr:
        return T.Resample(orig_freq=orig_sr, new_freq=target_sr)(audio)
    return audio


def match_audio_length(audio, target_length):
    """
    Match audio length to target length by padding or truncating.
    
    Args:
        audio: audio tensor
        target_length: target length in samples
        
    Returns:
        matched_audio: audio with target length
    """
    if audio.dim() == 1:
        audio = audio.unsqueeze(0)
    if audio.shape[1] > target_length:
        audio = audio[:, :target_length]
    elif audio.shape[1] < target_length:
        padding = torch.zeros(audio.shape[0], target_length - audio.shape[1], device=audio.device)
        audio = torch.cat([audio, padding], dim=1)
    return audio


def get_speaker_path(speaker):
    """
    Get file path for a speaker ID.
    
    Args:
        speaker: speaker identifier
        
    Returns:
        path: path to speaker audio file
    """
    parts = speaker.split('_')
    if len(parts) >= 3:
        return f"/iitjhome/m23mac003/VOX2/{parts[0]}/{parts[1]}/{parts[2]}.wav"
    return None


def extract_speaker_and_fileno(filename):
    """
    Extract speaker ID and file number from filename.
    
    Args:
        filename: audio filename
        
    Returns:
        speaker: speaker identifier
        fileno: file number
    """
    pattern = r"(id\d+_[^_]+_\d+)_(\d+)\.wav"
    match = re.match(pattern, filename)
    if match:
        return match.groups()
    raise ValueError(f"Filename format does not match expected pattern: {filename}")


def evaluate_separation(separated_dir):
    """
    Evaluate audio source separation performance.
    
    Args:
        separated_dir: directory containing separated audio files
        
    Returns:
        metrics: dictionary containing evaluation metrics for both speakers
    """
    pattern = r"(id\d+_[^_]+_\d+)_(\d+)\.wav"
    file_groups = defaultdict(list)
    
    # Group files by file number
    for file_name in os.listdir(separated_dir):
        match = re.match(pattern, file_name)
        if match:
            speaker, fileno = match.groups()
            file_groups[fileno].append(file_name)

    # Keep only file numbers with exactly 2 files (speaker pairs)
    paired_files = {fileno: files for fileno, files in file_groups.items() if len(files) == 2}
    sorted_paired_files = dict(sorted(paired_files.items(), key=lambda x: int(x[0])))

    # Initialize accumulators
    total_sdr1, total_sir1, total_sar1, total_pesq1 = 0, 0, 0, 0
    total_sdr2, total_sir2, total_sar2, total_pesq2 = 0, 0, 0, 0
    count = 0

    for fileno, files in sorted_paired_files.items():
        sorted_files = sorted(files)
        
        # First estimated file
        estimated_file_name = sorted_files[0]
        estimated_path = os.path.join(separated_dir, estimated_file_name)
        estimate, _ = torchaudio.load(estimated_path)
        
        speaker = extract_speaker_and_fileno(estimated_file_name)[0]
        original_audio_path = get_speaker_path(speaker)
        original_audio, sr = torchaudio.load(original_audio_path)
        reference = resample_audio(original_audio, sr, 8000)
        reference = match_audio_length(reference, estimate.shape[1])
        
        # Second estimated file (Interference)
        interference_file_name = sorted_files[1]
        speaker = extract_speaker_and_fileno(interference_file_name)[0]
        original_audio_path = get_speaker_path(speaker)
        original_audio, sr = torchaudio.load(original_audio_path)
        interference = resample_audio(original_audio, sr, 8000)
        interference = match_audio_length(interference, estimate.shape[1])
        
        # Compute SDR, SIR, SAR, PESQ for Speaker 1
        sdr1, sir1, sar1 = compute_sdr_sir_sar(reference, estimate, interference)
        pesq1 = compute_pesq(reference, estimate, sample_rate=8000)
        total_sdr1 += sdr1
        total_sir1 += sir1
        total_sar1 += sar1
        total_pesq1 += pesq1
        
        # Compute SDR, SIR, SAR, PESQ for Speaker 2
        estimated_file_name = sorted_files[1]
        estimated_path = os.path.join(separated_dir, estimated_file_name)
        estimate, _ = torchaudio.load(estimated_path)
        sdr2, sir2, sar2 = compute_sdr_sir_sar(interference, estimate, reference)
        pesq2 = compute_pesq(interference, estimate, sample_rate=8000)
        
        total_sdr2 += sdr2
        total_sir2 += sir2
        total_sar2 += sar2
        total_pesq2 += pesq2

        count += 1

    # Compute average metrics
    avg_metrics = {
        "speaker1": {
            "sdr": total_sdr1 / count,
            "sir": total_sir1 / count,
            "sar": total_sar1 / count,
            "pesq": total_pesq1 / count
        },
        "speaker2": {
            "sdr": total_sdr2 / count,
            "sir": total_sir2 / count,
            "sar": total_sar2 / count,
            "pesq": total_pesq2 / count
        }
    }
    
    return avg_metrics


def print_separation_results(metrics):
    """
    Print the separation evaluation results.
    
    Args:
        metrics: dictionary containing evaluation metrics
    """
    print(f"\nAverage Metrics for Speaker 1:")
    print(f"SDR: {metrics['speaker1']['sdr']:.2f}, SIR: {metrics['speaker1']['sir']:.2f}, "
          f"SAR: {metrics['speaker1']['sar']:.2f}, PESQ: {metrics['speaker1']['pesq']:.2f}")

    print(f"\nAverage Metrics for Speaker 2:")
    print(f"SDR: {metrics['speaker2']['sdr']:.2f}, SIR: {metrics['speaker2']['sir']:.2f}, "
          f"SAR: {metrics['speaker2']['sar']:.2f}, PESQ: {metrics['speaker2']['pesq']:.2f}")


if __name__ == "__main__":
    # Example usage
    separated_dir = "/path/to/separated/audio/files"
    metrics = evaluate_separation(separated_dir)
    print_separation_results(metrics)
