"""
Evaluation script for speaker identification model.
"""
import os
import argparse
import numpy as np
import torch
from transformers import AutoFeatureExtractor
import matplotlib.pyplot as plt

# Import modules
from config import VOX2_WAV_DIR, VOX1_DIR, TRIAL_PAIRS_PATH, DEVICE
from models import create_model
from datasets import (
    VoxCeleb1TrialDataset, 
    VoxCeleb1IdentificationDataset, 
    VoxCeleb2Dataset, 
    collate_fn, 
    collate_identification
)
from metrics import (
    evaluate_verification,
    evaluate_identification,
    evaluate_separation,
    print_separation_results
)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate speaker identification model")
    parser.add_argument('--model_path', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--dataset', type=str, default='both', choices=['voxceleb1', 'voxceleb2', 'both'], 
                        help='Dataset to evaluate on')
    parser.add_argument('--output_dir', type=str, default='results', help='Directory to save results')
    parser.add_argument('--separation_dir', type=str, default=None, help='Directory containing separated audio files')
    parser.add_argument('--plot_roc', action='store_true', help='Plot ROC curve for verification')
    return parser.parse_args()

def plot_roc_curve(labels, scores, output_path):
    """
    Plot ROC curve for verification results.
    
    Args:
        labels: ground truth binary labels
        scores: similarity scores
        output_path: path to save the ROC curve plot
    """
    from sklearn.metrics import roc_curve, auc
    
    fpr, tpr, _ = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    
    print(f"ROC curve saved to {output_path}")

def collect_scores_and_labels(model, dataset, device):
    """
    Collect similarity scores and labels from verification dataset.
    
    Args:
        model: speaker model
        dataset: trial pairs dataset
        device: computation device
        
    Returns:
        scores: similarity scores
        labels: ground truth labels
    """
    loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=False)
    
    all_scores = []
    all_labels = []
    
    with torch.no_grad():
        for batch in loader:
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
            emb1 = torch.nn.functional.normalize(emb1, dim=1)
            emb2 = torch.nn.functional.normalize(emb2, dim=1)
            scores = torch.sum(emb1 * emb2, dim=1).cpu().numpy()
            
            all_scores.extend(scores)
            all_labels.extend(labels)
    
    return np.array(all_scores), np.array(all_labels)

def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Using device: {DEVICE}")
    
    # Initialize feature extractor
    processor = AutoFeatureExtractor.from_pretrained("microsoft/wavlm-base-plus")
    
    # Initialize model
    print("Loading model...")
    model = create_model()
    
    # Load checkpoint
    checkpoint = torch.load(args.model_path, map_location=DEVICE)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(DEVICE)
    model.eval()
    
    # Evaluate on VoxCeleb1
    if args.dataset in ['voxceleb1', 'both']:
        print("\n=== Evaluating on VoxCeleb1 ===")
        
        # Load VoxCeleb1 datasets
        vox1_trial_dataset = VoxCeleb1TrialDataset(TRIAL_PAIRS_PATH, VOX1_DIR, processor)
        vox1_dataset = VoxCeleb1IdentificationDataset(VOX1_DIR, processor)
        
        # Verification
        print("Evaluating verification performance...")
        vox1_verification_metrics = evaluate_verification(model, vox1_trial_dataset, DEVICE)
        
        print(f"VoxCeleb1 EER: {vox1_verification_metrics['eer']:.2f}%")
        print(f"VoxCeleb1 TAR@1%FAR: {vox1_verification_metrics['tar_at_far']:.2f}%")
        print(f"VoxCeleb1 Verification Accuracy: {vox1_verification_metrics['accuracy']:.2f}%")
        
        # Plot ROC curve if requested
        if args.plot_roc:
            print("Collecting scores for ROC curve...")
            scores, labels = collect_scores_and_labels(model, vox1_trial_dataset, DEVICE)
            roc_path = os.path.join(args.output_dir, "verification_roc_curve.png")
            plot_roc_curve(labels, scores, roc_path)
        
        # Identification
        print("Evaluating identification performance...")
        vox1_identification_acc = evaluate_identification(
            model, vox1_dataset, DEVICE, collate_identification
        )
        
        print(f"VoxCeleb1 Identification Accuracy: {vox1_identification_acc:.2f}%")
        
        # Save results
        vox1_results = {
            "verification": {
                "eer": vox1_verification_metrics['eer'],
                "tar_at_far": vox1_verification_metrics['tar_at_far'],
                "accuracy": vox1_verification_metrics['accuracy'],
                "threshold": vox1_verification_metrics['threshold']
            },
            "identification": {
                "accuracy": vox1_identification_acc
            }
        }
        
        np.save(os.path.join(args.output_dir, "voxceleb1_results.npy"), vox1_results)
    
    # Evaluate on VoxCeleb2
    if args.dataset in ['voxceleb2', 'both']:
        print("\n=== Evaluating on VoxCeleb2 ===")
        
        # Load VoxCeleb2 dataset
        vox2_val_dataset = VoxCeleb2Dataset(
            root_dir=VOX2_WAV_DIR, 
            feature_extractor=processor, 
            split="val"
        )
        
        # Identification
        print("Evaluating identification performance...")
        vox2_identification_acc = evaluate_identification(
            model, vox2_val_dataset, DEVICE, collate_fn
        )
        
        print(f"VoxCeleb2 Identification Accuracy: {vox2_identification_acc:.2f}%")
        
        # Save results
        vox2_results = {
            "identification": {
                "accuracy": vox2_identification_acc
            }
        }
        
        np.save(os.path.join(args.output_dir, "voxceleb2_results.npy"), vox2_results)
    
    # Evaluate separation if directory is provided
    if args.separation_dir:
        print("\n=== Evaluating Audio Separation ===")
        print(f"Using separated audio files from: {args.separation_dir}")
        
        separation_metrics = evaluate_separation(args.separation_dir)
        print_separation_results(separation_metrics)
        
        # Save separation results
        np.save(os.path.join(args.output_dir, "separation_results.npy"), separation_metrics)
    
    # Print summary
    print("\n=== Evaluation Summary ===")
    if args.dataset in ['voxceleb1', 'both']:
        print("VoxCeleb1 Verification:")
        print(f"  - EER: {vox1_verification_metrics['eer']:.2f}%")
        print(f"  - TAR@1%FAR: {vox1_verification_metrics['tar_at_far']:.2f}%")
        print(f"  - Verification Accuracy: {vox1_verification_metrics['accuracy']:.2f}%")
        print("VoxCeleb1 Identification:")
        print(f"  - Identification Accuracy: {vox1_identification_acc:.2f}%")
    
    if args.dataset in ['voxceleb2', 'both']:
        print("VoxCeleb2 Identification:")
        print(f"  - Identification Accuracy: {vox2_identification_acc:.2f}%")
    
    if args.separation_dir:
        print("Audio Separation:")
        print(f"  - Speaker 1 SDR: {separation_metrics['speaker1']['sdr']:.2f}")
        print(f"  - Speaker 1 SIR: {separation_metrics['speaker1']['sir']:.2f}")
        print(f"  - Speaker 1 PESQ: {separation_metrics['speaker1']['pesq']:.2f}")
        print(f"  - Speaker 2 SDR: {separation_metrics['speaker2']['sdr']:.2f}")
        print(f"  - Speaker 2 SIR: {separation_metrics['speaker2']['sir']:.2f}")
        print(f"  - Speaker 2 PESQ: {separation_metrics['speaker2']['pesq']:.2f}")
    
    # Create comprehensive summary if both datasets are evaluated
    if args.dataset == 'both':
        summary = {
            "vox1_verification_eer": vox1_verification_metrics['eer'],
            "vox1_verification_tar": vox1_verification_metrics['tar_at_far'],
            "vox1_verification_acc": vox1_verification_metrics['accuracy'],
            "vox1_identification_acc": vox1_identification_acc,
            "vox2_identification_acc": vox2_identification_acc
        }
        
        if args.separation_dir:
            summary.update({
                "speaker1_sdr": separation_metrics['speaker1']['sdr'],
                "speaker1_sir": separation_metrics['speaker1']['sir'],
                "speaker1_sar": separation_metrics['speaker1']['sar'],
                "speaker1_pesq": separation_metrics['speaker1']['pesq'],
                "speaker2_sdr": separation_metrics['speaker2']['sdr'],
                "speaker2_sir": separation_metrics['speaker2']['sir'],
                "speaker2_sar": separation_metrics['speaker2']['sar'],
                "speaker2_pesq": separation_metrics['speaker2']['pesq']
            })
        
        np.save(os.path.join(args.output_dir, "evaluation_summary.npy"), summary)
        
        # Print as table
        print("\n=== Performance Table ===")
        print("| Metric | Value |")
        print("|--------|-------|")
        print(f"| VoxCeleb1 EER | {vox1_verification_metrics['eer']:.2f}% |")
        print(f"| VoxCeleb1 TAR@1%FAR | {vox1_verification_metrics['tar_at_far']:.2f}% |")
        print(f"| VoxCeleb1 Verification Accuracy | {vox1_verification_metrics['accuracy']:.2f}% |")
        print(f"| VoxCeleb1 Identification Accuracy | {vox1_identification_acc:.2f}% |")
        print(f"| VoxCeleb2 Identification Accuracy | {vox2_identification_acc:.2f}% |")
        
        if args.separation_dir:
            print(f"| Speaker 1 SDR | {separation_metrics['speaker1']['sdr']:.2f} |")
            print(f"| Speaker 1 SIR | {separation_metrics['speaker1']['sir']:.2f} |")
            print(f"| Speaker 1 SAR | {separation_metrics['speaker1']['sar']:.2f} |")
            print(f"| Speaker 1 PESQ | {separation_metrics['speaker1']['pesq']:.2f} |")
            print(f"| Speaker 2 SDR | {separation_metrics['speaker2']['sdr']:.2f} |")
            print(f"| Speaker 2 SIR | {separation_metrics['speaker2']['sir']:.2f} |")
            print(f"| Speaker 2 SAR | {separation_metrics['speaker2']['sar']:.2f} |")
            print(f"| Speaker 2 PESQ | {separation_metrics['speaker2']['pesq']:.2f} |")

if __name__ == "__main__":
    main()
