"""
Training script for speaker identification model using WavLM with LoRA adaptation.
"""
import os
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from transformers import AutoFeatureExtractor

# Import modules
from config import (
    VOX2_WAV_DIR, VOX1_DIR, TRIAL_PAIRS_PATH, DEVICE, BATCH_SIZE, 
    NUM_EPOCHS, LEARNING_RATE, CHECKPOINT_DIR, PLOTS_DIR,
    NUM_SPEAKERS, RANDOM_SEED
)
from models import create_model
from datasets import (
    VoxCeleb1TrialDataset, 
    VoxCeleb1IdentificationDataset, 
    VoxCeleb2Dataset, 
    collate_fn, 
    collate_identification
)
from utils import evaluate_verification, evaluate_identification, plot_metrics

def set_seed(seed):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train speaker identification model")
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE, help='Batch size')
    parser.add_argument('--epochs', type=int, default=NUM_EPOCHS, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=LEARNING_RATE, help='Learning rate')
    parser.add_argument('--seed', type=int, default=RANDOM_SEED, help='Random seed')
    parser.add_argument('--checkpoint_dir', type=str, default=CHECKPOINT_DIR, help='Checkpoint directory')
    parser.add_argument('--plots_dir', type=str, default=PLOTS_DIR, help='Plots directory')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    return parser.parse_args()

def train(args):
    """Main training function."""
    # Set random seed
    set_seed(args.seed)
    
    # Create directories
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.plots_dir, exist_ok=True)
    
    print(f"Using device: {DEVICE}")
    
    # Initialize feature extractor
    processor = AutoFeatureExtractor.from_pretrained("microsoft/wavlm-base-plus")
    
    # Load datasets
    print("Loading datasets...")
    vox2_train_dataset = VoxCeleb2Dataset(
        root_dir=VOX2_WAV_DIR, 
        feature_extractor=processor, 
        split="train"
    )
    vox2_val_dataset = VoxCeleb2Dataset(
        root_dir=VOX2_WAV_DIR, 
        feature_extractor=processor, 
        split="val"
    )
    vox1_trial_dataset = VoxCeleb1TrialDataset(
        TRIAL_PAIRS_PATH, 
        VOX1_DIR, 
        processor
    )
    vox1_dataset = VoxCeleb1IdentificationDataset(
        VOX1_DIR, 
        processor
    )
    
    # Create data loaders
    train_loader = DataLoader(
        vox2_train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        collate_fn=collate_fn
    )
    
    # Initialize model
    print("Creating model...")
    model = create_model(num_speakers=NUM_SPEAKERS)
    model = model.to(DEVICE)
    
    # If resume from checkpoint
    start_epoch = 0
    if args.resume:
        print(f"Loading checkpoint from {args.resume}")
        checkpoint = torch.load(args.resume, map_location=DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint.get('epoch', 0) + 1
    
    # Initialize optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()
    
    # Evaluation before training
    print("\n=== Baseline Evaluation (Pre-trained Model) ===")
    vox1_verification_metrics = evaluate_verification(model, vox1_trial_dataset, DEVICE)
    vox1_identification_acc = evaluate_identification(
        model, vox1_dataset, DEVICE, collate_identification
    )
    vox2_identification_acc = evaluate_identification(
        model, vox2_val_dataset, DEVICE, collate_fn
    )
    
    print(f"Baseline VoxCeleb1 EER: {vox1_verification_metrics['eer']:.2f}%")
    print(f"Baseline VoxCeleb1 TAR@1%FAR: {vox1_verification_metrics['tar_at_far']:.2f}%")
    print(f"Baseline VoxCeleb1 Verification Acc: {vox1_verification_metrics['accuracy']:.2f}%")
    print(f"Baseline VoxCeleb1 Identification Acc: {vox1_identification_acc:.2f}%")
    print(f"Baseline VoxCeleb2 Identification Acc: {vox2_identification_acc:.2f}%\n")
    
    # Training loop
    print("=== Starting Training ===")
    all_metrics = []
    
    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_loss = 0
        train_preds = []
        train_labels = []
        
        # Training
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}"):
            input_values = batch['input_values'].to(DEVICE)
            labels = batch['label'].to(DEVICE)
            
            optimizer.zero_grad()
            
            # Forward pass
            logits, _ = model(input_values, labels)
            loss = criterion(logits, labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Track metrics
            epoch_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            train_preds.extend(preds.cpu().numpy())
            train_labels.extend(labels.cpu().numpy())
        
        # Update learning rate
        scheduler.step()
        
        # Calculate training metrics
        from sklearn.metrics import accuracy_score
        train_acc = accuracy_score(train_labels, train_preds)
        avg_loss = epoch_loss / len(train_loader)
        
        # Evaluate on validation
        print("\nEvaluating...")
        vox1_verification_metrics = evaluate_verification(model, vox1_trial_dataset, DEVICE)
        vox1_identification_acc = evaluate_identification(
            model, vox1_dataset, DEVICE, collate_identification
        )
        vox2_identification_acc = evaluate_identification(
            model, vox2_val_dataset, DEVICE, collate_fn
        )
        
        # Print metrics
        print(f"\nEpoch {epoch+1}/{args.epochs}:")
        print(f"  Train Loss: {avg_loss:.4f}")
        print(f"  Train Accuracy: {train_acc*100:.2f}%")
        print(f"  VoxCeleb1 Verification:")
        print(f"    - EER: {vox1_verification_metrics['eer']:.2f}%")
        print(f"    - TAR@1%FAR: {vox1_verification_metrics['tar_at_far']:.2f}%")
        print(f"    - Verification Acc: {vox1_verification_metrics['accuracy']:.2f}%")
        print(f"  VoxCeleb1 Identification:")
        print(f"    - Identification Acc: {vox1_identification_acc:.2f}%")
        print(f"  VoxCeleb2 Identification:")
        print(f"    - Identification Acc: {vox2_identification_acc:.2f}%")
        
        # Record metrics
        epoch_metrics = [
            avg_loss,
            train_acc * 100,
            vox1_verification_metrics['eer'],
            vox1_verification_metrics['tar_at_far'],
            vox1_verification_metrics['accuracy'],
            vox1_identification_acc,
            vox2_identification_acc
        ]
        all_metrics.append(epoch_metrics)
        
        # Save checkpoint
        checkpoint_path = os.path.join(args.checkpoint_dir, f"epoch_{epoch+1}.pt")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
        }, checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")
    
    # Save final metrics
    metrics_path = os.path.join(args.plots_dir, "training_metrics.npy")
    np.save(metrics_path, np.array(all_metrics))
    
    # Plot metrics
    plot_metrics(all_metrics, args.plots_dir)
    
    print("\n=== Training Complete ===")
    
    # Final evaluation
    print("\n=== Final Evaluation ===")
    vox1_verification_metrics = evaluate_verification(model, vox1_trial_dataset, DEVICE)
    vox1_identification_acc = evaluate_identification(
        model, vox1_dataset, DEVICE, collate_identification
    )
    vox2_identification_acc = evaluate_identification(
        model, vox2_val_dataset, DEVICE, collate_fn
    )
    
    print("VoxCeleb1 Verification:")
    print(f"  - EER: {vox1_verification_metrics['eer']:.2f}%")
    print(f"  - TAR@1%FAR: {vox1_verification_metrics['tar_at_far']:.2f}%")
    print(f"  - Verification Accuracy: {vox1_verification_metrics['accuracy']:.2f}%")
    print("VoxCeleb1 Identification:")
    print(f"  - Identification Accuracy: {vox1_identification_acc:.2f}%")
    print("VoxCeleb2 Identification:")
    print(f"  - Identification Accuracy: {vox2_identification_acc:.2f}%")

if __name__ == "__main__":
    args = parse_args()
    train(args)
