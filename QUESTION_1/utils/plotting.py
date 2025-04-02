"""
Plotting utilities for visualizing training progress and evaluation results.
"""
import numpy as np
import matplotlib.pyplot as plt
import os
from config import PLOTS_DIR

def plot_metrics(train_metrics, save_dir=PLOTS_DIR, prefix=""):
    """
    Plot training and evaluation metrics.
    
    Args:
        train_metrics: List of metrics for each epoch
        save_dir: Directory to save plots
        prefix: Prefix for saved files
    """
    # Make directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Convert metrics to numpy array for easier manipulation
    metrics = np.array(train_metrics)
    epochs = np.arange(1, len(metrics) + 1)
    
    # Define metrics to plot
    plots = [
        ('Training Loss', metrics[:, 0], 'blue', 'Loss', False),
        ('Training Accuracy', metrics[:, 1], 'green', 'Accuracy (%)', False),
        ('VoxCeleb1 EER', metrics[:, 2], 'red', 'EER (%)', True),
        ('VoxCeleb1 TAR@1%FAR', metrics[:, 3], 'purple', 'TAR (%)', False),
        ('VoxCeleb1 Verification Accuracy', metrics[:, 4], 'orange', 'Accuracy (%)', False),
        ('VoxCeleb1 Identification Accuracy', metrics[:, 5], 'brown', 'Accuracy (%)', False),
        ('VoxCeleb2 Identification Accuracy', metrics[:, 6], 'cyan', 'Accuracy (%)', False)
    ]
    
    # Create plots
    for title, values, color, y_label, invert_y_axis in plots:
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, values, marker='o', color=color, label=title)
        plt.title(title)
        plt.xlabel('Epochs')
        plt.ylabel(y_label)
        plt.grid(True)
        plt.legend()
        
        if invert_y_axis:  # For EER, lower is better
            plt.gca().invert_yaxis()
            
        # Save plot
        filename = f"{prefix}_{title.replace(' ', '_')}.png"
        plt.savefig(os.path.join(save_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()
    
    # Combined plot for verification metrics
    plt.figure(figsize=(12, 8))
    plt.plot(epochs, metrics[:, 2], marker='o', color='red', label='EER (%)')
    plt.plot(epochs, metrics[:, 3], marker='s', color='purple', label='TAR@1%FAR (%)')
    plt.plot(epochs, metrics[:, 4], marker='^', color='orange', label='Verification Acc (%)')
    plt.title('VoxCeleb1 Verification Metrics')
    plt.xlabel('Epochs')
    plt.ylabel('Value (%)')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(save_dir, f"{prefix}_verification_metrics.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Combined plot for identification metrics
    plt.figure(figsize=(12, 8))
    plt.plot(epochs, metrics[:, 5], marker='o', color='brown', label='VoxCeleb1 Identification Acc (%)')
    plt.plot(epochs, metrics[:, 6], marker='s', color='cyan', label='VoxCeleb2 Identification Acc (%)')
    plt.title('Speaker Identification Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(save_dir, f"{prefix}_identification_metrics.png"), dpi=300, bbox_inches='tight')
    plt.close()


def plot_roc_curve(labels, scores, save_path=None):
    """
    Plot ROC curve for verification performance.
    
    Args:
        labels: ground truth binary labels
        scores: similarity scores
        save_path: path to save the plot
    """
    from sklearn.metrics import roc_curve, auc
    
    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)
    
    # Find EER point
    fnr = 1 - tpr
    idx_eer = np.nanargmin(np.abs(fpr - fnr))
    eer = fpr[idx_eer]
    
    # Plot
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.plot(fpr[idx_eer], tpr[idx_eer], 'ro', markersize=8, label=f'EER = {eer:.3f}')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
    plt.close()
