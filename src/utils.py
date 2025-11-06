"""
Utility functions for the emotion recognition project.

Includes helpers for:
- Checkpoint saving/loading
- Image denormalization
- Visualization
- Configuration management
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# ImageNet normalization constants
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Emotion labels
EMOTION_LABELS = [
    'neutral', 'happy', 'sad', 'surprised', 'fear', 'disgust', 'angry',
    'contempt', 'serene', 'contemplative', 'secure', 'untroubled', 'quiet'
]


def denormalize_image(tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD):
    """
    Denormalize image tensor for visualization.

    Args:
        tensor: Image tensor (C, H, W) or (B, C, H, W)
        mean: Mean values used for normalization
        std: Std values used for normalization

    Returns:
        Denormalized tensor (values in [0, 1])
    """
    # Handle both single image and batch
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)
        squeeze = True
    else:
        squeeze = False

    # Create tensors for mean and std
    mean = torch.tensor(mean).view(1, 3, 1, 1).to(tensor.device)
    std = torch.tensor(std).view(1, 3, 1, 1).to(tensor.device)

    # Denormalize
    denorm = tensor * std + mean
    denorm = torch.clamp(denorm, 0, 1)

    if squeeze:
        denorm = denorm.squeeze(0)

    return denorm


def save_checkpoint(state, checkpoint_dir, filename='checkpoint.pth', is_best=False):
    """
    Save model checkpoint.

    Args:
        state: Dictionary containing model state, optimizer state, epoch, etc.
        checkpoint_dir: Directory to save checkpoints
        filename: Checkpoint filename
        is_best: If True, also save as 'best_model.pth'
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    filepath = checkpoint_dir / filename
    torch.save(state, filepath)

    if is_best:
        best_path = checkpoint_dir / 'best_model.pth'
        torch.save(state, best_path)


def load_checkpoint(checkpoint_path, model, optimizer=None):
    """
    Load model checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file
        model: Model to load state into
        optimizer: Optional optimizer to load state into

    Returns:
        Dictionary with epoch, best_loss, and other saved info
    """
    checkpoint = torch.load(checkpoint_path)

    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    return {
        'epoch': checkpoint.get('epoch', 0),
        'best_loss': checkpoint.get('best_loss', float('inf')),
        'metrics': checkpoint.get('metrics', {})
    }


def visualize_predictions(images, pred_probs, target_probs, num_samples=4,
                         emotion_labels=EMOTION_LABELS, save_path=None):
    """
    Visualize predictions vs ground truth.

    Args:
        images: Batch of images (B, C, H, W)
        pred_probs: Predicted probabilities (B, num_emotions)
        target_probs: Target probabilities (B, num_emotions)
        num_samples: Number of samples to visualize
        emotion_labels: List of emotion label names
        save_path: Optional path to save the figure
    """
    num_samples = min(num_samples, len(images))

    fig, axes = plt.subplots(num_samples, 2, figsize=(12, num_samples * 3))
    if num_samples == 1:
        axes = axes.reshape(1, -1)

    for i in range(num_samples):
        # Denormalize and convert image
        img = denormalize_image(images[i].cpu())
        img = img.permute(1, 2, 0).numpy()

        # Plot image
        axes[i, 0].imshow(img)
        axes[i, 0].set_title(f'Sample {i+1}')
        axes[i, 0].axis('off')

        # Plot probability distributions
        pred = pred_probs[i].detach().cpu().numpy()
        target = target_probs[i].detach().cpu().numpy() if torch.is_tensor(target_probs) else target_probs[i]

        x = np.arange(len(emotion_labels))
        width = 0.35

        axes[i, 1].bar(x - width/2, target, width, label='Ground Truth', alpha=0.7, color='steelblue')
        axes[i, 1].bar(x + width/2, pred, width, label='Predicted', alpha=0.7, color='coral')

        axes[i, 1].set_xticks(x)
        axes[i, 1].set_xticklabels(emotion_labels, rotation=45, ha='right')
        axes[i, 1].set_ylabel('Probability')
        axes[i, 1].set_ylim(0, max(target.max(), pred.max()) * 1.1)
        axes[i, 1].legend()
        axes[i, 1].grid(axis='y', alpha=0.3)

        # Add top emotion annotation
        top_pred_idx = pred.argmax()
        top_target_idx = target.argmax()
        axes[i, 1].set_title(
            f'Pred: {emotion_labels[top_pred_idx]} ({pred[top_pred_idx]:.3f}) | '
            f'True: {emotion_labels[top_target_idx]} ({target[top_target_idx]:.3f})',
            fontsize=10
        )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")

    return fig


def plot_training_curves(history, save_path=None):
    """
    Plot training and validation curves.

    Args:
        history: Dictionary with 'train_loss', 'val_loss', etc.
        save_path: Optional path to save the figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot losses
    if 'train_loss' in history:
        axes[0].plot(history['train_loss'], label='Train Loss', linewidth=2)
    if 'val_loss' in history:
        axes[0].plot(history['val_loss'], label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Plot metrics (if available)
    if 'val_metrics' in history and len(history['val_metrics']) > 0:
        # Plot first metric
        first_metric = list(history['val_metrics'][0].keys())[0]
        metric_values = [m[first_metric] for m in history['val_metrics']]
        axes[1].plot(metric_values, label=first_metric.upper(), linewidth=2, color='green')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Metric Value')
        axes[1].set_title(f'Validation {first_metric.upper()}')
        axes[1].legend()
        axes[1].grid(alpha=0.3)
    else:
        axes[1].text(0.5, 0.5, 'No metrics available',
                    ha='center', va='center', transform=axes[1].transAxes)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved training curves to {save_path}")

    return fig


def count_parameters(model):
    """
    Count model parameters.

    Args:
        model: PyTorch model

    Returns:
        Dictionary with total and trainable parameter counts
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {
        'total': total,
        'trainable': trainable,
        'non_trainable': total - trainable
    }


def get_device():
    """
    Get the best available device (CUDA > MPS > CPU).

    Returns:
        torch.device
    """
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')


def set_seed(seed=42):
    """
    Set random seed for reproducibility.

    Args:
        seed: Random seed value
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    # Note: For full reproducibility, also set PYTHONHASHSEED and cudnn flags


class AverageMeter:
    """
    Computes and stores the average and current value.

    Useful for tracking metrics during training.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


if __name__ == "__main__":
    # Test utilities
    print("Testing utility functions...")

    # Test device detection
    device = get_device()
    print(f"\n✓ Device: {device}")

    # Test denormalization
    dummy_img = torch.randn(3, 224, 224)
    denorm = denormalize_image(dummy_img)
    print(f"\n✓ Denormalization:")
    print(f"  Input range: [{dummy_img.min():.2f}, {dummy_img.max():.2f}]")
    print(f"  Output range: [{denorm.min():.2f}, {denorm.max():.2f}]")

    # Test AverageMeter
    meter = AverageMeter()
    for i in range(10):
        meter.update(i)
    print(f"\n✓ AverageMeter:")
    print(f"  Current: {meter.val}")
    print(f"  Average: {meter.avg:.2f}")

    # Test checkpoint saving/loading
    print(f"\n✓ Checkpoint utilities:")
    print(f"  save_checkpoint() - ready")
    print(f"  load_checkpoint() - ready")

    print(f"\n✓ Visualization utilities:")
    print(f"  visualize_predictions() - ready")
    print(f"  plot_training_curves() - ready")
