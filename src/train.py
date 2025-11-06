"""
Training loop for emotion recognition models.

Includes:
- Training and validation loops
- TensorBoard logging
- Model checkpointing
- Early stopping
"""

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import time
from pathlib import Path

from .utils import save_checkpoint, AverageMeter, get_device
from .metrics import MetricsCalculator


class Trainer:
    """
    Trainer class for emotion recognition models.

    Args:
        model: PyTorch model
        train_loader: Training DataLoader
        val_loader: Validation DataLoader
        loss_fn: Loss function
        optimizer: Optimizer
        device: Device to train on (cuda/cpu)
        checkpoint_dir: Directory to save checkpoints
        log_dir: Directory for TensorBoard logs
        metrics: List of metrics to compute (default: ['mse', 'kl'])
    """

    def __init__(self, model, train_loader, val_loader, loss_fn, optimizer,
                 device=None, checkpoint_dir='checkpoints', log_dir='runs',
                 metrics=None):

        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_fn = loss_fn
        self.optimizer = optimizer

        # Device
        self.device = device if device is not None else get_device()
        self.model = self.model.to(self.device)

        # Directories
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # TensorBoard
        self.writer = SummaryWriter(log_dir=log_dir)

        # Metrics
        self.metrics_calculator = MetricsCalculator(metrics=metrics)

        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_metrics': []
        }

    def train_epoch(self):
        """Run one training epoch."""
        self.model.train()
        loss_meter = AverageMeter()

        pbar = tqdm(self.train_loader, desc=f'Epoch {self.current_epoch + 1} [Train]')

        for batch_idx, (images, targets) in enumerate(pbar):
            # Move to device
            images = images.to(self.device)
            targets = targets.to(self.device)

            # Forward pass
            outputs = self.model(images)
            loss = self.loss_fn(outputs, targets)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Update metrics
            loss_meter.update(loss.item(), images.size(0))

            # Update progress bar
            pbar.set_postfix({'loss': f'{loss_meter.avg:.4f}'})

            # Log to TensorBoard (every 100 batches)
            if batch_idx % 100 == 0:
                global_step = self.current_epoch * len(self.train_loader) + batch_idx
                self.writer.add_scalar('train/loss_batch', loss.item(), global_step)

        return loss_meter.avg

    def validate(self):
        """Run validation."""
        self.model.eval()
        loss_meter = AverageMeter()

        all_outputs = []
        all_targets = []

        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f'Epoch {self.current_epoch + 1} [Val]')

            for images, targets in pbar:
                # Move to device
                images = images.to(self.device)
                targets = targets.to(self.device)

                # Forward pass
                outputs = self.model(images)
                loss = self.loss_fn(outputs, targets)

                # Update metrics
                loss_meter.update(loss.item(), images.size(0))

                # Collect for metrics computation
                all_outputs.append(outputs.cpu())
                all_targets.append(targets.cpu())

                # Update progress bar
                pbar.set_postfix({'loss': f'{loss_meter.avg:.4f}'})

        # Compute metrics
        all_outputs = torch.cat(all_outputs, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        metrics = self.metrics_calculator.compute_all(all_outputs, all_targets)

        return loss_meter.avg, metrics

    def train(self, num_epochs, scheduler=None, early_stopping_patience=None):
        """
        Train the model.

        Args:
            num_epochs: Number of epochs to train
            scheduler: Optional learning rate scheduler
            early_stopping_patience: Stop if val loss doesn't improve for this many epochs
        """
        print(f"\nTraining on device: {self.device}")
        print(f"Model: {self.model.__class__.__name__}")
        print(f"Loss: {self.loss_fn}")
        print(f"Optimizer: {self.optimizer.__class__.__name__}")
        print(f"Epochs: {num_epochs}")
        print(f"Train batches: {len(self.train_loader)}")
        print(f"Val batches: {len(self.val_loader)}")
        print(f"Checkpoint dir: {self.checkpoint_dir}")
        print(f"TensorBoard logs: {self.writer.log_dir}\n")

        epochs_without_improvement = 0
        start_time = time.time()

        for epoch in range(num_epochs):
            self.current_epoch = epoch

            # Train
            train_loss = self.train_epoch()
            self.history['train_loss'].append(train_loss)

            # Validate
            val_loss, val_metrics = self.validate()
            self.history['val_loss'].append(val_loss)
            self.history['val_metrics'].append(val_metrics)

            # Learning rate scheduling
            if scheduler is not None:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val_loss)
                else:
                    scheduler.step()

            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']

            # Log to TensorBoard
            self.writer.add_scalar('epoch/train_loss', train_loss, epoch)
            self.writer.add_scalar('epoch/val_loss', val_loss, epoch)
            self.writer.add_scalar('epoch/learning_rate', current_lr, epoch)

            for metric_name, metric_value in val_metrics.items():
                self.writer.add_scalar(f'epoch/val_{metric_name}', metric_value, epoch)

            # Print epoch summary
            metrics_str = ' | '.join([f'{k}: {v:.4f}' for k, v in val_metrics.items()])
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss: {val_loss:.4f}")
            print(f"  Val Metrics: {metrics_str}")
            print(f"  LR: {current_lr:.6f}")

            # Save checkpoint
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                epochs_without_improvement = 0
                print(f"  ✓ New best model! (Val Loss: {val_loss:.4f})")
            else:
                epochs_without_improvement += 1

            checkpoint_state = {
                'epoch': epoch + 1,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_metrics': val_metrics,
                'best_val_loss': self.best_val_loss,
                'history': self.history
            }

            save_checkpoint(
                checkpoint_state,
                self.checkpoint_dir,
                filename=f'checkpoint_epoch_{epoch + 1}.pth',
                is_best=is_best
            )

            # Early stopping
            if early_stopping_patience is not None:
                if epochs_without_improvement >= early_stopping_patience:
                    print(f"\n⚠ Early stopping after {epoch + 1} epochs "
                          f"(no improvement for {early_stopping_patience} epochs)")
                    break

        # Training complete
        total_time = time.time() - start_time
        print(f"\n✓ Training complete!")
        print(f"  Total time: {total_time / 60:.2f} minutes")
        print(f"  Best val loss: {self.best_val_loss:.4f}")
        print(f"  Final train loss: {self.history['train_loss'][-1]:.4f}")
        print(f"  Final val loss: {self.history['val_loss'][-1]:.4f}")

        self.writer.close()

        return self.history

    def save_final_checkpoint(self, filename='final_model.pth'):
        """Save final model checkpoint."""
        checkpoint_state = {
            'epoch': self.current_epoch + 1,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'history': self.history
        }

        save_checkpoint(checkpoint_state, self.checkpoint_dir, filename=filename)
        print(f"✓ Saved final model to {self.checkpoint_dir / filename}")


def create_trainer(model, train_loader, val_loader, loss_fn, optimizer,
                   device=None, checkpoint_dir='checkpoints', log_dir='runs',
                   metrics=None):
    """
    Factory function to create a Trainer instance.

    Args:
        model: PyTorch model
        train_loader: Training DataLoader
        val_loader: Validation DataLoader
        loss_fn: Loss function
        optimizer: Optimizer
        device: Device to train on
        checkpoint_dir: Directory to save checkpoints
        log_dir: Directory for TensorBoard logs
        metrics: List of metrics to compute

    Returns:
        Trainer instance
    """
    return Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        device=device,
        checkpoint_dir=checkpoint_dir,
        log_dir=log_dir,
        metrics=metrics
    )


if __name__ == "__main__":
    print("Train module loaded successfully!")
    print("\nUsage example:")
    print("""
    from src.models import create_model
    from src.losses import get_loss_function
    from src.train import create_trainer

    # Create model, loss, optimizer
    model = create_model('resnet18', pretrained=True)
    loss_fn = get_loss_function('mse')
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Create trainer
    trainer = create_trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        checkpoint_dir='checkpoints/resnet18_mse',
        log_dir='runs/resnet18_mse'
    )

    # Train
    history = trainer.train(num_epochs=10)
    """)
