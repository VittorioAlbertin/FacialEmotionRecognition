"""
Loss functions for emotion probability distribution matching.

Implements various losses suitable for comparing probability distributions:
- MSE: Mean Squared Error
- L1/MAE: Mean Absolute Error
- KL Divergence: Kullback-Leibler Divergence
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MSELoss(nn.Module):
    """
    Mean Squared Error loss for probability distributions.

    Simple baseline loss that treats distributions as vectors.
    """

    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        """
        Args:
            pred: Predicted probability distribution (batch_size, num_emotions)
            target: Target probability distribution (batch_size, num_emotions)

        Returns:
            Scalar loss value
        """
        return F.mse_loss(pred, target)

    def __repr__(self):
        return "MSELoss()"


class L1Loss(nn.Module):
    """
    L1/Mean Absolute Error loss for probability distributions.

    Also known as Total Variation Distance when summed.
    """

    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        """
        Args:
            pred: Predicted probability distribution (batch_size, num_emotions)
            target: Target probability distribution (batch_size, num_emotions)

        Returns:
            Scalar loss value
        """
        return F.l1_loss(pred, target)

    def __repr__(self):
        return "L1Loss()"


class KLDivLoss(nn.Module):
    """
    Kullback-Leibler Divergence loss for probability distributions.

    KL(target || pred) measures how much information is lost when using pred
    to approximate target.

    Args:
        epsilon: Small constant for numerical stability (default: 1e-8)
    """

    def __init__(self, epsilon=1e-8):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, pred, target):
        """
        Args:
            pred: Predicted probability distribution (batch_size, num_emotions)
            target: Target probability distribution (batch_size, num_emotions)

        Returns:
            Scalar loss value
        """
        # Add epsilon for numerical stability
        pred = pred + self.epsilon
        target = target + self.epsilon

        # Normalize to ensure they're valid probability distributions
        pred = pred / pred.sum(dim=1, keepdim=True)
        target = target / target.sum(dim=1, keepdim=True)

        # KL divergence: sum(target * log(target / pred))
        kl_div = target * torch.log(target / pred)
        return kl_div.sum(dim=1).mean()

    def __repr__(self):
        return f"KLDivLoss(epsilon={self.epsilon})"


class CrossEntropyLoss(nn.Module):
    """
    Cross-Entropy loss adapted for probability distributions.

    Standard cross-entropy but with soft targets (probability distributions)
    instead of hard labels.

    Args:
        epsilon: Small constant for numerical stability (default: 1e-8)
    """

    def __init__(self, epsilon=1e-8):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, pred, target):
        """
        Args:
            pred: Predicted probability distribution (batch_size, num_emotions)
            target: Target probability distribution (batch_size, num_emotions)

        Returns:
            Scalar loss value
        """
        # Add epsilon for numerical stability
        pred = pred + self.epsilon

        # Cross-entropy: -sum(target * log(pred))
        cross_entropy = -(target * torch.log(pred))
        return cross_entropy.sum(dim=1).mean()

    def __repr__(self):
        return f"CrossEntropyLoss(epsilon={self.epsilon})"


class JSDivLoss(nn.Module):
    """
    Jensen-Shannon Divergence loss for probability distributions.

    Symmetric version of KL Divergence. JS(P||Q) = 0.5*KL(P||M) + 0.5*KL(Q||M)
    where M = 0.5*(P+Q).

    Args:
        epsilon: Small constant for numerical stability (default: 1e-8)
    """

    def __init__(self, epsilon=1e-8):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, pred, target):
        """
        Args:
            pred: Predicted probability distribution (batch_size, num_emotions)
            target: Target probability distribution (batch_size, num_emotions)

        Returns:
            Scalar loss value
        """
        # Add epsilon for numerical stability
        pred = pred + self.epsilon
        target = target + self.epsilon

        # Normalize
        pred = pred / pred.sum(dim=1, keepdim=True)
        target = target / target.sum(dim=1, keepdim=True)

        # Compute middle distribution
        m = 0.5 * (pred + target)

        # JS divergence
        kl_pm = (pred * torch.log(pred / m)).sum(dim=1)
        kl_qm = (target * torch.log(target / m)).sum(dim=1)
        js_div = 0.5 * (kl_pm + kl_qm)

        return js_div.mean()

    def __repr__(self):
        return f"JSDivLoss(epsilon={self.epsilon})"


class CombinedLoss(nn.Module):
    """
    Weighted combination of multiple loss functions.

    Args:
        losses: List of (loss_fn, weight) tuples
            Example: [(MSELoss(), 0.5), (KLDivLoss(), 0.5)]
    """

    def __init__(self, losses):
        super().__init__()
        self.losses = nn.ModuleList([loss_fn for loss_fn, _ in losses])
        self.weights = [weight for _, weight in losses]

        # Normalize weights
        total_weight = sum(self.weights)
        self.weights = [w / total_weight for w in self.weights]

    def forward(self, pred, target):
        """
        Args:
            pred: Predicted probability distribution (batch_size, num_emotions)
            target: Target probability distribution (batch_size, num_emotions)

        Returns:
            Scalar loss value
        """
        total_loss = 0.0
        for loss_fn, weight in zip(self.losses, self.weights):
            total_loss += weight * loss_fn(pred, target)
        return total_loss

    def __repr__(self):
        loss_strs = [f"({loss}, {weight:.2f})" for loss, weight in zip(self.losses, self.weights)]
        return f"CombinedLoss({', '.join(loss_strs)})"


def get_loss_function(loss_type='mse', **kwargs):
    """
    Factory function to create loss functions.

    Args:
        loss_type: Type of loss function
            - 'mse': Mean Squared Error
            - 'l1' or 'mae': L1/Mean Absolute Error
            - 'kl' or 'kldiv': KL Divergence
            - 'ce' or 'crossentropy': Cross-Entropy
            - 'js' or 'jsdiv': Jensen-Shannon Divergence
        **kwargs: Additional arguments for the loss function

    Returns:
        Loss function instance

    Example:
        >>> loss_fn = get_loss_function('mse')
        >>> loss_fn = get_loss_function('kl', epsilon=1e-7)
        >>> loss_fn = get_loss_function('combined', losses=[(MSELoss(), 0.5), (KLDivLoss(), 0.5)])
    """
    loss_type = loss_type.lower()

    if loss_type == 'mse':
        return MSELoss()
    elif loss_type in ['l1', 'mae']:
        return L1Loss()
    elif loss_type in ['kl', 'kldiv']:
        return KLDivLoss(**kwargs)
    elif loss_type in ['ce', 'crossentropy']:
        return CrossEntropyLoss(**kwargs)
    elif loss_type in ['js', 'jsdiv']:
        return JSDivLoss(**kwargs)
    elif loss_type == 'combined':
        return CombinedLoss(**kwargs)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}. "
                        f"Supported: mse, l1/mae, kl/kldiv, ce/crossentropy, js/jsdiv, combined")


if __name__ == "__main__":
    # Test loss functions
    print("Testing loss functions...")

    # Create dummy predictions and targets
    batch_size = 4
    num_emotions = 13

    # Predicted distribution (random probabilities, normalized)
    pred = torch.rand(batch_size, num_emotions)
    pred = pred / pred.sum(dim=1, keepdim=True)

    # Target distribution (random probabilities, normalized)
    target = torch.rand(batch_size, num_emotions)
    target = target / target.sum(dim=1, keepdim=True)

    print(f"\nInput shapes:")
    print(f"  Pred: {pred.shape}, sum: {pred.sum(dim=1)}")
    print(f"  Target: {target.shape}, sum: {target.sum(dim=1)}")

    # Test each loss function
    losses = {
        'MSE': MSELoss(),
        'L1': L1Loss(),
        'KL Divergence': KLDivLoss(),
        'Cross-Entropy': CrossEntropyLoss(),
        'JS Divergence': JSDivLoss(),
    }

    print(f"\n✓ Loss values:")
    for name, loss_fn in losses.items():
        loss_value = loss_fn(pred, target)
        print(f"  {name}: {loss_value.item():.6f}")

    # Test combined loss
    combined = CombinedLoss([
        (MSELoss(), 0.5),
        (KLDivLoss(), 0.5)
    ])
    combined_value = combined(pred, target)
    print(f"  Combined (MSE+KL): {combined_value.item():.6f}")

    # Test factory function
    print(f"\n✓ Factory function test:")
    loss_fn = get_loss_function('kl')
    print(f"  Created: {loss_fn}")
    print(f"  Loss: {loss_fn(pred, target).item():.6f}")
