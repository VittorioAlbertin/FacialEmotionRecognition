"""
Evaluation metrics for emotion probability distribution comparison.

Implements metrics for quantitative evaluation:
- MSE: Mean Squared Error
- Total Variation Distance
- KL Divergence
- Earth Mover's Distance (Wasserstein)
"""

import torch
import numpy as np
from scipy.stats import wasserstein_distance


def mse_metric(pred, target):
    """
    Mean Squared Error between probability distributions.

    Args:
        pred: Predicted probabilities (batch_size, num_emotions) or (num_emotions,)
        target: Target probabilities (batch_size, num_emotions) or (num_emotions,)

    Returns:
        MSE value (scalar or per-sample if batch)
    """
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()

    return np.mean((pred - target) ** 2, axis=-1)


def total_variation_distance(pred, target):
    """
    Total Variation Distance between probability distributions.

    TVD = 0.5 * sum(|p_i - q_i|)

    Args:
        pred: Predicted probabilities (batch_size, num_emotions) or (num_emotions,)
        target: Target probabilities (batch_size, num_emotions) or (num_emotions,)

    Returns:
        TVD value (scalar or per-sample if batch)
    """
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()

    return 0.5 * np.sum(np.abs(pred - target), axis=-1)


def kl_divergence_metric(pred, target, epsilon=1e-8):
    """
    Kullback-Leibler Divergence KL(target || pred).

    KL(P||Q) = sum(P_i * log(P_i / Q_i))

    Args:
        pred: Predicted probabilities (batch_size, num_emotions) or (num_emotions,)
        target: Target probabilities (batch_size, num_emotions) or (num_emotions,)
        epsilon: Small constant for numerical stability

    Returns:
        KL divergence value (scalar or per-sample if batch)
    """
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()

    # Add epsilon and normalize
    pred = pred + epsilon
    target = target + epsilon
    pred = pred / pred.sum(axis=-1, keepdims=True)
    target = target / target.sum(axis=-1, keepdims=True)

    # Compute KL divergence
    kl = np.sum(target * np.log(target / pred), axis=-1)
    return kl


def earth_movers_distance(pred, target):
    """
    Earth Mover's Distance (Wasserstein-1) between probability distributions.

    For 1D distributions, this is the optimal transport cost.

    Args:
        pred: Predicted probabilities (batch_size, num_emotions) or (num_emotions,)
        target: Target probabilities (batch_size, num_emotions) or (num_emotions,)

    Returns:
        EMD value (scalar or per-sample if batch)
    """
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()

    # Handle single sample
    if pred.ndim == 1:
        return wasserstein_distance(
            range(len(pred)), range(len(target)),
            pred, target
        )

    # Handle batch
    emd_values = []
    for p, t in zip(pred, target):
        emd = wasserstein_distance(
            range(len(p)), range(len(t)),
            p, t
        )
        emd_values.append(emd)

    return np.array(emd_values)


def js_divergence_metric(pred, target, epsilon=1e-8):
    """
    Jensen-Shannon Divergence between probability distributions.

    JS(P||Q) = 0.5*KL(P||M) + 0.5*KL(Q||M) where M = 0.5*(P+Q)

    Args:
        pred: Predicted probabilities (batch_size, num_emotions) or (num_emotions,)
        target: Target probabilities (batch_size, num_emotions) or (num_emotions,)
        epsilon: Small constant for numerical stability

    Returns:
        JS divergence value (scalar or per-sample if batch)
    """
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()

    # Add epsilon and normalize
    pred = pred + epsilon
    target = target + epsilon
    pred = pred / pred.sum(axis=-1, keepdims=True)
    target = target / target.sum(axis=-1, keepdims=True)

    # Compute middle distribution
    m = 0.5 * (pred + target)

    # JS divergence
    kl_pm = np.sum(pred * np.log(pred / m), axis=-1)
    kl_qm = np.sum(target * np.log(target / m), axis=-1)
    js = 0.5 * (kl_pm + kl_qm)

    return js


class MetricsCalculator:
    """
    Utility class to compute all metrics at once.

    Usage:
        calculator = MetricsCalculator()
        metrics = calculator.compute_all(pred, target)
    """

    def __init__(self, metrics=None):
        """
        Args:
            metrics: List of metric names to compute.
                     If None, computes all metrics.
                     Options: 'mse', 'tvd', 'kl', 'emd', 'js'
        """
        if metrics is None:
            metrics = ['mse', 'tvd', 'kl', 'emd', 'js']
        self.metrics = metrics

    def compute_all(self, pred, target):
        """
        Compute all configured metrics.

        Args:
            pred: Predicted probabilities (batch_size, num_emotions)
            target: Target probabilities (batch_size, num_emotions)

        Returns:
            Dictionary of metric_name -> mean_value
        """
        results = {}

        if 'mse' in self.metrics:
            results['mse'] = np.mean(mse_metric(pred, target))

        if 'tvd' in self.metrics:
            results['tvd'] = np.mean(total_variation_distance(pred, target))

        if 'kl' in self.metrics:
            results['kl'] = np.mean(kl_divergence_metric(pred, target))

        if 'emd' in self.metrics:
            results['emd'] = np.mean(earth_movers_distance(pred, target))

        if 'js' in self.metrics:
            results['js'] = np.mean(js_divergence_metric(pred, target))

        return results

    def compute_per_sample(self, pred, target):
        """
        Compute all configured metrics per sample.

        Args:
            pred: Predicted probabilities (batch_size, num_emotions)
            target: Target probabilities (batch_size, num_emotions)

        Returns:
            Dictionary of metric_name -> array of per-sample values
        """
        results = {}

        if 'mse' in self.metrics:
            results['mse'] = mse_metric(pred, target)

        if 'tvd' in self.metrics:
            results['tvd'] = total_variation_distance(pred, target)

        if 'kl' in self.metrics:
            results['kl'] = kl_divergence_metric(pred, target)

        if 'emd' in self.metrics:
            results['emd'] = earth_movers_distance(pred, target)

        if 'js' in self.metrics:
            results['js'] = js_divergence_metric(pred, target)

        return results

    def __repr__(self):
        return f"MetricsCalculator(metrics={self.metrics})"


if __name__ == "__main__":
    # Test metrics
    print("Testing evaluation metrics...")

    # Create dummy predictions and targets
    np.random.seed(42)

    # Single sample test
    pred_single = np.random.rand(13)
    pred_single = pred_single / pred_single.sum()

    target_single = np.random.rand(13)
    target_single = target_single / target_single.sum()

    print(f"\nSingle sample test:")
    print(f"  Pred sum: {pred_single.sum():.4f}")
    print(f"  Target sum: {target_single.sum():.4f}")

    print(f"\n✓ Metrics:")
    print(f"  MSE: {mse_metric(pred_single, target_single):.6f}")
    print(f"  TVD: {total_variation_distance(pred_single, target_single):.6f}")
    print(f"  KL Divergence: {kl_divergence_metric(pred_single, target_single):.6f}")
    print(f"  EMD: {earth_movers_distance(pred_single, target_single):.6f}")
    print(f"  JS Divergence: {js_divergence_metric(pred_single, target_single):.6f}")

    # Batch test
    batch_size = 4
    pred_batch = np.random.rand(batch_size, 13)
    pred_batch = pred_batch / pred_batch.sum(axis=1, keepdims=True)

    target_batch = np.random.rand(batch_size, 13)
    target_batch = target_batch / target_batch.sum(axis=1, keepdims=True)

    print(f"\n\nBatch test ({batch_size} samples):")
    print(f"  Pred shape: {pred_batch.shape}")
    print(f"  Target shape: {target_batch.shape}")

    # Test MetricsCalculator
    calculator = MetricsCalculator()
    metrics = calculator.compute_all(pred_batch, target_batch)

    print(f"\n✓ Average metrics:")
    for name, value in metrics.items():
        print(f"  {name.upper()}: {value:.6f}")

    # Test per-sample metrics
    per_sample = calculator.compute_per_sample(pred_batch, target_batch)
    print(f"\n✓ Per-sample metrics shape:")
    for name, values in per_sample.items():
        print(f"  {name.upper()}: {values.shape}")

    # Test with PyTorch tensors
    print(f"\n\n✓ PyTorch tensor test:")
    pred_torch = torch.from_numpy(pred_batch)
    target_torch = torch.from_numpy(target_batch)
    metrics_torch = calculator.compute_all(pred_torch, target_torch)
    print(f"  MSE: {metrics_torch['mse']:.6f}")
