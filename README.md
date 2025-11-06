# Emotion Recognition with Probability Distributions

A deep learning project for emotion recognition from facial images that predicts **probability distributions** over 13 emotion classes, rather than single labels.

## 📋 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Model Architectures](#model-architectures)
- [Loss Functions](#loss-functions)
- [Evaluation Metrics](#evaluation-metrics)
- [Training](#training)
- [Results](#results)
- [References](#references)

## 🎯 Overview

Unlike standard emotion recognition systems that output a single emotion label, this project predicts a **13-dimensional probability distribution** representing the likelihood of each emotion. This approach better captures the nuanced and multi-faceted nature of human emotions.

### Emotion Classes

The model predicts probabilities for 13 emotions:
- `neutral`, `happy`, `sad`, `surprised`, `fear`, `disgust`, `angry`
- `contempt`, `serene`, `contemplative`, `secure`, `untroubled`, `quiet`

### Key Challenge

The core challenge is designing loss functions that effectively measure similarity between probability distributions, moving beyond simple classification accuracy.

## 📂 Dataset

**AffectNetFused Dataset**

- **Training Set:** 308,468 images
- **Validation Set:** 3,999 images
- **Format:** RGB images with corresponding probability annotations

Each image has multiple annotation files, but we use only `*_prob_rank.txt` containing 13 comma-separated probability values.

**Download:** [Dataset Link](https://drive.google.com/file/d/1BmmyOhtuJ1dJa9mA3qvi_4jX_-2oPFly/view)

### Dataset Structure
```
AffectNetFused/
├── train_set/
│   ├── images/
│   │   ├── 0000001.jpg
│   │   ├── 0000002.jpg
│   │   └── ...
│   └── annotations/
│       ├── 0000001_prob_rank.txt
│       ├── 0000002_prob_rank.txt
│       └── ...
└── val_set/
    ├── images/
    └── annotations/
```

## ✨ Features

- **Multiple Model Architectures:** ResNet, EfficientNet, Vision Transformer
- **Diverse Loss Functions:** MSE, L1, KL Divergence, Cross-Entropy, Jensen-Shannon Divergence
- **Comprehensive Metrics:** MSE, Total Variation Distance, KL Divergence, Earth Mover's Distance
- **TensorBoard Integration:** Real-time training monitoring
- **Automatic Checkpointing:** Save best models and training state
- **Visualization Tools:** Prediction vs ground truth comparison
- **Modular Design:** Easy to extend with new models/losses
- **Hybrid Workflow:** Python scripts + Jupyter notebooks

## 🚀 Installation

### Prerequisites

- Python 3.10+
- CUDA-capable GPU (recommended)

### Setup

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd EmotionRecognition
   ```

2. **Create virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download and extract dataset:**
   - Download from [Dataset Link](https://drive.google.com/file/d/1BmmyOhtuJ1dJa9mA3qvi_4jX_-2oPFly/view)
   - Extract to `AffectNetFused/` in project root

## 🎬 Quick Start

### 1. Explore the Dataset

```bash
jupyter notebook 01_data_loading.ipynb
```

This notebook covers:
- Dataset exploration and statistics
- Sample visualization
- Data loading pipeline
- Probability distribution analysis

### 2. Train Your First Model

```bash
jupyter notebook 02_model_training.ipynb
```

Or use Python directly:

```python
from src.models import create_model
from src.losses import get_loss_function
from src.train import create_trainer
import torch.optim as optim

# Create model
model = create_model('resnet18', pretrained=True)

# Setup training
loss_fn = get_loss_function('mse')
optimizer = optim.Adam(model.parameters(), lr=0.001)

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

# Train!
history = trainer.train(num_epochs=10)
```

### 3. Monitor Training

```bash
tensorboard --logdir=runs
```

Open http://localhost:6006 in your browser to view:
- Training/validation loss curves
- Evaluation metrics over time
- Learning rate schedules
- Model performance comparisons

## 📁 Project Structure

```
EmotionRecognition/
├── src/                              # Core source code
│   ├── __init__.py
│   ├── models.py                     # Model architectures
│   ├── losses.py                     # Loss functions
│   ├── metrics.py                    # Evaluation metrics
│   ├── train.py                      # Training loop
│   └── utils.py                      # Utilities
│
├── AffectNetFused/                   # Dataset (not in git)
│   ├── train_set/
│   └── val_set/
│
├── checkpoints/                      # Saved models (not in git)
├── runs/                             # TensorBoard logs (not in git)
│
├── 01_data_loading.ipynb             # Data exploration notebook
├── 02_model_training.ipynb           # Training notebook
│
├── requirements.txt                  # Python dependencies
├── .gitignore
└── README.md                         # This file
```

## 📖 Usage

### Model Creation

```python
from src.models import create_model

# ResNet models
model = create_model('resnet18', pretrained=True)
model = create_model('resnet50', pretrained=True)

# EfficientNet models
model = create_model('efficientnet_b0', pretrained=True)

# Vision Transformer
model = create_model('vit_base_patch16_224', pretrained=True)
```

### Loss Functions

```python
from src.losses import get_loss_function

# Simple losses
loss_fn = get_loss_function('mse')        # Mean Squared Error
loss_fn = get_loss_function('l1')         # L1/MAE loss

# Distribution-based losses
loss_fn = get_loss_function('kl')         # KL Divergence
loss_fn = get_loss_function('ce')         # Cross-Entropy
loss_fn = get_loss_function('js')         # Jensen-Shannon

# Combined loss
from src.losses import MSELoss, KLDivLoss, CombinedLoss
loss_fn = CombinedLoss([
    (MSELoss(), 0.5),
    (KLDivLoss(), 0.5)
])
```

### Evaluation Metrics

```python
from src.metrics import MetricsCalculator

calculator = MetricsCalculator(metrics=['mse', 'kl', 'tvd', 'emd'])

# Compute all metrics
metrics = calculator.compute_all(predictions, targets)
print(metrics)  # {'mse': 0.0234, 'kl': 0.1456, ...}

# Per-sample metrics
per_sample = calculator.compute_per_sample(predictions, targets)
```

### Visualization

```python
from src.utils import visualize_predictions, plot_training_curves

# Visualize predictions
visualize_predictions(
    images,
    predictions,
    targets,
    num_samples=4,
    save_path='predictions.png'
)

# Plot training curves
plot_training_curves(
    history,
    save_path='training_curves.png'
)
```

### Loading Checkpoints

```python
from src.utils import load_checkpoint

# Load best model
checkpoint_info = load_checkpoint(
    'checkpoints/resnet18_mse/best_model.pth',
    model,
    optimizer
)

print(f"Loaded model from epoch {checkpoint_info['epoch']}")
print(f"Best validation loss: {checkpoint_info['best_loss']:.4f}")
```

## 🏗️ Model Architectures

### ResNet Family
- **ResNet18** - Fast, lightweight (11M parameters)
- **ResNet50** - Deeper, more accurate (23M parameters)

### EfficientNet Family
- **EfficientNet-B0** - Efficient, state-of-the-art (5M parameters)

### Vision Transformers
- **ViT-Base** - Transformer-based, requires more data

All models:
- Use ImageNet pretrained weights
- Custom 13-output head with softmax activation
- Dropout (0.3) for regularization

## 📉 Loss Functions

### Simple Baselines
- **MSE (Mean Squared Error):** Treats distributions as vectors
- **L1/MAE:** Manhattan distance between distributions

### Distribution-Specific
- **KL Divergence:** Measures information loss, asymmetric
- **Cross-Entropy:** Standard for probability distributions
- **Jensen-Shannon Divergence:** Symmetric version of KL

### Combined
- Weighted combinations of multiple losses
- Example: `0.5 * MSE + 0.5 * KL Divergence`

## 📊 Evaluation Metrics

All metrics computed on validation set:

| Metric | Description | Range |
|--------|-------------|-------|
| **MSE** | Mean Squared Error | [0, ∞) |
| **Total Variation Distance** | 0.5 * sum(\|p - q\|) | [0, 1] |
| **KL Divergence** | Information loss measure | [0, ∞) |
| **Earth Mover's Distance** | Optimal transport cost | [0, ∞) |
| **Jensen-Shannon** | Symmetric divergence | [0, 1] |

## 🎓 Training

### Basic Training

```python
trainer.train(
    num_epochs=20,
    scheduler=scheduler,
    early_stopping_patience=5
)
```

### Training Features

- **Automatic checkpointing:** Best model saved based on validation loss
- **Early stopping:** Stop if no improvement after N epochs
- **Learning rate scheduling:** Reduce LR on plateau
- **Progress bars:** Track training/validation progress
- **TensorBoard logging:** Real-time metrics visualization

### Recommended Hyperparameters

**For ResNet18/50:**
```python
learning_rate = 1e-3
batch_size = 32
optimizer = Adam
scheduler = ReduceLROnPlateau(patience=3, factor=0.5)
```

**For EfficientNet-B0:**
```python
learning_rate = 1e-4
batch_size = 64
optimizer = AdamW
```

## 📈 Results

*Results will be added after running experiments*

### Model Comparison

| Model | Loss | Val MSE | Val KL | Val TVD | Time/Epoch |
|-------|------|---------|--------|---------|------------|
| ResNet18 | MSE | TBD | TBD | TBD | TBD |
| ResNet18 | KL | TBD | TBD | TBD | TBD |
| EfficientNet-B0 | MSE | TBD | TBD | TBD | TBD |
| EfficientNet-B0 | KL | TBD | TBD | TBD | TBD |

### Best Model

*To be determined after experimentation*

## 🔬 Experiment Tracking

All experiments are logged to TensorBoard:

```bash
tensorboard --logdir=runs
```

Experiment naming convention: `{model}_{loss}`

Examples:
- `resnet18_mse`
- `resnet50_kl`
- `efficientnet_b0_mse`

## 🐛 Troubleshooting

### CUDA Out of Memory
```python
# Reduce batch size
BATCH_SIZE = 16

# Or use gradient accumulation
# (implementation in train.py)
```

### Slow Data Loading
```python
# Increase num_workers (if not on Windows)
train_loader = DataLoader(..., num_workers=4)

# Or use cached probabilities
# (already implemented in 01_data_loading.ipynb)
```

### NaN Loss with KL Divergence
```python
# Increase epsilon for numerical stability
loss_fn = get_loss_function('kl', epsilon=1e-7)
```

## 📚 References

### Papers
- [AffectNet Paper](https://arxiv.org/abs/1708.03985) - Original AffectNet dataset
- ResNet: "Deep Residual Learning for Image Recognition" (He et al., 2015)
- EfficientNet: "EfficientNet: Rethinking Model Scaling for CNNs" (Tan & Le, 2019)

### Libraries
- [PyTorch](https://pytorch.org/) - Deep learning framework
- [timm](https://github.com/huggingface/pytorch-image-models) - Pre-trained models
- [TensorBoard](https://www.tensorflow.org/tensorboard) - Experiment tracking

## 🤝 Contributing

This is an academic project for the Deep Learning course (Academic Year 2024/2025).

## 📝 License

This project is for educational purposes only.

## 👥 Authors

- Student: [Your Name]
- Course: Deep Learning, MUNER, First Year
- Institution: [University Name]
- Academic Year: 2024/2025

## 🙏 Acknowledgments

- AffectNet dataset creators
- Course instructors and TAs
- PyTorch and timm library maintainers

---

**Note:** For detailed implementation guidance, see `CLAUDE.md` and `IMPLEMENTATION_PLAN.md`.

**Dataset:** Remember to download the dataset before running the notebooks!
