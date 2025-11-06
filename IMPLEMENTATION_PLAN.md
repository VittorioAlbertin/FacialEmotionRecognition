# Implementation Plan for Emotion Recognition Project

## Phase 1: Core Infrastructure (Python Scripts)

### 1. Create `src/models.py`
- Base model class with unified interface
- **ResNet18** implementation (pretrained, 13-output head with softmax)
- **EfficientNet-B0** implementation (pretrained, 13-output head with softmax)
- **ResNet50** implementation (for comparison)
- Model factory function to easily switch between architectures

### 2. Create `src/losses.py`
- **MSE Loss** for probability distributions
- **L1 Loss (MAE)** for probability distributions
- **KL Divergence Loss** with numerical stability handling
- Wrapper to support multiple loss combinations

### 3. Create `src/metrics.py`
- MSE metric (for evaluation)
- Total Variation Distance
- KL Divergence metric
- Earth Mover's Distance (using scipy/POT)
- Metric computation utilities

### 4. Create `src/train.py`
- Training loop with validation
- TensorBoard logging (loss, metrics, learning rate)
- Model checkpointing (best model + last epoch)
- Early stopping support
- CLI arguments for hyperparameters

### 5. Create `src/utils.py`
- Configuration management
- Checkpoint loading/saving utilities
- Visualization helper functions
- Denormalization for image display

## Phase 2: Experiment Notebooks

### 6. Create `02_model_training.ipynb`
- Import models and training utilities from scripts
- Train baseline models (ResNet18 + EfficientNet-B0)
- Test each loss function (MSE, L1, KL Divergence)
- Compare training curves via TensorBoard
- Save best models

### 7. Create `03_evaluation.ipynb`
- Load trained models
- Compute all evaluation metrics on validation set
- Compare models across metrics
- Statistical analysis of results

### 8. Create `04_visualization.ipynb`
- Qualitative evaluation: predicted vs ground truth distributions
- Sample predictions with bar charts
- Error analysis (worst predictions)
- t-SNE/UMAP of learned embeddings (optional)

## Phase 3: Experiment Framework

### 9. Create `configs/` directory
- YAML/JSON config files for different experiments
- Base config with default hyperparameters
- Configs for each model+loss combination

### 10. Create `experiments/run_experiment.py`
- CLI script to run experiments from config files
- Logs to `runs/` directory with organized structure
- Enables batch experiment execution

## File Structure After Implementation
```
EmotionRecognition/
├── src/
│   ├── __init__.py
│   ├── models.py       # Model architectures
│   ├── losses.py       # Loss functions
│   ├── metrics.py      # Evaluation metrics
│   ├── train.py        # Training loop
│   └── utils.py        # Utilities
├── configs/
│   ├── base.yaml
│   ├── resnet18_mse.yaml
│   └── efficientnet_kl.yaml
├── experiments/
│   └── run_experiment.py
├── 01_data_loading.ipynb      # ✓ Complete
├── 02_model_training.ipynb    # Train models interactively
├── 03_evaluation.ipynb        # Evaluate and compare
├── 04_visualization.ipynb     # Visualize results
├── requirements.txt           # Update with new deps
└── runs/                      # TensorBoard logs (gitignored)
```

## Implementation Order
1. **models.py** → Basic architectures
2. **losses.py** → MSE, L1, KL Divergence
3. **metrics.py** → Evaluation metrics
4. **train.py** → Training loop with TensorBoard
5. **02_model_training.ipynb** → First training experiments
6. **utils.py** + **03_evaluation.ipynb** → Evaluation pipeline
7. **04_visualization.ipynb** → Qualitative analysis
8. **configs/** + **run_experiment.py** → Experiment automation

## Key Features
- Modular design: easy to add new models/losses
- Hybrid approach: scripts for logic, notebooks for exploration
- TensorBoard integration throughout
- Proper checkpointing and reproducibility
- Multiple models comparison ready from the start

## Dependencies to Add
- `tensorboard` - For experiment tracking
- `scipy` - For Earth Mover's Distance
- `POT` (Python Optimal Transport) - Alternative for EMD
- `pyyaml` - For config file parsing
- `timm` - For EfficientNet pretrained models

## Notes
- Start with Phase 1 to build solid foundation
- Test each component before moving to next
- Use notebooks (Phase 2) for iterative experimentation
- Phase 3 is optional but useful for running many experiments
