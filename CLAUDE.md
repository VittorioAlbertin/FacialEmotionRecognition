# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Deep Learning project for **emotion recognition from face images**, with the goal of predicting probability distributions over 13 emotion classes rather than single labels.

**Key characteristics:**
- Input: Face images from a custom dataset (AffectNetFused)
- Output: 13-dimensional probability vector over emotions: `neutral, happy, sad, surprised, fear, disgust, angry, contempt, serene, contemplative, secure, untroubled, quiet`
- Training approach: Custom loss functions for distribution matching (KL Divergence, MSE, L1, Cross-Entropy variants)
- Evaluation: Both quantitative (MSE, Total Variation Distance, EMD, KL Divergence) and qualitative (visualization of predicted vs ground truth distributions)

## Dataset Structure

The dataset (in `AffectNetFused`) contains training and validation splits. Each image `xxx` has a corresponding annotation file `xxx_prob_rank.txt` containing probability values for all 13 emotions in the order listed above.

Dataset link: https://drive.google.com/file/d/1BmmyOhtuJ1dJa9mA3qvi_4jX_-2oPFly/view

## Architecture Considerations

When implementing models for this project:

1. **Model output**: Must produce exactly 13 probability values (use softmax activation)
2. **Loss function design**: The core challenge is choosing/designing loss functions that effectively measure similarity between probability distributions:
   - Start with simple losses (MSE, L1) as baselines
   - Experiment with KL Divergence for distribution matching
   - Consider custom weighted combinations
3. **Evaluation pipeline**: Implement multiple distribution comparison metrics (MSE, Total Variation, EMD, KL) to compare model outputs with ground truth
4. **Visualization**: Create visualizations showing input images alongside bar charts of ground truth vs predicted probability distributions

## Model Options

The project allows any CNN or Vision Transformer architecture. Consider:
- Starting with pre-trained models (ResNet, EfficientNet, ViT) for transfer learning
- Fine-tuning with a custom head that outputs 13 probabilities
- The base architecture choice is less critical than the loss function and evaluation design

## Expected Development Workflow

Since no code exists yet, the typical development flow will be:

1. Extract and explore the dataset structure
2. Implement data loading pipeline (PyTorch Dataset/DataLoader)
3. Design/select model architecture with 13-output head
4. Implement multiple loss function variants for experimentation
5. Create training loop with validation monitoring
6. Implement evaluation metrics (multiple distribution comparison measures)
7. Build visualization tools for qualitative assessment
8. Run experiments comparing different loss functions and architectures

## Reference Materials

- `CONTEXT.md`: Full assignment specification
- `assignment/5 - emotion-ranking2.docx`: Detailed assignment document
- `useful_papers/`: Academic papers related to the project
