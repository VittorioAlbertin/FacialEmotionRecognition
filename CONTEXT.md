# Emotion Recognition Project — Assignment  
**Academic Year:** 2024/2025  

---

## 🎯 Project Objective
Train a **deep learning model for emotion recognition** using a custom dataset.

Unlike standard emotion recognition (where the output is a single label such as “sad”), this project aims to predict a **probability distribution** over a set of emotion classes.

---

## 📂 Dataset
**Dataset:** Custom dataset — [Download here](#)

### Structure
The dataset is divided into **training** and **validation** sets.  
Each folder contains:
- **Images**
- **Annotations**

For each image `xxx`, there is a corresponding annotation file:

```
xxx_prob_rank.txt
```

This file contains **probabilities** for a predefined set of **emotions**.

### Emotion Labels
The order of probabilities corresponds to the following emotions:

```
neutral, happy, sad, surprised, fear, disgust, angry, contempt, serene, contemplative, secure, untroubled, quiet
```

---

## 🧠 Network Model
You can use **any CNN or Vision Transformer** architecture.

- Start with a simple model or a pre-trained one.
- The model should output a **probability vector** of length *13* (number of emotions).

---

## ⚙️ Training Objective
Design and test **custom loss functions** suitable for predicting probability distributions.

### Suggested Loss Functions
- **Simple losses:** L1, MSE  
- **Distribution-based losses:** KL Divergence, Cross-Entropy variants

You are encouraged to experiment and justify your design choices.

---

## 📊 Evaluation

### Quantitative Evaluation
Try various metrics that compare probability distributions:
- **MSE**
- **Total Variation Distance**
- **Earth Mover’s Distance (EMD)**
- **KL Divergence**

### Qualitative Evaluation
Visualize:
- Input face images
- Ground truth probability distributions
- Predicted probability distributions  

This helps assess the **perceptual quality** of the model’s predictions.

---

## 🧪 Summary
| Component | Description |
|------------|--------------|
| **Task** | Predict emotion probability distributions |
| **Input** | Face image |
| **Output** | 13-dimensional probability vector |
| **Goal** | Design effective loss and evaluation metrics |
| **Evaluation** | Quantitative (metrics) + Qualitative (visualizations) |

---

## 📝 Notes
- Creativity and experimentation are encouraged.  
- Focus on interpreting and analyzing results.  
- Emphasize both **mathematical justification** and **empirical findings**.

---

*End of Assignment*
