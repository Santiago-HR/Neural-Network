<div align="center">

<h1>MNIST Neural Network Classification</h1>
<p>Fully connected and convolutional neural networks trained on handwritten digit recognition — from 93% to 99% F1.</p>

<img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
<img src="https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white"/>
<img src="https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=keras&logoColor=white"/>
<img src="https://img.shields.io/badge/CNN-99%25_F1-2ea44f?style=for-the-badge"/>

</div>

---

## Overview

Builds, trains, and evaluates progressively more complex neural network models on the **MNIST handwritten digits dataset** — starting from a minimal fully connected network and ending with a custom CNN achieving ~99% average F1 score.

---

## Dataset

**MNIST Handwritten Digits** — 70,000 grayscale images across 10 digit classes.

| Split | Images | Shape |
|-------|--------|-------|
| Train | 60,000 | 28×28 |
| Test  | 10,000 | 28×28 |

Pixel values are normalized to `[0, 1]` before training. Reshaped to `(784,)` for fully connected models and `(28, 28, 1)` for CNN models.

---

## Part 1 — Data Loading & Preprocessing

- Loaded MNIST via TensorFlow/Keras
- Normalized pixel values (uint8 → float32, ÷ 255)
- Reshaped for both FC and CNN input formats
- Visualized sample training images

---

## Part 2 — Baseline Fully Connected Network

**Architecture:**

```
Input (784) → Dense(8, ReLU) → Dense(10, Softmax)
```

| Setting | Value |
|---------|-------|
| Optimizer | Adam |
| Loss | Sparse Categorical Crossentropy |
| Batch size | 32 |
| Epochs | 10 |

**Result: ~93% accuracy**

Evaluation included training/validation curves, classification report, confusion matrix, and correct/incorrect prediction visualizations.

---

## Part 3 — Larger Hidden Layer

Increased hidden layer capacity to analyze the effect on convergence and accuracy.

**Architecture:**

```
Input (784) → Dense(128, ReLU) → Dense(10, Softmax)
```

<img width="1076" alt="model summary" src="https://github.com/user-attachments/assets/f99766e2-fe99-49f2-95fb-a192ca57934d"/>
<img width="544" alt="results" src="https://github.com/user-attachments/assets/f39c34ea-3b90-4ecd-8704-88a641dc4fcf"/>
<img width="2214" alt="training curves" src="https://github.com/user-attachments/assets/e0f836eb-98b2-44b6-a1d1-e33d7f568226"/>

**Result: ~98% accuracy** — faster convergence and lower loss vs. the 8-neuron baseline.

---

## Part 4 — Custom CNN for 99% F1

**Architecture:**

```
Conv2D(32, 3×3, ReLU) → MaxPool(2×2)
Conv2D(64, 3×3, ReLU) → MaxPool(2×2)
Flatten
Dense(128, ReLU)
Dense(10, Softmax)
```

**Result: ~99% accuracy · Average F1 ≈ 0.99**

<img width="2120" alt="CNN results" src="https://github.com/user-attachments/assets/fd79c464-c5bb-450f-9f59-136934966da3"/>
<img width="1292" alt="confusion matrix" src="https://github.com/user-attachments/assets/064bfc8e-b1fa-4f01-bc85-3bedce0a013f"/>
<img width="1584" alt="predictions" src="https://github.com/user-attachments/assets/a1d0a658-f0ed-400e-ac71-ac892a9f4643"/>
<img width="1998" alt="classification report" src="https://github.com/user-attachments/assets/3b0ab2b9-92df-4c2b-b4c8-b230d85e2669"/>
<img width="2136" alt="training curves" src="https://github.com/user-attachments/assets/8f25bd64-282a-4a78-b5a0-d75055df18ce"/>

---

<div align="center">

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-informational?style=flat-square)
![Seaborn](https://img.shields.io/badge/Seaborn-informational?style=flat-square)

</div>
