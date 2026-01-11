# MNIST Neural Network Classification 

This repository contains my implementation for a neural network, which focuses on building, training, and evaluating neural network models using the **MNIST handwritten digits dataset**.  
The assignment explores progressively more complex models, starting from a simple fully connected neural network and ending with a custom convolutional neural network (CNN) achieving ~99% average F1 score.

---

## Dataset
- **MNIST Handwritten Digits**
- 60,000 training images
- 10,000 testing images
- Image size: 28 × 28 grayscale
- Labels: digits 0–9

Pixel values are normalized to the range `[0, 1]` before training.

---

## Part 1: Data Loading and Preprocessing
- Loaded MNIST using TensorFlow/Keras
- Converted pixel values from integers to float32
- Normalized pixel values
- Reshaped images:
  - `(784,)` for fully connected models
  - `(28, 28, 1)` for CNN models
- Visualized sample training images

---

## Part 2: Building and Training a Neural Network Model (20 Marks)
A simple **fully connected neural network** was implemented to classify handwritten digits.

### Model Architecture
- Input layer: 784 neurons
- Hidden layer: 8 neurons (ReLU)
- Output layer: 10 neurons (Softmax)

### Training Setup
- Optimizer: Adam
- Loss function: Sparse Categorical Crossentropy
- Batch size: 32
- Epochs: 10
- ModelCheckpoint used to save the best model based on validation accuracy

### Evaluation
- Training and validation accuracy/loss plots
- Classification report (precision, recall, F1-score)
- Confusion matrix visualization
- Correct and incorrect prediction visualizations

This model achieved approximately **93% accuracy**, as shown in the classification report and confusion matrix included in the assignment PDF :contentReference[oaicite:0]{index=0}.

---

## Part 3: Experimenting with a Larger Hidden Layer (20 Marks)
To analyze the effect of model capacity, the hidden layer size was increased.
<img width="1076" height="248" alt="image" src="https://github.com/user-attachments/assets/f99766e2-fe99-49f2-95fb-a192ca57934d" />

### Model Architecture
- Input layer: 784 neurons
- Hidden layer: 128 neurons (ReLU)
- Output layer: 10 neurons (Softmax)
<img width="544" height="176" alt="image" src="https://github.com/user-attachments/assets/f39c34ea-3b90-4ecd-8704-88a641dc4fcf" />

<img width="2214" height="804" alt="image" src="https://github.com/user-attachments/assets/e0f836eb-98b2-44b6-a1d1-e33d7f568226" />



### Results
- Improved accuracy compared to the smaller network
- Faster convergence
- Reduced loss
- Achieved approximately **98% accuracy**


---

## Part 4: Custom Neural Network for 99% Average F1 Score (10 Marks)
A **Convolutional Neural Network (CNN)** was designed to achieve high classification performance.

### Model Architecture
- Conv2D (32 filters, 3×3) + ReLU
- MaxPooling2D (2×2)
- Conv2D (64 filters, 3×3) + ReLU
- MaxPooling2D (2×2)
- Flatten
- Dense (128 neurons, ReLU)
- Dense (10 neurons, Softmax)

### Performance
- Achieved approximately **99% accuracy**
- Average F1-score ≈ **0.99**
- Strong performance across all digit classes


<img width="2120" height="932" alt="image" src="https://github.com/user-attachments/assets/fd79c464-c5bb-450f-9f59-136934966da3" />

<img width="1292" height="1110" alt="image" src="https://github.com/user-attachments/assets/064bfc8e-b1fa-4f01-bc85-3bedce0a013f" />

<img width="1584" height="1058" alt="image" src="https://github.com/user-attachments/assets/a1d0a658-f0ed-400e-ac71-ac892a9f4643" />

<img width="1998" height="950" alt="image" src="https://github.com/user-attachments/assets/3b0ab2b9-92df-4c2b-b4c8-b230d85e2669" />

<img width="2136" height="1058" alt="image" src="https://github.com/user-attachments/assets/8f25bd64-282a-4a78-b5a0-d75055df18ce" />


## Technologies Used
- Python
- TensorFlow / Keras
- NumPy
- Matplotlib
- scikit-learn
- seaborn
