<h1 align="center">🖼️ Computer Vision – Fruits & Vegetables Classification</h1>

<p align="center">
  Academic project developed in <strong>Python + TensorFlow/Keras</strong> for the subject
  <em>Percepción Computacional</em>.  
  The project addresses a <strong>multi-class image classification</strong> problem using
  convolutional neural networks, comparing a model trained <strong>from scratch</strong> with a
  <strong>transfer learning + fine-tuning</strong> approach based on MobileNetV2.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white&style=for-the-badge"/>
  <img src="https://img.shields.io/badge/TensorFlow-FF6F00?logo=tensorflow&logoColor=white&style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Computer%20Vision-CNN-blueviolet?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Transfer%20Learning-MobileNetV2-success?style=for-the-badge"/>
</p>

<p align="center">
  <a href="https://deepwiki.com/maria2332/classification-of-fruits-and-vegetables-images" target="_blank">
    <img src="https://img.shields.io/badge/DeepWiki-Documentation-purple?style=for-the-badge"/>
  </a>
</p>

---

## 📚 Project Documentation (External)

An automatically generated documentation view of this repository is available via DeepWiki:

👉 https://deepwiki.com/maria2332/classification-of-fruits-and-vegetables-images

---

## 📌 Project Overview

This project focuses on the classification of **fruits and vegetables images** into
<strong>36 different categories</strong> using deep learning techniques.

Two different strategies are explored and compared:

1. **CNN trained from scratch**
2. **Transfer learning with MobileNetV2 + fine-tuning**

The objective is to analyze learning behavior, convergence, generalization capability,
and the impact of transfer learning on performance.

---

## 🗂️ Dataset

- **Domain:** Fruits and vegetables images  
- **Number of classes:** 36  
- **Class distribution:** Balanced  
- **Data split:**  
  - Training  
  - Validation  
  - Test  

Each class contains the same number of samples, allowing a fair comparison across categories.

---

## 🧠 Strategies Implemented

### 🔹 Strategy 1 – CNN from Scratch

A custom convolutional neural network was designed and trained from scratch, including:

- Convolutional blocks with Batch Normalization
- MaxPooling
- Dropout regularization
- Global Average Pooling
- Dense classifier

📉 **Observed behavior:**
- Slow convergence
- Accuracy around **0.4–0.5**
- Clear signs of **underfitting**

This result motivates the use of more advanced techniques.

---

### 🔹 Strategy 2 – Transfer Learning + Fine-tuning (MobileNetV2)

A pretrained **MobileNetV2** model (ImageNet weights) was used as feature extractor:

1. **Stage 1:**  
   - Base model frozen  
   - Training only the classification head  

2. **Stage 2 (Fine-tuning):**  
   - Partial unfreezing of the last layers  
   - Lower learning rate  
   - Gradual optimization of high-level features  

✔️ Data augmentation, learning rate scheduling and early stopping were applied.

📈 **Results:**
- Test accuracy ≈ **0.93**
- Stable convergence
- Small gap between training and validation
- No relevant overfitting detected

---

## 📊 Evaluation

- Accuracy, precision, recall and F1-score per class
- Confusion matrix analysis
- Visual comparison of predictions between:
  - CNN from scratch
  - Transfer learning model
 
The transfer learning approach significantly outperforms the model trained from scratch,
both in accuracy and generalization.

---

## 🖼️ Qualitative Results

Random test images are displayed comparing:

- Ground truth label
- Prediction from scratch model
- Prediction from transfer learning model

This visual analysis highlights the robustness and consistency of the transfer learning approach,
especially for visually similar classes.

---

## 🛠️ Technologies Used

- Python  
- TensorFlow / Keras  
- NumPy  
- Matplotlib  
- Scikit-learn  

---

## ✅ Conclusions

- Training a CNN from scratch is insufficient for complex multi-class visual tasks with limited data.
- Transfer learning with MobileNetV2 provides strong feature representations and fast convergence.
- Fine-tuning further improves performance without overfitting.
- The final model generalizes well across all classes.
