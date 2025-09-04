# CNN for Medical Diagnosis

A deep learning project that applies Convolutional Neural Networks (CNNs) for medical image classification using the **ChestXRay2017 dataset**.  
The model classifies chest X-ray images into two categories: **Normal** and **Pneumonia**, showcasing the potential of AI in medical diagnostics.

---

## 📌 Project Overview
- Implemented a **MobileNet-inspired CNN** architecture with depthwise separable convolutions, batch normalization, and ReLU activations for computational efficiency.  
- Applied preprocessing techniques including **resizing (224×224), normalization, and data augmentation** to improve model generalization.  
- Addressed **class imbalance** by applying class-weight balancing during training.  
- Leveraged **TensorFlow TPU strategy** for distributed training, enabling large-scale processing of 4,000+ images.  
- Achieved **high diagnostic accuracy (~XX%)**, with balanced precision and recall, demonstrating applicability in real-world digital pathology and radiology workflows.  

---

## ⚙️ Model Architecture
- Custom CNN model inspired by **MobileNet**.
- Key components:
  - Depthwise Separable Convolutions  
  - Batch Normalization  
  - ReLU Activation  
  - Average Pooling  
  - Dense layer with Softmax output  

---

## 🚀 Technologies Used
- **Languages & Libraries:** Python, TensorFlow, Keras, NumPy, OpenCV, Matplotlib, Seaborn  
- **Training:** TPU strategy (Google Colab)  
- **Visualization:** Matplotlib for loss/accuracy plots, confusion matrix  

---

## 📊 Results
- Achieved **~XX% accuracy** on validation set.  
- Precision, Recall, and F1-score demonstrate balanced performance.  
- Confusion matrix confirms effective classification between Normal and Pneumonia.  

---

## 📂 Repository Structure
📦 cnn-medical-diagnosis
┣ 📜 CNN_for_Medical_Diagnosis.ipynb
┣ 📜 README.md
