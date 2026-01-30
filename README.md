# A Comparative Analysis of Custom Neural Architectures for Multiclass Livestock Classification

This project explores the efficacy of different deep learning paradigms—Convolutional Recurrent Neural Networks (CRNN), Vision Transformers (ViT), and Graph Convolutional Neural Networks (GCNN)—for the multiclass classification of livestock. Using a custom dataset of cattle, pigs, sheep, and horses, we evaluate how spatial features, sequential dependencies, and attention mechanisms impact classification accuracy in agricultural contexts.

---

##  Live Demo & Deployment

The top-performing model (**Veterinary CRNN**) has been deployed as a real-time diagnostic tool. You can interact with the model via the web interface below:

* **Hugging Face Space:** [Cattle Disease Classifier](https://huggingface.co/spaces/hamzabinabdulmajeed/Cattle-Disease-Classifier)

---

## Project Overview

The objective of this research is to compare three distinct architectural approaches to image classification. Each model was trained for **25 epochs** using **PyTorch** on **GPU P100** hardware to ensure consistent benchmarking.

### Core Architectures:

* **CRNN:** Combines CNN feature extraction with RNN sequence modeling to capture spatial and temporal-like dependencies.
* **ViT (Vision Transformer):** Utilizes self-attention mechanisms to process image patches, moving away from traditional convolutional layers.
* **GCNN (Graph Convolutional Neural Network):** Maps spatial features onto a graph structure to exploit connectivity and geometric relationships.

---

## Dataset & Exploratory Data Analysis (EDA)

The dataset consists of **22,342 total instances** across 5 classes. A significant class imbalance is present, with Sheep being the majority class and "Class_4" representing a tiny minority.

### Instance Distribution

| Class ID | Animal Class | Total Instances |
| --- | --- | --- |
| 0 | Cow | 4,146 |
| 1 | Pig | 545 |
| 2 | Sheep | 12,388 |
| 3 | Horse | 5,239 |
| 4 | Class_4 (Minority) | 24 |
| **Total** |  | **22,342** |

### Data Splitting

The data was partitioned into Train, Validation, and Test sets as follows:

* **Train:** 16,457 instances (4,867 images)
* **Valid:** 4,012 instances (1,195 images)
* **Test:** 1,873 instances (508 images)

---

## Performance Summary

The **CRNN** architecture emerged as the superior model, significantly outperforming both the Transformer-based and Graph-based approaches in terms of accuracy and loss convergence.

### Final Comparative Results

| Model Architecture | Final Test Acc | Best Val Acc | Best Val Loss | Key Structural Feature |
| --- | --- | --- | --- | --- |
| **CRNN** | **94.49%** | **95.90%** | **0.1600** | CNN Features + RNN Sequence |
| **ViT** | 83.86% | 85.69% | 0.4690 | Self-Attention Mechanism |
| **GCNN** | 72.05% | 72.97% | 0.6286 | Spatial Graph Connectivity |

---

## Individual Model Evaluations

### 1. CRNN (Top Performer)

* **Best Validation Accuracy:** 95.90% (Epoch 23)
* **Final Test Accuracy:** 94.49%
* **Analysis:** The hybrid nature of the CRNN allowed for robust feature extraction via convolutions while the recurrent layers (Bi-LSTM) refined the classification by treating spatial features as a sequence.

### 2. Vision Transformer (ViT)

* **Best Validation Accuracy:** 85.69% (Epoch 22)
* **Best Validation Loss:** 0.4690 (Epoch 18)
* **Final Test Accuracy:** 83.86%
* **Analysis:** While effective, the ViT required more data or longer training to match the inductive bias inherent in the CRNN for this specific dataset size.

### 3. Graph CNN (GCNN)

* **Best Validation Accuracy:** 72.97% (Epoch 24)
* **Best Validation Loss:** 0.6286 (Epoch 24)
* **Final Test Accuracy:** 72.05%
* **Analysis:** The GCNN struggled with the high variance in image-based livestock data compared to the more rigid feature maps used by the CRNN and ViT.

---

## Links & Resources

* **Dataset:** [Kaggle Livestock Dataset](https://www.kaggle.com/datasets/amiteshpatra07/cattle-dataset-pig-sheep-cow-horse)
* **EDA Notebook:** [View Exploratory Data Analysis](https://www.kaggle.com/code/hamzabinbutt/eda-vet)
* **CRNN Implementation:** [View Kaggle Notebook](https://www.kaggle.com/code/hamzabinbutt/crnn-cat)
* **ViT Implementation:** [View Kaggle Notebook](https://www.kaggle.com/code/hamzabinbutt/vit-cat)
* **GCNN Implementation:** [View Kaggle Notebook](https://www.kaggle.com/code/hamzabinbutt/gcnn-cat)

---

## Environment & Requirements

* **Framework:** PyTorch
* **Hardware:** P100 GPU
* **Training Duration:** 25 Epochs per model

---

**Would you like me to generate a specific "How to Run" section if you want people to be able to clone and run your training scripts locally?**
