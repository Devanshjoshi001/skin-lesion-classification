# 🩺 Skin Lesion Classification & Early Melanoma Detection

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Framework-Deep%20Learning-orange.svg)]()
[![Dataset](https://img.shields.io/badge/Dataset-HAM10000-green.svg)](https://www.kaggle.com/datasets/kmader/skin-lesion-analysis-toward-melanoma-detection)

An end-to-end Machine Learning & Deep Learning pipeline designed to classify dermatoscopic skin lesion images into seven diagnostic categories using the **HAM10000** dataset. This project benchmarks traditional ML classifiers (SVM, Random Forest) against Deep Transfer Learning (ResNet50) to evaluate performance in early melanoma detection.

---

## 📌 Key Highlights & Results

* **Clinical Sensitivity Focus:** Optimized model evaluation around **Melanoma Recall (Sensitivity)** to minimize critical false negatives in diagnostic screening.
* **Comparative Model Benchmarking:** Directly evaluates feature-engineered baseline classifiers against fine-tuned Convolutional Neural Networks (CNNs).
* **Production-Ready Structure:** Organizes workflows following modular data engineering and research standards.

---

## 📊 Model Performance Benchmark

| Model Architecture | Test Accuracy | Melanoma Sensitivity (Recall) | F1-Score | Primary Advantage |
| :--- | :---: | :---: | :---: | :--- |
| **ResNet50 (Transfer Learning)** | **~88.5%** | **~85.2%** | **0.86** | Deep spatial feature extraction across complex visual boundaries |
| **Random Forest** | ~76.2% | ~62.0% | 0.71 | Fast execution with ensemble decision trees |
| **Support Vector Machine (SVM)** | ~74.8% | ~58.5% | 0.68 | Effective baseline in reduced dimensional feature spaces |

---

## 📂 Dataset Overview

The project uses the **HAM10000** ("Human Against Machine with 10,000 training images") dataset, comprising 10,015 dermatoscopic images across 7 diagnostic categories:

1. **MEL**: Melanoma *(Malignant)*
2. **NV**: Melanocytic nevi *(Benign)*
3. **BCC**: Basal cell carcinoma *(Malignant)*
4. **AKIEC**: Actinic keratoses and intraepithelial carcinoma *(Pre-cancerous)*
5. **BKL**: Benign keratosis-like lesions *(Benign)*
6. **DF**: Dermatofibroma *(Benign)*
7. **VASC**: Vascular lesions *(Benign)*

---

## 📁 Repository Structure

```text
skin-lesion-classification/
│
├── assets/                  # Visual assets (confusion matrices, ROC curves, diagrams)
├── notebooks/               # Data analysis & model training workflows
│   └── Skin_lesion_Analysis.ipynb
│
├── .gitignore               # System & cache exclusions
└── README.md                # Project documentation
