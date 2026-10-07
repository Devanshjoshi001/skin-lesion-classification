# Skin Lesion Classification: Machine Learning vs. Deep Learning on HAM10000

A comparative study of four classifiers for multi-class skin lesion classification from dermoscopic images: Random Forest, Support Vector Machine (SVM), a custom Convolutional Neural Network (CNN), and ResNet50 transfer learning. Developed as an M.Sc. Computer Science major project at the Department of Computer Science, Central University of Rajasthan, under the supervision of Dr. Ravi Raj Chaudhary.

> This project is for research and educational purposes only. It is not a medical device and must not be used for diagnosis.

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Results](#results)
- [Key Observations](#key-observations)
- [Limitations](#limitations)
- [Future Work](#future-work)
- [Repository Structure](#repository-structure)
- [How to Run](#how-to-run)
- [References](#references)
- [Author](#author)

## Overview

Different types of skin lesions often look visually similar, which makes manual classification slow and dependent on specialist availability. This project compares traditional machine learning on handcrafted image features with deep learning on raw images, to understand how each approach behaves on an imbalanced medical image dataset.

**Objectives**

- Preprocess dermoscopic images from the HAM10000 dataset.
- Extract handcrafted features (HOG and color histograms) for classical models.
- Train and evaluate Random Forest, SVM, a custom CNN, and ResNet50 transfer learning.
- Compare the models using accuracy, precision, recall, F1-score, confusion matrices, and AUC-ROC.

## Dataset

The project uses **HAM10000** ("Human Against Machine with 10,000 training images"), which contains 10,015 dermoscopic images across seven lesion categories.

| Label | Lesion type | Images | Share |
|---|---|---:|---:|
| nv | Melanocytic nevi | 6,705 | 66.9% |
| mel | Melanoma | 1,113 | 11.1% |
| bkl | Benign keratosis-like lesions | 1,099 | 11.0% |
| bcc | Basal cell carcinoma | 514 | 5.1% |
| akiec | Actinic keratoses | 327 | 3.3% |
| vasc | Vascular lesions | 142 | 1.4% |
| df | Dermatofibroma | 115 | 1.1% |

The dataset is heavily imbalanced: about two thirds of the images are melanocytic nevi. The images are not included in this repository. Download them from the [Kaggle dataset page](https://www.kaggle.com/datasets/kmader/skin-lesion-analysis-toward-melanoma-detection) and see the original paper in the references.

![Class distribution](assets/class_distribution.png)

## Methodology

1. **Preprocessing.** Images were resized, normalized, and converted to grayscale where needed. Inputs were downsampled to 64x64 pixels.
2. **Feature extraction (classical models).**
   - Histogram of Oriented Gradients (HOG) for edge, shape, and texture.
   - Color histograms for pigmentation and color distribution.
   - HOG and color features were combined into one feature vector per image.
3. **Feature scaling.** Features were standardized with `StandardScaler`. For the SVM, Principal Component Analysis (PCA) reduced dimensionality.
4. **Models.**

   | Model | Input | Notes |
   |---|---|---|
   | Random Forest | Handcrafted features | Tuned number of trees, maximum depth, and minimum samples per split |
   | SVM | Scaled and PCA-reduced features | RBF kernel |
   | Custom CNN | Images | Convolution, max pooling, dropout, flatten, and dense layers with softmax output; trained from scratch |
   | ResNet50 | Images | ImageNet pretrained weights, with added dense and softmax layers; categorical cross-entropy loss and Adam optimizer |

5. **Evaluation.** Models were tested on unseen images and scored with accuracy, precision, recall, F1-score (macro and weighted), confusion matrix, and AUC-ROC.

## Results

| Model | Accuracy | Macro F1 | Weighted F1 | Mean AUC-ROC | Training time |
|---|---:|---:|---:|---:|---:|
| Random Forest | 67.50% | 13.75% | 55.34% | 0.8281 | 11.6 sec |
| SVM (RBF) | 70.90% | 29.16% | 66.37% | 0.8171 | 35.81 sec |
| Custom CNN | 47.21% | 39.53% | 52.09% | 0.8514 | 26.63 min |
| **ResNet50 (transfer learning)** | **71.06%** | **72.70%** | **73.85%** | **0.9589** | 39.17 min |

**ResNet50 per-class results**

| Class | Precision | Recall | F1-score | AUC |
|---|---:|---:|---:|---:|
| Actinic keratoses | 0.67 | 0.88 | 0.76 | 0.978 |
| Basal cell carcinoma | 0.67 | 0.87 | 0.76 | 0.975 |
| Benign keratosis | 0.57 | 0.76 | 0.65 | 0.936 |
| Dermatofibroma | 0.73 | 1.00 | 0.85 | 1.000 |
| Melanoma | 0.33 | 0.76 | 0.46 | 0.885 |
| Melanocytic nevi | 0.98 | 0.67 | 0.79 | 0.956 |
| Vascular lesions | 0.80 | 0.86 | 0.83 | 0.982 |

The per-class values are read from the figures in the project report.

![ResNet50 ROC curves](assets/resnet50_roc_curves.png)

## Key Observations

- **Accuracy alone is misleading on this dataset.** Because about 67% of images are melanocytic nevi, a model that predicts "nevi" for everything scores close to 67% accuracy. Random Forest reached 67.50% accuracy but only 13.75% macro F1, and its confusion matrix shows it predicted almost every image as nevi.
- **SVM improved accuracy but still missed small classes.** It scored an F1 of 0 on dermatofibroma and vascular lesions.
- **ResNet50 gave the best balanced performance.** Its macro F1 (72.70%) and mean AUC-ROC (0.9589) are the highest, and it was the only model that performed reasonably across the minority classes. Its accuracy is only slightly above the SVM's, so the gain comes from the minority classes.
- **The custom CNN underperformed on accuracy** (47.21%). It was trained from scratch on a small, imbalanced dataset with limited tuning, though its macro F1 was higher than Random Forest and SVM.
- **Melanoma was the hardest class** for the best model: recall was 0.76, but precision was only 0.33, so many other lesions were flagged as melanoma.
- **Classical models are faster and easier to interpret**, taking seconds to train compared with tens of minutes for the deep models.

## Limitations

- Images were downsampled to 64x64 pixels, which removes fine dermoscopic detail such as pigment networks and globules.
- Class imbalance was not corrected with resampling, class weights, or data augmentation.
- Each model was trained and evaluated in a separate run. A single shared stratified test split would make the comparison strictly like-for-like.
- Results come from one dataset and have not been validated on external data or by clinicians.

## Future Work

- Train at higher resolution (224x224 or 256x256).
- Add data augmentation and class weighting to address imbalance.
- Try other pretrained architectures (EfficientNet, DenseNet) and Vision Transformers.
- Add attention modules such as CBAM.
- Combine models in an ensemble.
- Optimize explicitly for melanoma sensitivity, since missed melanomas are the costliest errors.

## Repository Structure

```text
skin-lesion-classification/
├── notebook/        # Jupyter notebook: preprocessing, feature extraction, training, evaluation
├── report/          # Full project report (PDF)
├── assets/          # Figures used in this README
└── README.md
```

## How to Run

**Requirements:** Python 3.10 or later and Jupyter.

```bash
pip install numpy pandas scikit-learn opencv-python matplotlib jupyter
```

Also install the deep learning library imported in the notebook, which is needed for the CNN and ResNet50 models.

1. Download HAM10000 and place the images and metadata where the notebook expects them.
2. Open the notebook in the `notebook/` folder.
3. Run the cells from top to bottom. The ResNet50 and CNN models are much slower than Random Forest and SVM, so a GPU is recommended.

## References

1. Tschandl, P., Rosendahl, C., and Kittler, H. (2018). The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions. *Scientific Data*.
2. He, K., Zhang, X., Ren, S., and Sun, J. (2016). Deep residual learning for image recognition. *CVPR*.
3. Esteva, A., et al. (2017). Dermatologist-level classification of skin cancer with deep neural networks. *Nature*.
4. Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research*.

## Author

**Devansh Joshi**
M.Sc. Computer Science, Central University of Rajasthan
Supervisor: Dr. Ravi Raj Chaudhary

- LinkedIn: [linkedin.com/in/contactdevanshJoshi](https://linkedin.com/in/contactdevanshJoshi)
- GitHub: [github.com/Devanshjoshi001](https://github.com/Devanshjoshi001)
