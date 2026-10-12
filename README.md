# 🖋️ Handwritten Digit Recognition

An end-to-end Machine Learning pipeline and interactive Jupyter Notebook for recognizing handwritten digits (0 through 9) from the **Scikit-Learn Digits Dataset** ([Optical Recognition of Handwritten Digits](https://scikit-learn.org/1.5/auto_examples/datasets/plot_digits_last_image.html)) and the **UCI Pen-Based Handwritten Digits Dataset** ([Pen-Based Recognition](https://archive.ics.uci.edu/dataset/81/pen+based+recognition+of+handwritten+digits)).

---

## 📌 Project Overview

This project reproduces and expands upon the classic Scikit-Learn handwritten digit recognition benchmark. Given an $8 \times 8$ grayscale image or an 8-point stroke trajectory of a handwritten digit, the model accurately predicts which number ($0 - 9$) it represents, replicating the visual identification interface shown in the project prompt:

```
          [ 8x8 Grayscale Image ]
                   │
                   ▼ (Model Inference)
     ┌──────────────────────────────────────────────┐
     │  0   1   2   3  [4]  5   6   7   8   9       │
     └───────────────────▲──────────────────────────┘
                         │
              Identified Digit: Class 4
```

### 🔗 Official References & Dataset Sources
* **Scikit-Learn Digits Dataset**: [plot_digits_last_image.html](https://scikit-learn.org/1.5/auto_examples/datasets/plot_digits_last_image.html)
* **Scikit-Learn Classification Example**: [plot_digits_classification.html](https://scikit-learn.org/stable/auto_examples/classification/plot_digits_classification.html)
* **UCI Pen-Based Digits**: [UCI Machine Learning Repository - Pen-Based Digits](https://archive.ics.uci.edu/dataset/81/pen+based+recognition+of+handwritten+digits) (Included in `assets/Dataset/`)

---

## 🚀 Key Features

1. **Jupyter Notebook (`handwritten_digit_recognition.ipynb`)**:
   - **Part I: Pixel-Based Recognition (Scikit-Learn Digits)**:
     - Complete exploratory data analysis (EDA) with pixel intensity heatmaps and class distributions.
     - **PCA & t-SNE** 2D manifold visualizations demonstrating cluster separation in latent space.
     - Comparison across **5 diverse machine learning algorithms**:
       - **Support Vector Classifier (SVC)** (RBF Kernel)
       - **Logistic Regression** (Multinomial with L2 regularization)
       - **Random Forest Classifier** (100 Decision Trees)
       - **K-Nearest Neighbors (KNN)** ($k=5$)
       - **Multi-Layer Perceptron (MLP)** (Neural Network)
     - **Hyperparameter Optimization** using `GridSearchCV`.
     - In-depth evaluation: Precision, Recall, F1-scores, Confusion Matrix heatmap, and Misclassified Sample inspection.
     - **Custom Diagrammatic Recognizer**: Recreates the visual digit display and keypad button interface with confidence scores.
   - **Part II: Trajectory-Based Recognition (UCI PenDigits)**:
     - 8-point resampled stroke trajectory visualization for digits 0–9.
     - Writer-independent generalization testing on 3,498 unseen test samples (**98.26% accuracy**).
   - **Model Serialization**: Saves trained weights and scalers to `models/` with `joblib`.

2. **Standalone Python Script (`main.py`)**:
   - Command-line runner executing data preparation, model training, cross-validation, grid search, evaluation, and visualization.

---

## 📊 Dataset Comparison

| Property | Scikit-Learn Digits | UCI PenDigits (`assets/Dataset/`) |
| :--- | :--- | :--- |
| **Input Representation** | $8 \times 8$ Grayscale Pixel Matrix | 8 Resampled $(x, y)$ Pen Trajectory Coordinates |
| **Feature Dimensions** | 64 numerical values ($0 - 16$) | 16 numerical values ($0 - 100$) |
| **Training Samples** | 1,437 samples (80% split) | 7,494 samples (from 30 writers) |
| **Test Samples** | 360 samples (20% split) | 3,498 samples (from 14 unseen writers) |
| **Best Test Accuracy** | **99.17%** (SVC RBF) | **98.26%** (SVC RBF) |

---

## 📈 Model Performance Benchmark (Scikit-Learn Digits)

Models evaluated using 5-Fold Stratified Cross-Validation on training data and tested on the held-out test set:

| Model | 5-Fold CV Accuracy (%) | Test Accuracy (%) | Weighted F1-Score |
| :--- | :---: | :---: | :---: |
| **Support Vector Classifier (SVC - RBF)** | **98.82%** | **99.17%** | **0.9917** |
| **Multi-Layer Perceptron (MLP Neural Net)** | 97.63% | 98.61% | 0.9861 |
| **K-Nearest Neighbors (KNN, k=5)** | 98.68% | 98.33% | 0.9834 |
| **Random Forest (100 Trees)** | 97.29% | 96.11% | 0.9611 |
| **Logistic Regression (Multinomial)** | 96.38% | 95.83% | 0.9583 |

> **Best Tuned Model**: `SVC(C=5.0, gamma=0.05, kernel='rbf')` achieves **>99% accuracy** on the test set.

---

## 📁 Repository Structure

```
Hand-Digit-Recognition/
├── handwritten_digit_recognition.ipynb  # Primary interactive Jupyter Notebook (Executed with all outputs)
├── main.py                              # Standalone Python training & evaluation pipeline
├── generate_notebook.py                 # Automated generator & executor for the notebook
├── requirements.txt                     # Project dependencies
├── README.md                            # Comprehensive documentation
├── assets/
│   ├── Dataset/                         # UCI PenDigits dataset (pendigits.tra, pendigits.tes)
│   └── links/                           # Reference links
├── images/                              # Dedicated directory for generated plots & diagrams
│   ├── confusion_matrix.png
│   ├── digit_recognition_demo.png
│   ├── mean_digits_template.png
│   ├── model_comparison.png
│   ├── pca_tsne_projections.png
│   └── pen_trajectories.png
├── models/                              # Serialized trained model & scaler artifacts
│   ├── best_digits_svm_model.joblib
│   ├── minmax_scaler.joblib
│   ├── pendigits_svm_model.joblib
│   └── pendigits_scaler.joblib
└── requirements.txt                     # Project dependencies
```

---

## ⚙️ Installation & Usage

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Jupyter Notebook
Open the notebook in VS Code, JupyterLab, or Jupyter Notebook:
```bash
jupyter notebook handwritten_digit_recognition.ipynb
```
All cells come pre-executed with generated plots, tables, and inference demonstrations saved to `images/`.

### 3. Run via Command Line
Execute the standalone Python pipeline directly:
```bash
python main.py
```
This will:
- Train and cross-validate all models
- Optimize the Support Vector Classifier
- Output the full classification report
- Save generated figures to the dedicated `images/` directory
- Export trained artifacts to the `models/` directory

---

## 🖼️ Visual Recognition Demo

When running inference on a sample digit (e.g. digit `4`), the pipeline produces:
1. The rendered $8 \times 8$ grayscale digit image.
2. A highlighted keypad button matching the prediction (`[0] [1] [2] [3] [*4*] [5] [6] [7] [8] [9]`).
3. An arrow directly connecting the digit to the detected class.
4. A full probability confidence distribution bar chart indicating prediction certainty (>97%).
