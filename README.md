# 🖋️ Handwritten Digit Recognition
### Benchmarking Controlled Laboratory Baselines vs. Real-World Handwritten Digit Realities (Trial 1)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.5+-orange.svg)](https://scikit-learn.org/)
[![Status](https://img.shields.io/badge/Benchmark-99.17%25-brightgreen.svg)](#-laboratory-benchmark-performance)
[![Trial 1 Real World](https://img.shields.io/badge/Trial%201%20Real%20World-50.0%25-critical.svg)](#-the-reality-gap-benchmark-vs-real-world-domain-shift-trial-1)

An end-to-end Machine Learning pipeline, interactive Jupyter Notebook, and computer vision evaluation suite for handwritten digit recognition ($0 - 9$).

This repository benchmarks ideal laboratory conditions against real-world camera captures, contrasting **Scikit-Learn Digits** ($8 \times 8$ pixels, **99.17% accuracy**) and **UCI PenDigits** (16-feature trajectories, **98.26% accuracy**) with **Custom Trial 1**—real-world smartphone handwriting photographed on lined notebook paper, showcasing empirical **domain shift** and practical computer vision preprocessing.

---

## 📌 Project Architecture & Two-World Philosophy

In machine learning research, controlled datasets often report >99% test accuracy. However, deploying those models on raw, uncurated human handwriting frequently reveals severe vulnerabilities. This repository cleanly divides data and evaluation into two distinct domains:

```
                      ┌───────────────────────────────────────────────┐
                      │         Handwritten Digit Recognition         │
                      └───────────────────────┬───────────────────────┘
                                              │
                      ┌───────────────────────┴───────────────────────┐
                      ▼                                               ▼
          [ Ideal Benchmark World ]                       [ Real-World Trial World ]
       • Controlled normalization                      • Uncurated camera captures
       • Centered 8x8 pixel matrices                   • Notebook ruling lines & paper textures
       • Scikit-Learn Digits (99.17%)                  • Ballpoint pen stroke variations
       • UCI PenDigits Trajectory (98.26%)             • Custom Trial 1 (`self.jpeg` -> 50%)
```

1. **The Ideal Benchmark (`data/benchmark/` & `results/benchmark/`)**:
   - Standardized Scikit-Learn Digits ($8 \times 8$ grayscale, 1,797 samples).
   - UCI Pen-Based Trajectory Digits (16 coordinates, 10,992 samples).
   - High test accuracy (**99.17%** for pixels, **98.26%** for pen trajectories).
2. **Real-World Custom Trials (`data/custom_trials/` & `results/custom_trials/`)**:
   - **Trial 1 (`trial_01_phone_lined_paper`)**: An unedited photo of 10 digits (`0` through `9`) written in blue ballpoint ink on red-lined notebook paper taken with a smartphone camera (`self.jpeg`).
   - Evaluates real-world domain adaptation, color space segmentation, morphological dilation, and classification bottlenecks.

---

## 🔬 The Reality Gap: Benchmark vs. Real-World Domain Shift (Trial 1)

| Dimension | Ideal Benchmark (Scikit-Learn) | Real-World Trial 1 (`self.jpeg`) |
| :--- | :--- | :--- |
| **Source Image** | Digitized, pre-centered $8 \times 8$ bitmaps | High-res smartphone photo ($4000 \times 3000$ px) |
| **Writing Tool** | NIST standardization (thick uniform strokes) | Thin blue ballpoint pen (~1–2 px stroke) |
| **Canvas / Background**| Uniform solid dark background (value 0) | Cream paper with red notebook ruling lines |
| **Preprocessing** | None required (pre-extracted features) | RGB color thresholding + Morphological Dilation |
| **Model Test Accuracy** | **99.17% (357 / 360 correct)** | **50.0% (5 / 10 correct)** |

### 🔍 Key Findings from Trial 1:
1. **Rule Line Noise Elimination**: Naive grayscale conversion fails because horizontal notebook lines merge with the digits. Utilizing color difference filtering (`(b - r) > 10` & `(r+g+b)/3 < 160`) isolates blue ballpoint ink while completely rejecting red rule lines.
2. **Stroke Thickness Discrepancy**: NIST/Scikit-Learn digits have thick strokes relative to an $8 \times 8$ grid. A thin ballpoint pen line thins out or breaks when downsampled to $8 \times 8$. Applying 3 iterations of **morphological dilation** was critical to preserve character topology.
3. **Per-Digit Evaluation on `self.jpeg`**:
   - **Correctly Classified (5/10)**: `0`, `1`, `2`, `3`, `7` (digits with simple strokes or distinct closed contours).
   - **Misclassified (5/10)**: `4`, `5`, `6`, `8`, `9` (digits whose internal loops and crosses collapsed into vertical strokes under $8 \times 8$ compression, leading to biased predictions toward classes `1` and `4`).

![Trial 1 Evaluation](results/custom_trials/trial_01_phone_lined_paper/self_handwriting_evaluation.png)

---

## 📈 Laboratory Benchmark Performance

Evaluated using 5-Fold Stratified Cross-Validation and held-out test splits:

| Model | 5-Fold CV Accuracy (%) | Test Accuracy (%) | Weighted F1-Score |
| :--- | :---: | :---: | :---: |
| **Support Vector Classifier (SVC - RBF)** | **98.82%** | **99.17%** | **0.9917** |
| **Multi-Layer Perceptron (MLP Neural Net)** | 97.63% | 98.61% | 0.9861 |
| **K-Nearest Neighbors (KNN, k=5)** | 98.68% | 98.33% | 0.9834 |
| **Random Forest (100 Trees)** | 97.29% | 96.11% | 0.9611 |
| **Logistic Regression (Multinomial)** | 96.38% | 95.83% | 0.9583 |

> **Best Tuned Model**: `SVC(C=5.0, gamma=0.05, kernel='rbf')` achieves **>99% accuracy** on the Scikit-Learn test set and **98.26%** on UCI PenDigits.

---

## 📁 Repository Structure

```
Hand-Digit-Recognition/
├── handwritten_digit_recognition.ipynb  # Primary interactive Jupyter Notebook (Executed with all outputs)
├── requirements.txt                     # Project dependencies
├── README.md                            # Comprehensive documentation & findings
├── LICENSE                              # MIT License
├── CONTRIBUTING.md                      # Contribution guidelines
├── CODE_OF_CONDUCT.md                   # Community guidelines
│
├── data/                                # Cleanly categorized raw inputs
│   ├── benchmark/                       # Controlled lab datasets
│   │   ├── uci_pendigits/               # UCI pendigits.tra, pendigits.tes (10,992 samples)
│   │   ├── zip/                         # Archived dataset source archives
│   │   └── references/                  # Dataset reference URLs & documentation
│   └── custom_trials/                   # Real-world and external test inputs
│       ├── trial_01_phone_lined_paper/  # self.jpeg (user phone camera handwriting)
│       └── samples/                     # Synthetic test sample digits
│
├── results/                             # Categorized analytical outputs & charts
│   ├── benchmark/                       # Benchmark plots (confusion matrix, PCA/t-SNE, demo)
│   │   ├── confusion_matrix.png
│   │   ├── digit_recognition_demo.png
│   │   ├── mean_digits_template.png
│   │   ├── model_comparison.png
│   │   ├── pca_tsne_projections.png
│   │   └── pen_trajectories.png
│   └── custom_trials/                   # Custom trial diagnostics
│       ├── trial_01_phone_lined_paper/  # self_handwriting_evaluation.png (10-digit grid)
│       └── samples/                     # custom_prediction_sample_digit_3.png
│
├── scripts/                             # Executable Python workflows
│   ├── run_benchmark.py                 # Full benchmark training & evaluation suite
│   ├── evaluate_trial_1.py              # Real-world Trial 1 multi-digit evaluator
│   ├── predict_single_image.py          # Single image inference CLI tool
│   └── generate_notebook.py             # Notebook regeneration & execution script
│
└── models/                              # Serialized model & scaler joblib artifacts
    ├── best_digits_svm_model.joblib
    ├── minmax_scaler.joblib
    ├── pendigits_svm_model.joblib
    └── pendigits_scaler.joblib
```

---

## ⚙️ Installation & Usage

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Benchmark Pipeline
Train the benchmark models, run hyperparameter tuning, evaluate metrics, and save figures:
```bash
python scripts/run_benchmark.py
```

### 3. Evaluate Real-World Custom Trial 1 (`self.jpeg`)
Run the color space segmentation, morphological dilation, and multi-digit evaluation on the phone-captured lined paper:
```bash
python scripts/evaluate_trial_1.py
```
This produces `results/custom_trials/trial_01_phone_lined_paper/self_handwriting_evaluation.png`.

### 4. Test Any Arbitrary Single Image
Test inference on an individual handwritten digit image with automatic centering and polarity detection:
```bash
python scripts/predict_single_image.py data/custom_trials/samples/sample_digit_3.png
```
Or provide any custom `.png` / `.jpg` path:
```bash
python scripts/predict_single_image.py path/to/your/digit.png
```

### 5. Interactive Jupyter Notebook
Open the notebook in VS Code, JupyterLab, or Jupyter Notebook:
```bash
jupyter notebook handwritten_digit_recognition.ipynb
```
To regenerate and execute all notebook cells automatically:
```bash
python scripts/generate_notebook.py
```

---

## 🖼️ Benchmark Visual Recognizer

When running inference on ideal digits, the pipeline produces:
1. Rendered $8 \times 8$ grayscale digit image.
2. Highlighted keypad button (`[0] [1] [2] [3] [*4*] [5] [6] [7] [8] [9]`).
3. Softmax probability confidence distribution (>97% certainty).

![Benchmark Demo](results/benchmark/digit_recognition_demo.png)

---

## 📄 License & Community

* **License**: Released under the [MIT License](LICENSE).
* **Code of Conduct**: Please review our [Code of Conduct](CODE_OF_CONDUCT.md).
* **Contributing**: Check out our [Contributing Guidelines](CONTRIBUTING.md) to get involved.

