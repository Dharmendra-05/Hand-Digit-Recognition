# Contributing to Handwritten Digit Recognition

Thank you for your interest in contributing to **Handwritten Digit Recognition**! Whether you are fixing bugs, improving documentation, adding new machine learning models, or suggesting enhancements, all contributions are welcome.

Please take a moment to review these guidelines before getting started.

---

## 📜 Code of Conduct

By participating in this project, you agree to abide by our [Code of Conduct](CODE_OF_CONDUCT.md). Please treat all contributors with kindness, empathy, and respect.

---

## 🛠️ Getting Started

### 1. Fork and Clone the Repository
Fork the repository on GitHub, then clone your fork locally:
```bash
git clone https://github.com/<your-username>/Hand-Digit-Recognition.git
cd Hand-Digit-Recognition
```

### 2. Set Up a Virtual Environment
We recommend using Python 3.10+ in a dedicated virtual environment:
```bash
# Windows (PowerShell)
python -m venv venv
.\venv\Scripts\Activate.ps1

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
Install all required libraries using `requirements.txt`:
```bash
pip install -r requirements.txt
```

---

## 🚀 Development Workflow

### Creating a Feature Branch
Create a new branch with a descriptive name representing your change:
```bash
git checkout -b feature/your-feature-name
# or for bug fixes:
git checkout -b fix/issue-description
```

### Code Style & Best Practices
- **PEP 8**: Follow standard Python style guidelines.
- **Documentation**: Provide clear comments and docstrings for newly introduced functions or classes.
- **Image Artifacts**: If your changes produce or modify plots, ensure they save to the dedicated `images/` directory.
- **Model Exports**: If adding new trained models, save them to the `models/` directory using `joblib`.

### Running Verification Tests
Before submitting changes, test that both the script and notebook work as expected:
```bash
# Test the standalone CLI pipeline
python main.py

# Test notebook generation and execution
python generate_notebook.py
```

---

## 💡 How You Can Contribute

* **New Classifier Implementations**: Experiment with Convolutional Neural Networks (CNNs using PyTorch/TensorFlow), XGBoost, or ensemble voting classifiers.
* **Feature Engineering**: Implement novel spatial filters, edge detection, or custom dimensionality reduction techniques.
* **Interactive UI**: Build a Streamlit, Gradio, or Tkinter drawing canvas enabling users to draw digits with a mouse and receive real-time predictions.
* **Documentation & Tutorials**: Improve explanations, add mathematical formula derivations, or translate docs.

---

## 📥 Submitting a Pull Request (PR)

1. **Commit your changes**: Write clear, imperative commit messages:
   ```bash
   git commit -m "Add Streamlit interactive drawing canvas for digit prediction"
   ```
2. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```
3. **Open a Pull Request**:
   - Go to the original repository on GitHub and click **Compare & pull request**.
   - Provide a concise description of your changes, what problem they solve, and attach sample screenshots/plots if visual features were modified.
   - Link any related issues (e.g., `Closes #12`).

---

## 💬 Getting Help & Questions

If you have questions, encounter an issue, or want to propose a large architectural change before writing code, please [open an issue](https://github.com/Dharmendra-05/Hand-Digit-Recognition/issues) or reach out via email at **dharmendrapradhan737@gmail.com**.
