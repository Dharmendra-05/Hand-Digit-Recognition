"""
Handwritten Digit Recognition Benchmark Pipeline
=================================================
Trains, cross-validates, and evaluates 5 Machine Learning classifiers
on Scikit-Learn's Optical Recognition of Handwritten Digits dataset (Ideal Case),
and exports diagnostic benchmark results.

Output:
- Saves confusion matrix and keypad demo plots into results/benchmark/
- Saves trained models and scalers into models/
"""

import os
import sys
import time
import argparse
import warnings

# Ensure UTF-8 output encoding on Windows console
if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

warnings.filterwarnings('ignore')

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns

from sklearn import datasets
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


# Resolve repository root
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))


def load_and_preprocess_data():
    """Loads digits dataset, scales features, and performs stratified split."""
    print(">> [1/5] Loading Scikit-Learn Digits Dataset (Benchmark)...")
    digits = datasets.load_digits()
    
    print(f"   Total samples: {len(digits.images)}")
    print(f"   Image resolution: {digits.images[0].shape[0]}x{digits.images[0].shape[1]}")
    print(f"   Features per sample: {digits.data.shape[1]}")
    print(f"   Target classes: {np.unique(digits.target).tolist()}")
    
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(digits.data)
    y = digits.target
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.20, random_state=42, stratify=y
    )
    
    return digits, scaler, X_train, X_test, y_train, y_test


def benchmark_models(X_train, y_train, X_test, y_test):
    """Trains and compares 5 machine learning models."""
    print("\n>> [2/5] Training & Benchmarking Models with 5-Fold Stratified CV...")
    
    models = {
        "Support Vector Machine (SVC)": SVC(kernel='rbf', gamma='scale', C=10.0, probability=True, random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "K-Nearest Neighbors (KNN)": KNeighborsClassifier(n_neighbors=5),
        "Multi-Layer Perceptron (MLP)": MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42)
    }
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = []
    fitted_models = {}
    
    for name, clf in models.items():
        t0 = time.time()
        cv_scores = cross_val_score(clf, X_train, y_train, cv=cv, scoring='accuracy')
        clf.fit(X_train, y_train)
        duration = time.time() - t0
        
        y_pred = clf.predict(X_test)
        test_acc = accuracy_score(y_test, y_pred)
        
        fitted_models[name] = clf
        results.append({
            "Model": name,
            "CV Accuracy (%)": round(cv_scores.mean() * 100, 2),
            "Test Accuracy (%)": round(test_acc * 100, 2),
            "Fit Time (s)": round(duration, 3)
        })
        print(f"   ✓ {name:<30} | CV: {cv_scores.mean()*100:5.2f}% | Test: {test_acc*100:5.2f}% | Time: {duration:5.3f}s")
        
    df_results = pd.DataFrame(results).sort_values(by="Test Accuracy (%)", ascending=False).reset_index(drop=True)
    return df_results, fitted_models


def tune_best_model(X_train, y_train, X_test, y_test):
    """Tunes hyper-parameters for Support Vector Classifier via GridSearchCV."""
    print("\n>> [3/5] Hyperparameter Tuning for Support Vector Classifier (SVC)...")
    
    param_grid = {
        'C': [1, 5, 10, 20],
        'gamma': ['scale', 0.1, 0.05, 0.01],
        'kernel': ['rbf']
    }
    
    grid = GridSearchCV(
        SVC(probability=True, random_state=42),
        param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1
    )
    grid.fit(X_train, y_train)
    
    best_clf = grid.best_estimator_
    y_pred = best_clf.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)
    
    print(f"   🏆 Best Params: {grid.best_params_}")
    print(f"   🏆 Best 5-Fold CV Accuracy: {grid.best_score_*100:.2f}%")
    print(f"   🏆 Final Test Accuracy:     {test_acc*100:.2f}%")
    
    return best_clf, y_pred


def evaluate_and_plot(y_test, y_pred, save_path):
    """Generates detailed classification report and saves confusion matrix heatmap."""
    print("\n>> [4/5] Detailed Model Evaluation & Confusion Matrix...")
    print(classification_report(y_test, y_pred, digits=4))
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, square=True,
                xticklabels=range(10), yticklabels=range(10))
    plt.title("Confusion Matrix for Digit Recognition (Benchmark)", fontsize=13, fontweight='bold')
    plt.xlabel("Predicted Digit", fontsize=11, fontweight='bold')
    plt.ylabel("True Digit", fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"   Saved confusion matrix to '{save_path}'")


def visualize_digit_prediction(image_2d, model, scaler, save_path):
    """
    Renders diagram matching benchmark prompt:
    8x8 digit image pointing to identified keypad digit button [0-9].
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    flat = image_2d.reshape(1, -1)
    flat_scaled = scaler.transform(flat)
    
    prediction = model.predict(flat_scaled)[0]
    probabilities = model.predict_proba(flat_scaled)[0]
    confidence = probabilities[prediction] * 100
    
    fig = plt.figure(figsize=(11, 5))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.2, 1])
    
    # Left subplot: digit image and 0-9 buttons
    ax_img = fig.add_subplot(gs[0])
    ax_img.set_xlim(-1, 10)
    ax_img.set_ylim(-2, 10)
    ax_img.axis('off')
    
    # Image
    ax_img.imshow(image_2d, cmap=plt.cm.gray_r, interpolation='nearest', extent=[2.2, 6.8, 3.5, 8.5])
    rect = patches.Rectangle((2.15, 3.45), 4.7, 5.1, fill=False, edgecolor='#333333', linewidth=1.5)
    ax_img.add_patch(rect)
    
    # Buttons 0 to 9
    for digit in range(10):
        bx = digit * 1.0 - 0.5
        by = -0.5
        is_pred = (digit == prediction)
        
        box = patches.FancyBboxPatch(
            (bx + 0.05, by - 0.4), 0.85, 0.85,
            boxstyle="round,pad=0.15",
            facecolor='#6F42C1' if is_pred else '#F8F9FA',
            edgecolor='#495057' if not is_pred else '#4A148C',
            linewidth=2 if is_pred else 1.2
        )
        ax_img.add_patch(box)
        
        ax_img.text(
            bx + 0.475, by + 0.02, str(digit),
            ha='center', va='center',
            fontsize=13, fontweight='bold',
            color='white' if is_pred else '#333333'
        )
        
        if is_pred:
            ax_img.annotate(
                '',
                xy=(bx + 0.475, by + 0.5),
                xytext=(4.5, 3.4),
                arrowprops=dict(arrowstyle="->", color="#222222", lw=2, mutation_scale=15)
            )

    ax_img.text(4.5, 9.3, "Identify hand written digits recognition", ha='center', fontsize=13, fontweight='bold')
    ax_img.text(4.5, -1.5, f"Identified Digit: {prediction} (Confidence: {confidence:.2f}%)", ha='center', fontsize=11, fontweight='bold', color='#4A148C')

    # Right subplot: Probability bar chart
    ax_prob = fig.add_subplot(gs[1])
    colors = ['#6F42C1' if i == prediction else '#90CAF9' for i in range(10)]
    bars = ax_prob.bar(range(10), probabilities * 100, color=colors, edgecolor='black', width=0.6)
    ax_prob.set_xticks(range(10))
    ax_prob.set_xlabel("Digit Class", fontsize=11, fontweight='bold')
    ax_prob.set_ylabel("Probability (%)", fontsize=11, fontweight='bold')
    ax_prob.set_title("Prediction Confidence Distribution", fontsize=12, fontweight='bold')
    ax_prob.set_ylim(0, 105)
    
    for bar in bars:
        h = bar.get_height()
        if h > 1:
            ax_prob.text(bar.get_x() + bar.get_width()/2.0, h + 2, f"{h:.1f}%", ha='center', va='bottom', fontsize=8.5, fontweight='bold')
            
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"   Saved prediction demonstration to '{save_path}'")


def main():
    parser = argparse.ArgumentParser(description="Handwritten Digit Recognition Benchmark Pipeline")
    parser.add_argument("--save-model", action="store_true", default=True, help="Save trained model to models/ folder")
    parser.add_argument("--results-dir", type=str, default=os.path.join(REPO_ROOT, "results", "benchmark"),
                        help="Target directory to save benchmark results and plots")
    parser.add_argument("--models-dir", type=str, default=os.path.join(REPO_ROOT, "models"),
                        help="Target directory to save serialized models")
    args = parser.parse_args()
    
    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs(args.models_dir, exist_ok=True)
    
    digits, scaler, X_train, X_test, y_train, y_test = load_and_preprocess_data()
    df_results, _ = benchmark_models(X_train, y_train, X_test, y_test)
    best_model, y_pred = tune_best_model(X_train, y_train, X_test, y_test)
    
    cm_path = os.path.join(args.results_dir, "confusion_matrix.png")
    evaluate_and_plot(y_test, y_pred, save_path=cm_path)
    
    print("\n>> [5/5] Recreating Benchmark Demonstration (Digit 4)...")
    idx_4 = np.where(digits.target == 4)[0][0]
    demo_path = os.path.join(args.results_dir, "digit_recognition_demo.png")
    visualize_digit_prediction(digits.images[idx_4], best_model, scaler, save_path=demo_path)
    
    if args.save_model:
        joblib.dump(best_model, os.path.join(args.models_dir, "best_digits_svm_model.joblib"))
        joblib.dump(scaler, os.path.join(args.models_dir, "minmax_scaler.joblib"))
        print(f"\n✅ Saved model and scaler to '{args.models_dir}' directory.")
        
    print("\n🎉 Benchmark pipeline execution completed successfully!")


if __name__ == "__main__":
    main()
