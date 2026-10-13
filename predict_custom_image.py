"""
Custom Handwritten Digit Predictor
===================================
Allows users to pass their own handwritten digit image (PNG, JPG, etc.),
applies preprocessing (inversion, auto-crop, centering, and 8x8 resizing),
and predicts the digit using the trained Support Vector Machine model.

Usage:
    python predict_custom_image.py path/to/your_image.png
"""

import os
import sys
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
import matplotlib.pyplot as plt
from PIL import Image, ImageOps, ImageFilter


def preprocess_image(image_path):
    """
    Preprocesses an arbitrary handwritten image into an 8x8 matrix
    compatible with the scikit-learn digits format:
    1. Grayscale conversion
    2. Polarity detection & inversion (scikit-learn uses 0=background, 16=stroke)
    3. Bounding box cropping & aspect-ratio-preserving centering
    4. Anti-aliased downsampling to 8x8 pixels
    5. Normalization to [0, 16] range
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at: {image_path}")
        
    orig_img = Image.open(image_path).convert('RGB')
    gray_img = orig_img.convert('L')
    
    arr = np.array(gray_img, dtype=np.float32)
    
    # 1. Detect background polarity: if borders are bright, invert so background is dark
    border_pixels = np.concatenate([
        arr[0, :], arr[-1, :], arr[:, 0], arr[:, -1]
    ])
    if np.median(border_pixels) > 127:
        arr = 255.0 - arr
        
    # 2. Subtract background noise baseline
    p_low = np.percentile(arr, 10)
    arr = np.clip(arr - p_low, 0, 255)
    
    # 3. Find bounding box of the digit stroke
    stroke_mask = arr > (arr.max() * 0.25)
    if not stroke_mask.any():
        raise ValueError("Image appears to be empty or uniform (no digit strokes detected).")
        
    rows = np.any(stroke_mask, axis=1)
    cols = np.any(stroke_mask, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    
    digit_crop = arr[rmin:rmax+1, cmin:cmax+1]
    
    # 4. Center inside a square canvas with 15% margin
    h, w = digit_crop.shape
    size = max(h, w)
    pad = int(size * 0.18)
    total_size = size + 2 * pad
    
    square_canvas = np.zeros((total_size, total_size), dtype=np.float32)
    y_offset = pad + (size - h) // 2
    x_offset = pad + (size - w) // 2
    square_canvas[y_offset:y_offset+h, x_offset:x_offset+w] = digit_crop
    
    # 5. Smooth slightly before downsampling to prevent aliasing
    pil_square = Image.fromarray(square_canvas.astype(np.uint8))
    pil_square = pil_square.filter(ImageFilter.GaussianBlur(radius=max(1, total_size // 40)))
    
    # 6. Resize to 8x8 using high-quality Bilinear/Lanczos interpolation
    resized_8x8 = pil_square.resize((8, 8), Image.Resampling.BILINEAR)
    arr_8x8 = np.array(resized_8x8, dtype=np.float32)
    
    # 7. Scale intensities to [0, 16] as expected by the digits dataset
    if arr_8x8.max() > 0:
        arr_8x8 = (arr_8x8 / arr_8x8.max()) * 16.0
        
    return orig_img, arr_8x8


def predict_digit(image_path, model_path="models/best_digits_svm_model.joblib",
                  scaler_path="models/minmax_scaler.joblib",
                  output_dir="images"):
    """Loads model, runs prediction, and plots diagnostic visualization."""
    orig_img, arr_8x8 = preprocess_image(image_path)
    
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    
    flat = arr_8x8.reshape(1, -1)
    flat_scaled = scaler.transform(flat)
    
    prediction = model.predict(flat_scaled)[0]
    probabilities = model.predict_proba(flat_scaled)[0]
    confidence = probabilities[prediction] * 100
    
    print("\n" + "=" * 50)
    print(f"👉 Input Image:       {image_path}")
    print(f"🎯 Predicted Digit:   {prediction}")
    print(f"📊 Model Confidence:  {confidence:.2f}%")
    print("=" * 50)
    
    # Generate visualization
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    save_path = os.path.join(output_dir, f"custom_prediction_{base_name}.png")
    
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), gridspec_kw={'width_ratios': [1, 1, 1.4]})
    
    # 1. Original image
    axes[0].imshow(orig_img)
    axes[0].set_title(f"Original Custom Image\n({orig_img.size[0]}×{orig_img.size[1]} px)", fontsize=11, fontweight='bold', pad=10)
    axes[0].axis('off')
    
    # 2. Preprocessed 8x8 matrix
    im = axes[1].imshow(arr_8x8, cmap=plt.cm.gray_r, interpolation='nearest')
    axes[1].set_title("Preprocessed 8×8 Grid\n(Model Input View)", fontsize=11, fontweight='bold', pad=10)
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04, label="Pixel Intensity (0–16)")
    
    # 3. Probability distribution
    colors = ['#6F42C1' if i == prediction else '#90CAF9' for i in range(10)]
    bars = axes[2].bar(range(10), probabilities * 100, color=colors, edgecolor='black', width=0.6)
    axes[2].set_xticks(range(10))
    axes[2].set_xlabel("Digit Class", fontsize=11, fontweight='bold')
    axes[2].set_ylabel("Probability (%)", fontsize=11, fontweight='bold')
    axes[2].set_title(f"Predicted: Class {prediction} ({confidence:.1f}% Confidence)", fontsize=12, fontweight='bold', color='#4A148C', pad=10)
    axes[2].set_ylim(0, 105)
    
    for bar in bars:
        h = bar.get_height()
        if h > 1:
            axes[2].text(bar.get_x() + bar.get_width()/2.0, h + 2, f"{h:.1f}%", ha='center', va='bottom', fontsize=8.5, fontweight='bold')
            
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"🖼️ Saved diagnostic chart to: '{save_path}'")
    
    return prediction, confidence, save_path


def main():
    parser = argparse.ArgumentParser(description="Predict handwritten digit from a custom image file")
    parser.add_argument("image_path", help="Path to your handwritten digit image (PNG, JPG, etc.)")
    parser.add_argument("--model", default="models/best_digits_svm_model.joblib", help="Path to trained model")
    parser.add_argument("--scaler", default="models/minmax_scaler.joblib", help="Path to scaler")
    parser.add_argument("--outdir", default="images", help="Directory to save prediction visualization")
    args = parser.parse_args()
    
    predict_digit(args.image_path, model_path=args.model, scaler_path=args.scaler, output_dir=args.outdir)


if __name__ == "__main__":
    main()
