"""
Evaluation Script for Real-World Custom Trial 2 (Vertical Column on Lined Paper)
================================================================================
Evaluates the user's second real-world handwriting photo (`sample_02.jpeg`):
1. Loads raw image from data/custom_trials/trial_02_vertical_column/sample_02.jpeg.
2. Background subtraction and horizontal rule line suppression.
3. Automatically segments all 10 handwritten digits listed in a single vertical column (0 through 9).
4. Shows both the real photo crop and the 8x8 model representation.
5. Predicts each digit using the pre-trained SVM model.
6. Generates a 10-digit diagnostic gallery saved to:
   results/custom_trials/trial_02_vertical_column/trial_02_evaluation.png.
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
from PIL import Image
import scipy.ndimage as ndi

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))


def evaluate_trial_2(
    image_path=os.path.join(REPO_ROOT, "data", "custom_trials", "trial_02_vertical_column", "sample_02.jpeg"),
    model_path=os.path.join(REPO_ROOT, "models", "best_digits_svm_model.joblib"),
    scaler_path=os.path.join(REPO_ROOT, "models", "minmax_scaler.joblib"),
    output_path=os.path.join(REPO_ROOT, "results", "custom_trials", "trial_02_vertical_column", "trial_02_evaluation.png")
):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Input image not found: {image_path}")
        
    print(f">> [1/4] Loading Trial 2 vertical column photo: '{image_path}'...")
    img = Image.open(image_path)
    arr_rgb = np.array(img, dtype=np.float32)
    gray = np.array(img.convert('L'), dtype=np.float32)
    
    print(">> [2/4] Suppressing horizontal rule lines and estimating background illumination...")
    bg = ndi.gaussian_filter(gray, sigma=20)
    diff = bg - gray
    line_bg = ndi.grey_opening(diff, size=(1, 50))
    ink_isolated = np.clip(diff - line_bg, 0, None)
    
    # Restrict to central handwriting column
    mask = np.zeros_like(ink_isolated, dtype=bool)
    mask[:, 220:480] = (ink_isolated[:, 220:480] > 40)
    
    print(">> [3/4] Segmenting all 10 vertically aligned digits (0 to 9)...")
    dilated = ndi.binary_dilation(mask, iterations=4)
    labeled, num_features = ndi.label(dilated)
    objects = ndi.find_objects(labeled)
    
    boxes = []
    for sl in objects:
        h = sl[0].stop - sl[0].start
        w = sl[1].stop - sl[1].start
        area = np.sum(mask[sl])
        if area > 100 and h > 15:
            boxes.append((sl[0].start, sl[0].stop, sl[1].start, sl[1].stop))
            
    # Sort top to bottom by y coordinate
    boxes = sorted(boxes, key=lambda b: b[0])
    
    if len(boxes) != 10:
        print(f"   Notice: Found {len(boxes)} components. Taking first 10 if available.")
        
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    
    print("\n>> [4/4] Preprocessing and evaluating with trained SVM model:")
    print("=" * 65)
    
    fig, axes = plt.subplots(4, 5, figsize=(15, 11))
    correct_count = 0
    
    for true_digit, (ymin, ymax, xmin, xmax) in enumerate(boxes[:10]):
        col = true_digit % 5
        row_photo = 0 if true_digit < 5 else 2
        row_8x8 = 1 if true_digit < 5 else 3
        
        pad_raw = 15
        y0 = max(0, ymin - pad_raw)
        y1 = min(arr_rgb.shape[0], ymax + pad_raw)
        x0 = max(0, xmin - pad_raw)
        x1 = min(arr_rgb.shape[1], xmax + pad_raw)
        raw_crop = arr_rgb[y0:y1, x0:x1].astype(np.uint8)
        
        crop_mask = mask[ymin:ymax, xmin:xmax]
        crop_thick = ndi.binary_dilation(crop_mask, iterations=3).astype(np.float32) * 255.0
        
        h, w = crop_thick.shape
        size = max(h, w)
        pad = int(size * 0.08)
        tot = size + 2 * pad
        sq = np.zeros((tot, tot), dtype=np.float32)
        sq[pad + (size - h) // 2 : pad + (size - h) // 2 + h,
           pad + (size - w) // 2 : pad + (size - w) // 2 + w] = crop_thick
        
        res = Image.fromarray(sq.astype(np.uint8)).resize((8, 8), Image.Resampling.BILINEAR)
        arr_8 = (np.array(res, dtype=np.float32) / 255.0) * 16.0
            
        flat_scaled = scaler.transform(arr_8.reshape(1, -1))
        pred = model.predict(flat_scaled)[0]
        prob = model.predict_proba(flat_scaled)[0][pred] * 100
        
        is_match = (pred == true_digit)
        if is_match:
            correct_count += 1
            status_text = f"MATCH (Class {pred})"
            status_color = "#2E7D32"
            badge = "CORRECT"
        else:
            status_text = f"MISMATCH (Pred: {pred})"
            status_color = "#C62828"
            badge = "MISMATCH"
            
        print(f"   True Digit: {true_digit} | Model Predicted: {pred} ({prob:5.1f}% conf) | {status_text}")
        
        ax_p = axes[row_photo, col]
        ax_p.imshow(raw_crop)
        ax_p.set_title(f"Handwritten '{true_digit}' (Photo)", fontsize=11, fontweight='bold', pad=4)
        ax_p.axis('off')
        
        ax_m = axes[row_8x8, col]
        ax_m.imshow(arr_8, cmap=plt.cm.gray_r, interpolation='nearest')
        ax_m.set_title(
            f"8×8 Input ➔ Pred: {pred}\n[{badge}] ({prob:.1f}%)",
            fontsize=10.5, fontweight='bold', color=status_color, pad=4
        )
        ax_m.axis('off')
        
    print("=" * 65)
    print(f"🏆 Final Score on Trial 2 (Vertical Column Handwriting): {correct_count}/10 ({correct_count*10}% accuracy)\n")
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.suptitle(
        f"Trial 2 Evaluation: Vertically Arranged Digits 0 to 9 ('sample_02.jpeg')\n"
        f"Overall Accuracy: {correct_count}/10 ({correct_count*10}%) — Green: Correct Match | Red: Mismatch",
        fontsize=14, fontweight='bold', y=0.99
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"🖼️ Saved visual report to: '{output_path}'")
    return correct_count


def main():
    parser = argparse.ArgumentParser(description="Evaluate Real-World Custom Trial 2 (sample_02.jpeg)")
    parser.add_argument("--image", default=os.path.join(REPO_ROOT, "data", "custom_trials", "trial_02_vertical_column", "sample_02.jpeg"),
                        help="Path to input photo")
    parser.add_argument("--output", default=os.path.join(REPO_ROOT, "results", "custom_trials", "trial_02_vertical_column", "trial_02_evaluation.png"),
                        help="Path to output visual report")
    args = parser.parse_args()
    evaluate_trial_2(image_path=args.image, output_path=args.output)


if __name__ == "__main__":
    main()
