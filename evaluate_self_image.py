"""
Evaluation Script for User's Handwriting (self.jpeg)
===================================================
Processes the user's actual handwriting photo `self.jpeg`:
1. Corrects orientation (counter-clockwise 90° rotation).
2. Filters out red lined-paper rules and isolates blue pen ink.
3. Automatically segments all 10 handwritten digits (0 through 9).
4. Shows both the real photo crop and the 8x8 model representation.
5. Predicts each digit using the trained SVM model.
6. Generates a 10-digit diagnostic gallery saved to images/self_handwriting_evaluation.png.
"""

import os
import sys
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


def evaluate_user_image(image_path="self.jpeg",
                        model_path="models/best_digits_svm_model.joblib",
                        scaler_path="models/minmax_scaler.joblib",
                        output_path="images/self_handwriting_evaluation.png"):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
        
    print(f">> [1/4] Loading and orienting '{image_path}'...")
    # Rotate 90° counter-clockwise to make notebook text upright
    img = Image.open(image_path).rotate(90, expand=True)
    arr = np.array(img, dtype=np.float32)
    
    print(">> [2/4] Isolating blue pen ink and suppressing notebook lines...")
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    brightness = (r + g + b) / 3.0
    blue_ink = (brightness < 160) & ((b - r) > 10)
    
    print(">> [3/4] Segmenting all 10 handwritten digits...")
    dilated = ndi.binary_dilation(blue_ink, iterations=6)
    labeled, num_feats = ndi.label(dilated)
    objects = ndi.find_objects(labeled)
    
    boxes = []
    for sl in objects:
        if np.sum(blue_ink[sl]) > 100:
            boxes.append((sl[0].start, sl[0].stop, sl[1].start, sl[1].stop))
            
    # Split into Row 1 (digits 0-7, Y < 300) and Row 2 (digits 8-9, Y >= 300)
    row1 = sorted([b for b in boxes if (b[0] + b[1]) / 2 < 300], key=lambda b: b[2])
    row2 = sorted([b for b in boxes if (b[0] + b[1]) / 2 >= 300], key=lambda b: b[2])
    digits_order = row1 + row2
    
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    
    print("\n>> [4/4] Preprocessing and evaluating with trained SVM model:")
    print("=" * 60)
    
    # Create 4x5 subplot: for each digit, Row 1 has photo crop, Row 2 has 8x8 matrix
    fig, axes = plt.subplots(4, 5, figsize=(15, 11))
    
    correct_count = 0
    
    for true_digit, (ymin, ymax, xmin, xmax) in enumerate(digits_order[:10]):
        # Grid placement:
        # Digits 0-4: rows 0 (photo) & 1 (8x8)
        # Digits 5-9: rows 2 (photo) & 3 (8x8)
        col = true_digit % 5
        row_photo = 0 if true_digit < 5 else 2
        row_8x8 = 1 if true_digit < 5 else 3
        
        # Raw color photo crop with small context margin
        pad_raw = 15
        y0 = max(0, ymin - pad_raw)
        y1 = min(arr.shape[0], ymax + pad_raw)
        x0 = max(0, xmin - pad_raw)
        x1 = min(arr.shape[1], xmax + pad_raw)
        raw_crop = arr[y0:y1, x0:x1].astype(np.uint8)
        
        # Dilate thin ballpoint strokes to match NIST training stroke thickness
        crop_mask = blue_ink[ymin:ymax, xmin:xmax]
        crop_thick = ndi.binary_dilation(crop_mask, iterations=3).astype(np.float32) * 255.0
        
        # Center in square with 8% margin
        h, w = crop_thick.shape
        size = max(h, w)
        pad = int(size * 0.08)
        tot = size + 2 * pad
        sq = np.zeros((tot, tot), dtype=np.float32)
        sq[pad + (size - h) // 2 : pad + (size - h) // 2 + h,
           pad + (size - w) // 2 : pad + (size - w) // 2 + w] = crop_thick
        
        # Resize to 8x8
        res = Image.fromarray(sq.astype(np.uint8)).resize((8, 8), Image.Resampling.BILINEAR)
        arr_8 = np.array(res, dtype=np.float32)
        if arr_8.max() > 0:
            arr_8 = (arr_8 / arr_8.max()) * 16.0
            
        flat_scaled = scaler.transform(arr_8.reshape(1, -1))
        pred = model.predict(flat_scaled)[0]
        prob = model.predict_proba(flat_scaled)[0][pred] * 100
        
        is_match = (pred == true_digit)
        if is_match:
            correct_count += 1
            status_text = f"MATCH (Class {pred})"
            status_color = "#2E7D32" # Green
            badge = "CORRECT"
        else:
            status_text = f"MISMATCH (Pred: {pred})"
            status_color = "#C62828" # Red
            badge = "MISMATCH"
            
        print(f"   True Digit: {true_digit} | Model Predicted: {pred} ({prob:5.1f}% conf) | {status_text}")
        
        # Plot Photo Crop
        ax_p = axes[row_photo, col]
        ax_p.imshow(raw_crop)
        ax_p.set_title(f"Handwritten '{true_digit}' (Photo)", fontsize=11, fontweight='bold', pad=4)
        ax_p.axis('off')
        
        # Plot 8x8 Model View
        ax_m = axes[row_8x8, col]
        ax_m.imshow(arr_8, cmap=plt.cm.gray_r, interpolation='nearest')
        ax_m.set_title(
            f"8×8 Input ➔ Pred: {pred}\n[{badge}] ({prob:.1f}%)",
            fontsize=10.5, fontweight='bold', color=status_color, pad=4
        )
        ax_m.axis('off')
        
    print("=" * 60)
    print(f"🏆 Final Score on your handwritten digits: {correct_count}/10 ({correct_count*10}% accuracy)\n")
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.suptitle(
        f"Evaluation on User's Photo ('self.jpeg') Across All 10 Digits (0 to 9)\n"
        f"Overall Accuracy: {correct_count}/10 ({correct_count*10}%) — Green: Correct Match | Red: Mismatch",
        fontsize=14, fontweight='bold', y=0.99
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"🖼️ Saved visual report to: '{output_path}'")
    return correct_count


if __name__ == "__main__":
    evaluate_user_image()
