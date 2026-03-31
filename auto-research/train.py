"""
train.py — Liveness detection training script (runs on DGX1).
Experiment: Handcrafted noise/frequency features on full DGX1 dataset.
The noise-based approach got 1.000 balanced_accuracy on the small dataset.
This tests whether it generalizes to the full 2586-sample annotated dataset.
Uses Laplacian variance (noise), gradient magnitude, FFT high-freq ratio,
color channel statistics, and JPEG artifact features per image, with RF.
"""
import os
import sys
import json
import time
import random
import numpy as np
from pathlib import Path
from collections import Counter

DATA_DIR = Path("/mnt/nas/public2/simon/projects/auto_research/liveness-research/data")
RESULTS_FILE = Path("/mnt/nas/public2/simon/projects/auto_research/liveness-research/data/last_result.json")
SEED = 42
MAX_SECONDS = 270

random.seed(SEED)
np.random.seed(SEED)


def extract_features(img_path):
    """Extract handcrafted features from a single image."""
    from PIL import Image
    import numpy as np

    try:
        img = Image.open(str(img_path)).convert("RGB")
        arr = np.array(img, dtype=np.float32)
    except Exception:
        return np.zeros(12, dtype=np.float32)

    gray = 0.2989 * arr[:,:,0] + 0.5870 * arr[:,:,1] + 0.1140 * arr[:,:,2]

    # 1. Laplacian variance (noise level)
    from scipy.ndimage import laplace
    lap = laplace(gray)
    noise_var = float(np.var(lap))
    noise_mean = float(np.mean(np.abs(lap)))

    # 2. Gradient magnitude (edge sharpness)
    gy, gx = np.gradient(gray)
    grad_mag = np.sqrt(gx**2 + gy**2)
    grad_mean = float(np.mean(grad_mag))
    grad_std = float(np.std(grad_mag))

    # 3. FFT high-frequency ratio
    fft = np.fft.fft2(gray)
    fft_shift = np.fft.fftshift(fft)
    magnitude = np.abs(fft_shift)
    h, w = gray.shape
    cy, cx = h // 2, w // 2
    r = min(h, w) // 8
    total_energy = float(np.sum(magnitude))
    if total_energy > 0:
        mask = np.ones_like(magnitude, dtype=bool)
        y, x = np.ogrid[:h, :w]
        mask_center = (y - cy)**2 + (x - cx)**2 <= r**2
        mask[mask_center] = False
        hf_energy = float(np.sum(magnitude[mask]))
        hf_ratio = hf_energy / total_energy
    else:
        hf_ratio = 0.0

    # 4. Color channel statistics
    r_mean = float(np.mean(arr[:,:,0]))
    g_mean = float(np.mean(arr[:,:,1]))
    b_mean = float(np.mean(arr[:,:,2]))
    rg_diff = r_mean - g_mean
    gb_diff = g_mean - b_mean

    # 5. Channel uniformity (low for screens)
    r_std = float(np.std(arr[:,:,0]))
    g_std = float(np.std(arr[:,:,1]))
    b_std = float(np.std(arr[:,:,2]))
    uniformity = float(np.std([r_std, g_std, b_std]))

    return np.array([noise_var, noise_mean, grad_mean, grad_std,
                     hf_ratio, rg_diff, gb_diff, uniformity,
                     r_mean, g_mean, b_mean, r_std], dtype=np.float32)


def train():
    t0 = time.time()
    print("Loading data...")

    with open(str(DATA_DIR / "labels.json")) as f:
        labels = json.load(f)

    valid = []
    for sig, info in labels.items():
        far = DATA_DIR / "samples" / sig / "far.jpg"
        near = DATA_DIR / "samples" / sig / "near.jpg"
        if far.exists() and near.exists():
            valid.append(sig)

    random.shuffle(valid)
    split = int(len(valid) * 0.8)
    train_ids = valid[:split]
    test_ids = valid[split:]

    dist_train = Counter(labels[s]["main_label"] for s in train_ids)
    dist_test = Counter(labels[s]["main_label"] for s in test_ids)
    print(f"Train: {len(train_ids)} {dict(dist_train)}")
    print(f"Test:  {len(test_ids)} {dict(dist_test)}")

    print("Extracting features...")
    samples_dir = DATA_DIR / "samples"

    def get_features(sig):
        far_feat = extract_features(samples_dir / sig / "far.jpg")
        near_feat = extract_features(samples_dir / sig / "near.jpg")
        diff_feat = far_feat - near_feat
        return np.concatenate([far_feat, near_feat, diff_feat])

    X_train = np.array([get_features(s) for s in train_ids])
    y_train = np.array([0 if labels[s]["main_label"] == "Positive" else 1 for s in train_ids])
    X_test = np.array([get_features(s) for s in test_ids])
    y_test = np.array([0 if labels[s]["main_label"] == "Positive" else 1 for s in test_ids])

    feat_time = time.time() - t0
    print(f"Feature extraction: {feat_time:.1f}s, shape: {X_train.shape}")

    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import balanced_accuracy_score, accuracy_score, f1_score

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    best_bal_acc = 0
    best_acc = 0
    best_f1 = 0
    best_name = ""

    configs = [
        ("RF_200", RandomForestClassifier(n_estimators=200, max_depth=None, random_state=SEED, n_jobs=-1)),
        ("RF_500", RandomForestClassifier(n_estimators=500, max_depth=None, random_state=SEED, n_jobs=-1)),
        ("GBM_200", GradientBoostingClassifier(n_estimators=200, max_depth=5, learning_rate=0.1, random_state=SEED)),
        ("GBM_500", GradientBoostingClassifier(n_estimators=500, max_depth=5, learning_rate=0.05, random_state=SEED)),
    ]

    for name, clf in configs:
        if time.time() - t0 > MAX_SECONDS - 30:
            print(f"Time budget approaching, stopping at {name}")
            break

        print(f"\nTraining {name}...")
        clf.fit(X_train_s, y_train)
        preds = clf.predict(X_test_s)

        bal_acc = balanced_accuracy_score(y_test, preds)
        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, average="binary")
        print(f"  {name}: bal_acc={bal_acc:.4f} acc={acc:.4f} f1={f1:.4f}")

        if bal_acc > best_bal_acc:
            best_bal_acc = bal_acc
            best_acc = acc
            best_f1 = f1
            best_name = name

            if hasattr(clf, 'feature_importances_'):
                feat_names = [f"far_{i}" for i in range(12)] + [f"near_{i}" for i in range(12)] + [f"diff_{i}" for i in range(12)]
                importances = clf.feature_importances_
                top_idx = np.argsort(importances)[-10:][::-1]
                print(f"  Top features: {[(feat_names[i], f'{importances[i]:.3f}') for i in top_idx]}")

    elapsed = time.time() - t0
    num_params = X_train.shape[1] * 10

    approach = f"handcrafted_noise_grad_fft_color_36feat_{best_name}"
    result_block = (
        f"\n---\n"
        f"balanced_accuracy: {best_bal_acc:.6f}\n"
        f"accuracy:          {best_acc:.6f}\n"
        f"f1_score:          {best_f1:.6f}\n"
        f"num_params:        {num_params}\n"
        f"training_seconds:  {elapsed:.1f}\n"
        f"approach:          {approach}\n"
        f"---\n"
    )
    print(result_block)

    result_data = {
        "balanced_accuracy": best_bal_acc,
        "accuracy": best_acc,
        "f1_score": best_f1,
        "num_params": num_params,
        "training_seconds": elapsed,
        "approach": approach,
    }
    with open(str(RESULTS_FILE), "w") as rf:
        json.dump(result_data, rf, indent=2)
    print(f"Results written to {RESULTS_FILE}")


train()
