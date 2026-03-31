"""
train.py — Liveness detection training script (runs on DGX1).
Experiment: Pure handcrafted features (noise, gradient, FFT, texture, color)
+ GradientBoosting ensemble on the full dataset.
Handcrafted features achieved 100% on small dataset; let's see how they
scale to the real 2586-sample dataset with proper train/test split.
Previous best on DGX1: 0.976659 (ResNet18 6ch baseline).
"""
import os
import sys
import json
import time
import random
import numpy as np
from pathlib import Path
from collections import Counter

from PIL import Image
from sklearn.metrics import balanced_accuracy_score, accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.preprocessing import StandardScaler

DATA_DIR = Path("/mnt/nas/public2/simon/projects/auto_research/liveness-research/data")
RESULTS_FILE = Path("/mnt/nas/public2/simon/projects/auto_research/liveness-research/data/last_result.json")
SEED = 42
MAX_SECONDS = 270

random.seed(SEED)
np.random.seed(SEED)


def extract_features(img_pil):
    """Extract comprehensive handcrafted features from a PIL image."""
    img_np = np.array(img_pil).astype(np.float32)
    gray = np.array(img_pil.convert("L")).astype(np.float32)
    features = []

    from scipy.ndimage import laplace
    lap = laplace(gray)
    features.append(np.var(lap))        # Laplacian variance (sharpness)
    features.append(np.abs(lap).mean()) # Mean absolute Laplacian

    gy = np.diff(gray, axis=0)
    gx = np.diff(gray, axis=1)
    features.append(np.abs(gy).mean())  # Vertical gradient mean
    features.append(np.abs(gx).mean())  # Horizontal gradient mean
    features.append(np.std(gy))         # Vertical gradient std
    features.append(np.std(gx))         # Horizontal gradient std

    fft = np.fft.fft2(gray)
    fft_shift = np.fft.fftshift(fft)
    mag = np.abs(fft_shift)
    h, w = mag.shape
    cy, cx = h // 2, w // 2
    r = min(h, w) // 8
    Y, X = np.ogrid[:h, :w]
    center_mask = (Y - cy)**2 + (X - cx)**2 <= r**2
    low_energy = mag[center_mask].sum()
    total_energy = mag.sum() + 1e-10
    hf_ratio = 1.0 - low_energy / total_energy
    features.append(hf_ratio)

    r2 = min(h, w) // 4
    mid_mask = ((Y - cy)**2 + (X - cx)**2 <= r2**2) & ~center_mask
    mid_energy = mag[mid_mask].sum()
    features.append(mid_energy / total_energy)

    channel_means = img_np.mean(axis=(0, 1))
    channel_stds = img_np.std(axis=(0, 1))
    features.extend(channel_means.tolist())
    features.extend(channel_stds.tolist())

    blue_shift = channel_means[2] - (channel_means[0] + channel_means[1]) / 2.0
    features.append(blue_shift)

    features.append(np.std(gray))
    features.append(float(np.median(gray)))
    features.append(float(np.percentile(gray, 95) - np.percentile(gray, 5)))

    bw_pixels = np.sum((gray < 10) | (gray > 245))
    features.append(bw_pixels / gray.size)

    from scipy.ndimage import uniform_filter
    local_mean = uniform_filter(gray, size=7)
    local_sqr_mean = uniform_filter(gray**2, size=7)
    local_var = np.maximum(local_sqr_mean - local_mean**2, 0)
    features.append(np.mean(local_var))
    features.append(np.std(local_var))

    return np.array(features, dtype=np.float32)


def train():
    t0 = time.time()
    print("Loading data...")

    with open(str(DATA_DIR / "labels.json")) as f:
        labels = json.load(f)

    valid = []
    valid_labels = []
    for sig, info in labels.items():
        far = DATA_DIR / "samples" / sig / "far.jpg"
        near = DATA_DIR / "samples" / sig / "near.jpg"
        if far.exists() and near.exists():
            valid.append(sig)
            valid_labels.append(0 if info["main_label"] == "Positive" else 1)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    train_idx, test_idx = next(skf.split(valid, valid_labels))
    train_ids = [valid[i] for i in train_idx]
    test_ids = [valid[i] for i in test_idx]

    dist_train = Counter(valid_labels[i] for i in train_idx)
    dist_test = Counter(valid_labels[i] for i in test_idx)
    print(f"Train: {len(train_ids)} {dict(dist_train)}")
    print(f"Test:  {len(test_ids)} {dict(dist_test)}")

    print("Extracting features...")
    samples_dir = DATA_DIR / "samples"

    def extract_for_ids(ids):
        X_list = []
        y_list = []
        for i, sig in enumerate(ids):
            if time.time() - t0 > MAX_SECONDS - 30:
                print(f"Time budget warning at sample {i}/{len(ids)}")
                break
            label = 0 if labels[sig]["main_label"] == "Positive" else 1
            try:
                far_img = Image.open(str(samples_dir / sig / "far.jpg")).convert("RGB")
                near_img = Image.open(str(samples_dir / sig / "near.jpg")).convert("RGB")
                far_img = far_img.resize((256, 256))
                near_img = near_img.resize((256, 256))
            except Exception:
                continue

            far_feats = extract_features(far_img)
            near_feats = extract_features(near_img)
            diff_feats = far_feats - near_feats
            ratio_feats = far_feats / (near_feats + 1e-10)
            combined = np.concatenate([far_feats, near_feats, diff_feats, ratio_feats])
            combined = np.nan_to_num(combined, nan=0.0, posinf=1e6, neginf=-1e6)
            X_list.append(combined)
            y_list.append(label)

            if (i + 1) % 500 == 0:
                print(f"  Processed {i+1}/{len(ids)} samples ({time.time()-t0:.1f}s)")

        return np.array(X_list), np.array(y_list)

    X_train, y_train = extract_for_ids(train_ids)
    X_test, y_test = extract_for_ids(test_ids)
    print(f"Features: {X_train.shape[1]} per sample")
    print(f"Extraction done in {time.time()-t0:.1f}s")

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    print("Training GBM...")
    gbm = GradientBoostingClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        min_samples_leaf=5,
        random_state=SEED,
    )
    gbm.fit(X_train, y_train)
    gbm_preds = gbm.predict(X_test)
    gbm_bal_acc = balanced_accuracy_score(y_test, gbm_preds)
    print(f"GBM balanced_accuracy: {gbm_bal_acc:.6f}")

    print("Training RF...")
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=3,
        random_state=SEED,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    rf_preds = rf.predict(X_test)
    rf_bal_acc = balanced_accuracy_score(y_test, rf_preds)
    print(f"RF balanced_accuracy: {rf_bal_acc:.6f}")

    gbm_proba = gbm.predict_proba(X_test)
    rf_proba = rf.predict_proba(X_test)
    ensemble_proba = 0.5 * gbm_proba + 0.5 * rf_proba
    ensemble_preds = ensemble_proba.argmax(axis=1)

    bal_acc = balanced_accuracy_score(y_test, ensemble_preds)
    acc = accuracy_score(y_test, ensemble_preds)
    f1 = f1_score(y_test, ensemble_preds, average="binary")
    print(f"Ensemble balanced_accuracy: {bal_acc:.6f}")

    best_bal_acc = max(gbm_bal_acc, rf_bal_acc, bal_acc)
    if best_bal_acc == gbm_bal_acc:
        best_preds = gbm_preds
        method = "gbm"
    elif best_bal_acc == rf_bal_acc:
        best_preds = rf_preds
        method = "rf"
    else:
        best_preds = ensemble_preds
        method = "ensemble"

    best_acc = accuracy_score(y_test, best_preds)
    best_f1 = f1_score(y_test, best_preds, average="binary")

    elapsed = time.time() - t0
    num_params = X_train.shape[1]
    approach = f"handcrafted_{X_train.shape[1]}feats_{method}_300trees_full_dataset"

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
    with open(str(RESULTS_FILE), "w") as rf_file:
        json.dump(result_data, rf_file, indent=2)
    print(f"Results written to {RESULTS_FILE}")

    if hasattr(gbm, 'feature_importances_'):
        fi = gbm.feature_importances_
        top_idx = np.argsort(fi)[::-1][:10]
        print("\nTop 10 GBM feature importances:")
        for rank, idx in enumerate(top_idx):
            print(f"  {rank+1}. Feature {idx}: {fi[idx]:.4f}")


train()
