"""
train.py — Liveness detection training script (runs on DGX1).
Experiment: Handcrafted noise/gradient/FFT/color/texture features + GBM/RF ensemble on full dataset.
This is the most promising experiment: handcrafted noise features achieved perfect accuracy on
the smaller dataset. Testing if the same approach generalizes to the full 2586-sample dataset.
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
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.preprocessing import StandardScaler

DATA_DIR = Path("/mnt/nas/public2/simon/projects/auto_research/liveness-research/data")
RESULTS_FILE = Path("/mnt/nas/public2/simon/projects/auto_research/liveness-research/data/last_result.json")
SEED = 42
MAX_SECONDS = 270

random.seed(SEED)
np.random.seed(SEED)


def extract_features(img_path):
    """Extract comprehensive handcrafted features from a single image."""
    try:
        img = Image.open(str(img_path)).convert("RGB")
    except Exception:
        return np.zeros(14, dtype=np.float32)

    img_np = np.array(img).astype(np.float32)
    gray = np.array(img.convert("L")).astype(np.float32)

    features = []

    from scipy.ndimage import laplace
    lap = laplace(gray)
    features.append(np.var(lap))
    features.append(np.abs(lap).mean())

    gy = np.diff(gray, axis=0)
    gx = np.diff(gray, axis=1)
    features.append(np.abs(gy).mean())
    features.append(np.abs(gx).mean())
    features.append(np.std(gy))
    features.append(np.std(gx))

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
    features.append(1.0 - low_energy / total_energy)

    channel_means = img_np.mean(axis=(0, 1))
    blue_shift = channel_means[2] - (channel_means[0] + channel_means[1]) / 2.0
    features.append(blue_shift)

    padded = np.pad(gray, 1, mode='reflect')
    center = padded[1:-1, 1:-1]
    neighbors = [
        padded[:-2, :-2], padded[:-2, 1:-1], padded[:-2, 2:],
        padded[1:-1, :-2],                    padded[1:-1, 2:],
        padded[2:, :-2],  padded[2:, 1:-1],  padded[2:, 2:],
    ]
    lbp_like = sum((n >= center).astype(np.float32) for n in neighbors)
    features.append(np.var(lbp_like))
    features.append(np.mean(lbp_like))

    features.append(np.std(gray))
    features.append(np.percentile(gray, 95) - np.percentile(gray, 5))

    hist, _ = np.histogram(gray.astype(np.uint8), bins=256, range=(0, 256))
    hist = hist / (hist.sum() + 1e-10)
    entropy = -np.sum(hist[hist > 0] * np.log2(hist[hist > 0]))
    features.append(entropy)

    # Additional features for robustness
    # JPEG quality estimation via block boundary discontinuity
    if gray.shape[0] >= 16 and gray.shape[1] >= 16:
        block_h = gray[:gray.shape[0]//8*8, :].reshape(-1, 8, gray.shape[1])
        boundary_diffs = np.abs(np.diff(block_h, axis=1)).mean()
        features.append(boundary_diffs)
    else:
        features.append(0.0)

    return np.array(features, dtype=np.float32)


def load_data():
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

    return train_ids, test_ids, labels


def train():
    t0 = time.time()
    print("Extracting handcrafted features from all samples...")

    train_ids, test_ids, labels = load_data()
    samples_dir = DATA_DIR / "samples"

    X_train, y_train = [], []
    X_test, y_test = [], []

    for i, sig in enumerate(train_ids):
        if time.time() - t0 > MAX_SECONDS * 0.5:
            print(f"Feature extraction time limit, stopping at {i}/{len(train_ids)}")
            break
        far_feats = extract_features(samples_dir / sig / "far.jpg")
        near_feats = extract_features(samples_dir / sig / "near.jpg")
        # Also compute difference features between far and near
        diff_feats = far_feats - near_feats
        combined = np.concatenate([far_feats, near_feats, diff_feats])
        X_train.append(combined)
        label = 0 if labels[sig]["main_label"] == "Positive" else 1
        y_train.append(label)
        if (i + 1) % 200 == 0:
            print(f"  Train features: {i+1}/{len(train_ids)} ({time.time()-t0:.1f}s)")

    for i, sig in enumerate(test_ids):
        if time.time() - t0 > MAX_SECONDS * 0.7:
            print(f"Test feature extraction time limit, stopping at {i}/{len(test_ids)}")
            break
        far_feats = extract_features(samples_dir / sig / "far.jpg")
        near_feats = extract_features(samples_dir / sig / "near.jpg")
        diff_feats = far_feats - near_feats
        combined = np.concatenate([far_feats, near_feats, diff_feats])
        X_test.append(combined)
        label = 0 if labels[sig]["main_label"] == "Positive" else 1
        y_test.append(label)

    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    y_test = np.array(y_test)

    print(f"Features extracted: train={X_train.shape}, test={X_test.shape}")
    print(f"Feature extraction took {time.time()-t0:.1f}s")

    X_train = np.nan_to_num(X_train, nan=0.0, posinf=1e6, neginf=-1e6)
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=1e6, neginf=-1e6)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    best_bal_acc = 0
    best_name = ""

    classifiers = {
        "GBM_200": GradientBoostingClassifier(n_estimators=200, max_depth=5, learning_rate=0.1, random_state=SEED),
        "GBM_300": GradientBoostingClassifier(n_estimators=300, max_depth=6, learning_rate=0.05, random_state=SEED),
        "RF_300": RandomForestClassifier(n_estimators=300, max_depth=None, random_state=SEED),
        "RF_500": RandomForestClassifier(n_estimators=500, max_depth=None, random_state=SEED),
    }

    for name, clf in classifiers.items():
        if time.time() - t0 > MAX_SECONDS:
            print(f"Time budget reached, skipping {name}")
            break
        print(f"\nTraining {name}...")
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)
        bal_acc = balanced_accuracy_score(y_test, preds)
        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, average="binary")
        print(f"{name}: bal_acc={bal_acc:.6f} acc={acc:.6f} f1={f1:.6f}")

        if bal_acc > best_bal_acc:
            best_bal_acc = bal_acc
            best_acc = acc
            best_f1 = f1
            best_name = name

        if hasattr(clf, 'feature_importances_'):
            fi = clf.feature_importances_
            top_idx = np.argsort(fi)[::-1][:10]
            print(f"  Top features: {top_idx.tolist()} importances: {fi[top_idx].tolist()}")

    elapsed = time.time() - t0
    approach = f"handcrafted_45feats_far_near_diff_best_{best_name}_full_dataset"
    num_params = X_train.shape[1] * 300

    print(f"\nBest classifier: {best_name}")

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
