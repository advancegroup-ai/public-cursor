"""
train.py — Liveness detection training script (runs on DGX1).
Experiment: Handcrafted features (noise, gradient, FFT, color) + 
            RandomForest + GradientBoosting ensemble.
            These features achieved 100% on small dataset.
            Test on full annotations_full.jsonl dataset (3000/class).
            FAST: no GPU needed, feature extraction + RF takes seconds.
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
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier

DATA_DIR = Path("/mnt/nas/public2/simon/projects/auto_research/liveness-research/data")
RESULTS_FILE = Path("/mnt/nas/public2/simon/projects/auto_research/liveness-research/data/last_result.json")
SEED = 42
MAX_SECONDS = 250
MAX_PER_CLASS = 3000

random.seed(SEED)
np.random.seed(SEED)


def compute_features(img_path):
    """Compute comprehensive handcrafted features from an image."""
    try:
        img = Image.open(img_path).convert("RGB")
        arr = np.array(img, dtype=np.float32) / 255.0
    except Exception:
        return None

    gray = arr.mean(axis=2)

    # 1. Laplacian noise std
    lap_k = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
    from scipy.signal import convolve2d
    lap = convolve2d(gray, lap_k, mode='same', boundary='symm')
    noise_std = lap.std()

    # 2. Gradient features
    gx = np.diff(gray, axis=1)
    gy = np.diff(gray, axis=0)
    grad_mean = (np.abs(gx).mean() + np.abs(gy).mean()) / 2
    grad_std = (gx.std() + gy.std()) / 2

    # 3. FFT features
    fft = np.fft.fft2(gray)
    fft_mag = np.abs(np.fft.fftshift(fft))
    h, w = fft_mag.shape
    ch, cw = h // 2, w // 2
    qh, qw = h // 4, w // 4
    low_freq = fft_mag[ch-qh:ch+qh, cw-qw:cw+qw].mean()
    high_freq = fft_mag.mean()
    freq_ratio = high_freq / (low_freq + 1e-8)

    # 4. Color features
    r, g, b = arr[:,:,0], arr[:,:,1], arr[:,:,2]
    blue_shift = b.mean() - (r.mean() + g.mean()) / 2
    color_std = np.array([r.std(), g.std(), b.std()]).mean()

    # 5. Black/white fraction
    bw_frac = ((gray < 0.05) | (gray > 0.95)).mean()

    # 6. Channel uniformity (std of channel means)
    ch_means = np.array([r.mean(), g.mean(), b.mean()])
    ch_uniformity = ch_means.std()

    # 7. Edge density
    edge_threshold = 0.1
    edge_density = (np.abs(gx) > edge_threshold).mean()

    # 8. Contrast (IQR)
    q25, q75 = np.percentile(gray, [25, 75])
    contrast_iqr = q75 - q25

    return [noise_std, grad_mean, grad_std, freq_ratio, blue_shift,
            color_std, bw_frac, ch_uniformity, edge_density, contrast_iqr]


def load_data():
    t0 = time.time()
    pos_sigs, neg_sigs = [], []
    labels = {}
    with open(str(DATA_DIR / "annotations_full.jsonl")) as f:
        for line in f:
            entry = json.loads(line.strip())
            sig = entry.get("sig", "")
            lbl = entry.get("label", "")
            if sig and lbl in ("Positive", "Negative"):
                numeric = 0 if lbl == "Positive" else 1
                labels[sig] = numeric
                if numeric == 0:
                    pos_sigs.append(sig)
                else:
                    neg_sigs.append(sig)
    print(f"Loaded annotations in {time.time()-t0:.1f}s: pos={len(pos_sigs)}, neg={len(neg_sigs)}")

    random.shuffle(pos_sigs)
    random.shuffle(neg_sigs)
    pos_sigs = pos_sigs[:MAX_PER_CLASS]
    neg_sigs = neg_sigs[:MAX_PER_CLASS]

    all_valid = pos_sigs + neg_sigs
    random.shuffle(all_valid)

    split = int(len(all_valid) * 0.8)
    train_ids = all_valid[:split]
    test_ids = all_valid[split:]

    dist_train = Counter(labels[s] for s in train_ids)
    dist_test = Counter(labels[s] for s in test_ids)
    print(f"Train: {len(train_ids)} (0={dist_train[0]}, 1={dist_train[1]})")
    print(f"Test:  {len(test_ids)} (0={dist_test[0]}, 1={dist_test[1]})")
    return train_ids, test_ids, labels


def extract_features(sig_ids, labels):
    """Extract features for all samples."""
    samples_dir = str(DATA_DIR / "samples")
    X, y = [], []
    skipped = 0
    for i, sig in enumerate(sig_ids):
        if i % 500 == 0:
            print(f"  Extracting features: {i}/{len(sig_ids)}...")
        far_path = os.path.join(samples_dir, sig, "far.jpg")
        near_path = os.path.join(samples_dir, sig, "near.jpg")

        far_feats = compute_features(far_path)
        near_feats = compute_features(near_path)

        if far_feats is None or near_feats is None:
            skipped += 1
            continue

        combined = far_feats + near_feats  # 20 features total
        X.append(combined)
        y.append(labels[sig])

    print(f"  Extracted {len(X)} samples, skipped {skipped}")
    return np.array(X), np.array(y)


def train():
    t0 = time.time()
    print("Device: cpu (handcrafted features + RandomForest)")

    train_ids, test_ids, labels = load_data()
    if len(train_ids) == 0:
        print("ERROR: No valid training data!")
        return

    print("Extracting train features...")
    X_train, y_train = extract_features(train_ids, labels)
    print(f"Train features shape: {X_train.shape}")

    if time.time() - t0 > MAX_SECONDS:
        print("Time budget reached during train feature extraction")
        return

    print("Extracting test features...")
    X_test, y_test = extract_features(test_ids, labels)
    print(f"Test features shape: {X_test.shape}")

    if time.time() - t0 > MAX_SECONDS:
        print("Time budget reached during test feature extraction")
        return

    # Handle NaN/Inf
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=1e6, neginf=-1e6)
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=1e6, neginf=-1e6)

    print("Training RF + GB ensemble...")
    rf = RandomForestClassifier(n_estimators=200, max_depth=None, random_state=SEED, n_jobs=-1)
    gb = GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=SEED)
    
    rf.fit(X_train, y_train)
    gb.fit(X_train, y_train)

    # Ensemble via soft voting
    rf_probs = rf.predict_proba(X_test)
    gb_probs = gb.predict_proba(X_test)
    ensemble_probs = 0.6 * rf_probs + 0.4 * gb_probs
    preds = ensemble_probs.argmax(axis=1)

    bal_acc = balanced_accuracy_score(y_test, preds)
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average="binary")

    # Also check individual models
    rf_preds = rf.predict(X_test)
    gb_preds = gb.predict(X_test)
    rf_bal = balanced_accuracy_score(y_test, rf_preds)
    gb_bal = balanced_accuracy_score(y_test, gb_preds)
    print(f"RF alone: bal_acc={rf_bal:.4f}")
    print(f"GB alone: bal_acc={gb_bal:.4f}")
    print(f"Ensemble: bal_acc={bal_acc:.4f}")

    # Feature importances
    feat_names = [f"far_{n}" for n in ["noise","grad_mean","grad_std","freq_ratio","blue_shift",
                                       "color_std","bw_frac","ch_uniform","edge_dens","contrast"]] + \
                 [f"near_{n}" for n in ["noise","grad_mean","grad_std","freq_ratio","blue_shift",
                                        "color_std","bw_frac","ch_uniform","edge_dens","contrast"]]
    importances = rf.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]
    print("\nTop 10 features:")
    for i in sorted_idx[:10]:
        print(f"  {feat_names[i]}: {importances[i]:.4f}")

    elapsed = time.time() - t0
    num_params = rf.n_estimators * 100 + gb.n_estimators * 100  # rough estimate
    approach = "handcrafted_20feats_RF200_GB100_ensemble_3000perclass"

    result_block = (
        f"\n---\n"
        f"balanced_accuracy: {bal_acc:.6f}\n"
        f"accuracy:          {acc:.6f}\n"
        f"f1_score:          {f1:.6f}\n"
        f"num_params:        {num_params}\n"
        f"training_seconds:  {elapsed:.1f}\n"
        f"approach:          {approach}\n"
        f"---\n"
    )
    print(result_block)

    result_data = {
        "balanced_accuracy": float(bal_acc),
        "accuracy": float(acc),
        "f1_score": float(f1),
        "num_params": num_params,
        "training_seconds": elapsed,
        "approach": approach,
    }
    with open(str(RESULTS_FILE), "w") as rf2:
        json.dump(result_data, rf2, indent=2)
    print(f"Results written to {RESULTS_FILE}")


train()
