"""
train.py — Liveness detection training script (runs on DGX1).
Experiment: Pure handcrafted features (Laplacian noise, gradient, FFT, color)
+ GradientBoosting ensemble on full 2586-sample dataset with 5-fold CV.

Rationale: Handcrafted noise/gradient features achieved 1.0 balanced accuracy
on the ~300 sample subset. This tests if they generalize to the full dataset.
The previous best on DGX1 was 0.976659 with ResNet18.
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
RESULTS_FILE = DATA_DIR / "last_result.json"
SEED = 42
MAX_SECONDS = 270

random.seed(SEED)
np.random.seed(SEED)


def load_labels():
    labels = {}
    for fname in ["neg_batch.json", "pos_batch.json"]:
        with open(str(DATA_DIR / fname)) as f:
            batch = json.load(f)
        for d in batch.get("data", []):
            parts = d["sample_id"].split("_")
            if len(parts) >= 4:
                sig = parts[3]
                main_label = d["pn"].split("/")[0]
                if main_label in ("Negative Type", "Negative_Type"):
                    main_label = "Negative"
                labels[sig] = {"main_label": main_label, "pn": d["pn"]}
    return labels


def extract_features(img_path):
    """Extract noise, gradient, FFT, and color features from an image."""
    import cv2
    try:
        img = cv2.imread(str(img_path))
        if img is None:
            return None
        img = cv2.resize(img, (224, 224))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    except Exception:
        return None

    features = []

    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    features.append(laplacian.std())
    features.append(np.abs(laplacian).mean())
    features.append(np.percentile(np.abs(laplacian), 90))

    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.sqrt(gx**2 + gy**2)
    features.append(grad_mag.mean())
    features.append(grad_mag.std())
    features.append(np.percentile(grad_mag, 90))

    gray_f = gray.astype(np.float64)
    fft = np.fft.fft2(gray_f)
    fft_shift = np.fft.fftshift(fft)
    magnitude = np.log1p(np.abs(fft_shift))
    h, w = magnitude.shape
    cy, cx = h // 2, w // 2

    for r_frac in [1/8, 1/4, 1/2]:
        r = int(min(h, w) * r_frac)
        low_freq = magnitude[cy-r:cy+r, cx-r:cx+r].mean()
        features.append(low_freq)

    features.append(magnitude.mean())
    high_freq = magnitude.copy()
    r = min(h, w) // 4
    high_freq[cy-r:cy+r, cx-r:cx+r] = 0
    features.append(high_freq[high_freq > 0].mean() if (high_freq > 0).any() else 0)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    features.append(hsv[:,:,1].mean())
    features.append(hsv[:,:,1].std())
    features.append(hsv[:,:,2].std())

    b, g, r_ch = img[:,:,0].astype(float), img[:,:,1].astype(float), img[:,:,2].astype(float)
    features.append(np.abs(b - g).mean())
    features.append(np.abs(b - r_ch).mean())
    features.append(np.abs(g - r_ch).mean())

    bw_frac = np.mean((gray < 10) | (gray > 245))
    features.append(bw_frac)
    features.append(gray.std())
    features.append(gray.mean())

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    features.append(lab[:,:,1].std())
    features.append(lab[:,:,2].std())

    return np.array(features, dtype=np.float64)


def train():
    t0 = time.time()
    print("Loading labels...")
    labels = load_labels()

    samples_dir = DATA_DIR / "samples"
    all_sigs = set(os.listdir(str(samples_dir)))

    valid_sigs = [s for s in labels if s in all_sigs
                  and (samples_dir / s / "far.jpg").exists()
                  and (samples_dir / s / "near.jpg").exists()]

    binary_labels_list = [0 if labels[s]["main_label"] == "Positive" else 1 for s in valid_sigs]
    print(f"Total valid samples: {len(valid_sigs)}")
    print(f"Distribution: {Counter(binary_labels_list)}")

    print("Extracting features...")
    X_all = []
    y_all = []
    valid_sigs_filtered = []

    for i, sig in enumerate(valid_sigs):
        if time.time() - t0 > MAX_SECONDS - 90:
            print(f"Feature extraction time budget at sample {i}/{len(valid_sigs)}")
            break

        far_path = samples_dir / sig / "far.jpg"
        near_path = samples_dir / sig / "near.jpg"

        far_feats = extract_features(far_path)
        near_feats = extract_features(near_path)

        if far_feats is None or near_feats is None:
            continue

        diff_feats = far_feats - near_feats
        combined = np.concatenate([far_feats, near_feats, diff_feats])

        X_all.append(combined)
        y_all.append(0 if labels[sig]["main_label"] == "Positive" else 1)
        valid_sigs_filtered.append(sig)

        if (i + 1) % 500 == 0:
            elapsed = time.time() - t0
            print(f"  Processed {i+1}/{len(valid_sigs)} samples ({elapsed:.1f}s)")

    X_all = np.array(X_all)
    y_all = np.array(y_all)
    print(f"Features extracted: {X_all.shape} in {time.time()-t0:.1f}s")

    from sklearn.model_selection import StratifiedKFold
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import balanced_accuracy_score, accuracy_score, f1_score

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    fold_results = []

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X_all, y_all)):
        if time.time() - t0 > MAX_SECONDS - 30:
            print(f"Training time budget at fold {fold_idx}")
            break

        X_train, X_test = X_all[train_idx], X_all[test_idx]
        y_train, y_test = y_all[train_idx], y_all[test_idx]

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        rf = RandomForestClassifier(
            n_estimators=300, max_depth=None, class_weight='balanced',
            random_state=SEED, n_jobs=-1
        )
        gb = GradientBoostingClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.05,
            subsample=0.8, random_state=SEED
        )

        ensemble = VotingClassifier(
            estimators=[('rf', rf), ('gb', gb)],
            voting='soft'
        )
        ensemble.fit(X_train_s, y_train)
        preds = ensemble.predict(X_test_s)

        bal_acc = balanced_accuracy_score(y_test, preds)
        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, average="binary")

        fold_results.append((bal_acc, acc, f1))
        print(f"Fold {fold_idx+1}/5: bal_acc={bal_acc:.4f} acc={acc:.4f} f1={f1:.4f}")

    mean_bal_acc = np.mean([r[0] for r in fold_results])
    mean_acc = np.mean([r[1] for r in fold_results])
    mean_f1 = np.mean([r[2] for r in fold_results])
    std_bal_acc = np.std([r[0] for r in fold_results])
    print(f"\nCV Mean: bal_acc={mean_bal_acc:.4f} +/- {std_bal_acc:.4f}")

    best_bal_acc = fold_results[0][0]
    best_acc = fold_results[0][1]
    best_f1 = fold_results[0][2]

    elapsed = time.time() - t0
    num_params = X_all.shape[1] * 500

    approach = "handcrafted_noise_grad_fft_color_diff_feats_RF300_GB200_ensemble_5fold"
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
