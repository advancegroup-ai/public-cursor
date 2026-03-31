"""
train.py — Liveness detection: Fast handcrafted features + RF baseline on full DGX1 dataset
"""
import os, sys, json, time, random
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
    """Extract noise and frequency features from a single image."""
    from PIL import Image
    img = Image.open(str(img_path)).convert("RGB")
    arr = np.array(img, dtype=np.float32)
    
    gray = 0.299 * arr[:,:,0] + 0.587 * arr[:,:,1] + 0.114 * arr[:,:,2]
    
    # Laplacian noise (high-frequency energy)
    lap = np.abs(gray[2:,1:-1] + gray[:-2,1:-1] + gray[1:-1,2:] + gray[1:-1,:-2] - 4*gray[1:-1,1:-1])
    noise_mean = float(np.mean(lap))
    noise_std = float(np.std(lap))
    
    # Gradient features
    gx = np.abs(gray[:,1:] - gray[:,:-1])
    gy = np.abs(gray[1:,:] - gray[:-1,:])
    grad_mean = float(np.mean(gx) + np.mean(gy)) / 2
    grad_std = float(np.std(gx) + np.std(gy)) / 2
    
    # FFT high-frequency ratio
    fft = np.fft.fft2(gray)
    fft_shift = np.fft.fftshift(fft)
    h, w = gray.shape
    cy, cx = h // 2, w // 2
    r = min(h, w) // 8
    mask = np.ones((h, w), dtype=bool)
    y, x = np.ogrid[:h, :w]
    mask[(y - cy)**2 + (x - cx)**2 <= r**2] = False
    mag = np.abs(fft_shift)
    total_energy = float(np.sum(mag))
    hf_energy = float(np.sum(mag[mask]))
    hf_ratio = hf_energy / (total_energy + 1e-8)
    
    # Color statistics
    r_mean = float(np.mean(arr[:,:,0]))
    g_mean = float(np.mean(arr[:,:,1]))
    b_mean = float(np.mean(arr[:,:,2]))
    color_var = float(np.var([r_mean, g_mean, b_mean]))
    
    # Saturation
    maxc = np.max(arr, axis=2)
    minc = np.min(arr, axis=2)
    sat = (maxc - minc) / (maxc + 1e-8)
    sat_mean = float(np.mean(sat))
    
    return [noise_mean, noise_std, grad_mean, grad_std, hf_ratio, color_var, sat_mean]


def load_and_extract():
    with open(str(DATA_DIR / "labels.json")) as f:
        labels = json.load(f)

    valid = []
    for sig, info in labels.items():
        far = DATA_DIR / "samples" / sig / "far.jpg"
        near = DATA_DIR / "samples" / sig / "near.jpg"
        if far.exists() and near.exists():
            valid.append(sig)

    random.shuffle(valid)
    
    print(f"Total valid samples: {len(valid)}")
    
    features = []
    targets = []
    for i, sig in enumerate(valid):
        if i % 200 == 0:
            print(f"  Extracting features: {i}/{len(valid)}")
        
        info = labels[sig]
        label = 0 if info["main_label"] == "Positive" else 1
        
        far_path = DATA_DIR / "samples" / sig / "far.jpg"
        near_path = DATA_DIR / "samples" / sig / "near.jpg"
        
        try:
            far_feats = extract_features(far_path)
            near_feats = extract_features(near_path)
            feat_vec = far_feats + near_feats
            
            # Cross-image features
            feat_vec.append(far_feats[0] - near_feats[0])  # noise diff
            feat_vec.append(far_feats[2] - near_feats[2])  # grad diff
            feat_vec.append(far_feats[4] - near_feats[4])  # hf_ratio diff
            
            features.append(feat_vec)
            targets.append(label)
        except Exception as e:
            print(f"  Skipping {sig}: {e}")
            continue
    
    return np.array(features), np.array(targets), labels


def train():
    t0 = time.time()
    
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.metrics import balanced_accuracy_score, accuracy_score, f1_score
    from sklearn.model_selection import StratifiedKFold
    
    X, y, labels = load_and_extract()
    print(f"Features shape: {X.shape}, Labels dist: {Counter(y.tolist())}")
    
    # 5-fold stratified CV for robust evaluation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    
    all_bal_accs = []
    all_accs = []
    all_f1s = []
    
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        clf = RandomForestClassifier(n_estimators=200, max_depth=None, min_samples_leaf=2,
                                     class_weight='balanced', random_state=SEED, n_jobs=-1)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        
        bal_acc = balanced_accuracy_score(y_test, y_pred)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='binary')
        
        all_bal_accs.append(bal_acc)
        all_accs.append(acc)
        all_f1s.append(f1)
        
        print(f"  Fold {fold+1}: bal_acc={bal_acc:.4f} acc={acc:.4f} f1={f1:.4f}")
    
    mean_bal_acc = np.mean(all_bal_accs)
    mean_acc = np.mean(all_accs)
    mean_f1 = np.mean(all_f1s)
    
    elapsed = time.time() - t0
    approach = "handcrafted_17feat_noise_grad_fft_color_sat_crossimg_RF200_5foldCV"
    
    result_block = (
        f"\n---\n"
        f"balanced_accuracy: {mean_bal_acc:.6f}\n"
        f"accuracy:          {mean_acc:.6f}\n"
        f"f1_score:          {mean_f1:.6f}\n"
        f"num_params:        0\n"
        f"training_seconds:  {elapsed:.1f}\n"
        f"approach:          {approach}\n"
        f"---\n"
    )
    print(result_block)

    result_data = {
        "balanced_accuracy": mean_bal_acc,
        "accuracy": mean_acc,
        "f1_score": mean_f1,
        "num_params": 0,
        "training_seconds": elapsed,
        "approach": approach,
    }
    with open(str(RESULTS_FILE), "w") as rf:
        json.dump(result_data, rf, indent=2)
    print(f"Results written to {RESULTS_FILE}")


train()
