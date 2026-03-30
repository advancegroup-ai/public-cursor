"""
Fixed infrastructure for auto-research liveness detection.
Data loading, evaluation, and constants. DO NOT MODIFY.

Usage:
    python prepare.py                   # download sample dataset
    python prepare.py --stats           # show dataset statistics
"""

import os
import json
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
from sklearn.metrics import (
    balanced_accuracy_score, accuracy_score, precision_score,
    recall_score, f1_score, classification_report, confusion_matrix,
)
from sklearn.model_selection import StratifiedKFold

# ---------------------------------------------------------------------------
# Constants (fixed, do not modify)
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).parent / "data"
SAMPLES_DIR = DATA_DIR / "samples"
LABELS_FILE = DATA_DIR / "labels.json"
RESULTS_FILE = Path(__file__).parent / "results.tsv"

IMAGE_SIZE = (224, 224)
RANDOM_SEED = 42
N_FOLDS = 5
TEST_FOLD = 0

BINARY_LABELS = {"Positive": 0, "Negative": 1, "Negative_Type": 1}

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_labels() -> Dict[str, dict]:
    """Load annotation labels. Returns {signature_id: {label, sublabel, ...}}"""
    with open(LABELS_FILE) as f:
        return json.load(f)


def load_image(path: str, size: Optional[Tuple[int, int]] = IMAGE_SIZE) -> np.ndarray:
    """Load image as RGB numpy array, resized to `size`."""
    # Support both .npy (synthetic) and .jpg (real) formats
    npy_path = path.replace('.jpg', '.npy') if path.endswith('.jpg') else path
    if os.path.exists(npy_path) and npy_path.endswith('.npy'):
        img = np.load(npy_path).astype(np.float32) / 255.0
        if size and (img.shape[0] != size[0] or img.shape[1] != size[1]):
            # Simple nearest-neighbor resize for npy
            from scipy.ndimage import zoom
            fy, fx = size[0] / img.shape[0], size[1] / img.shape[1]
            img = zoom(img, (fy, fx, 1), order=1)
        return img
    elif os.path.exists(path):
        try:
            from PIL import Image as PILImage
            pil_img = PILImage.open(path).convert("RGB")
            if size:
                pil_img = pil_img.resize(size, PILImage.LANCZOS)
            return np.array(pil_img, dtype=np.float32) / 255.0
        except ImportError:
            raise RuntimeError(f"Need Pillow to load {path}. pip install Pillow")
    else:
        raise FileNotFoundError(f"Image not found: {path} (or {npy_path})")


def load_dataset() -> Tuple[List[dict], List[int]]:
    """
    Load the full dataset.
    Returns (samples, labels) where:
    - samples: list of dicts with keys 'id', 'far', 'near', 'card' (image paths)
    - labels: list of binary labels (0=Positive/live, 1=Negative/attack)
    """
    label_data = load_labels()
    samples = []
    labels = []

    for sig_id, info in sorted(label_data.items()):
        sample_dir = SAMPLES_DIR / sig_id
        # Support both .npy (synthetic) and .jpg (real) formats
        far_path = sample_dir / "far.npy" if (sample_dir / "far.npy").exists() else sample_dir / "far.jpg"
        near_path = sample_dir / "near.npy" if (sample_dir / "near.npy").exists() else sample_dir / "near.jpg"
        card_path = sample_dir / "card.npy" if (sample_dir / "card.npy").exists() else sample_dir / "card.jpg"

        if not far_path.exists() or not near_path.exists():
            continue

        main_label = info["label"].split("/")[0]
        if main_label not in BINARY_LABELS:
            continue

        samples.append({
            "id": sig_id,
            "far": str(far_path),
            "near": str(near_path),
            "card": str(card_path) if card_path.exists() else None,
            "sublabel": info.get("sublabel", ""),
            "region": info.get("region", ""),
        })
        labels.append(BINARY_LABELS[main_label])

    return samples, labels


def get_train_test_split(
    samples: List[dict], labels: List[int]
) -> Tuple[List[dict], List[int], List[dict], List[int]]:
    """
    Deterministic stratified train/test split using fold 0 as test.
    Returns (train_samples, train_labels, test_samples, test_labels).
    """
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    labels_arr = np.array(labels)

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(samples, labels_arr)):
        if fold_idx == TEST_FOLD:
            train_samples = [samples[i] for i in train_idx]
            train_labels = [labels[i] for i in train_idx]
            test_samples = [samples[i] for i in test_idx]
            test_labels = [labels[i] for i in test_idx]
            return train_samples, train_labels, test_samples, test_labels

    raise RuntimeError("Should not reach here")


# ---------------------------------------------------------------------------
# Evaluation (DO NOT CHANGE — this is the fixed metric)
# ---------------------------------------------------------------------------

def evaluate(y_true: List[int], y_pred: List[int]) -> Dict[str, float]:
    """
    Evaluate predictions against ground truth.
    Primary metric: balanced_accuracy (handles class imbalance).
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    bal_acc = balanced_accuracy_score(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    return {
        "balanced_accuracy": bal_acc,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
    }


def print_results(metrics: Dict[str, float], num_features: int, num_samples: int):
    """Print results in the standard format for grep extraction."""
    print("---")
    print(f"balanced_accuracy: {metrics['balanced_accuracy']:.6f}")
    print(f"accuracy:          {metrics['accuracy']:.6f}")
    print(f"precision:         {metrics['precision']:.6f}")
    print(f"recall:            {metrics['recall']:.6f}")
    print(f"f1_score:          {metrics['f1_score']:.6f}")
    print(f"num_features:      {num_features}")
    print(f"num_samples:       {num_samples}")


def print_detailed_report(y_true, y_pred):
    """Print detailed classification report and confusion matrix."""
    print("\nClassification Report:")
    print(classification_report(
        y_true, y_pred,
        target_names=["Positive (live)", "Negative (attack)"],
        zero_division=0,
    ))
    print("Confusion Matrix:")
    cm = confusion_matrix(y_true, y_pred)
    print(f"  Predicted:   Live   Attack")
    print(f"  Actual Live: {cm[0][0]:5d}  {cm[0][1]:5d}")
    print(f"  Actual Atk:  {cm[1][0]:5d}  {cm[1][1]:5d}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--stats", action="store_true", help="Show dataset statistics")
    args = parser.parse_args()

    if args.stats:
        samples, labels = load_dataset()
        labels_arr = np.array(labels)
        n_pos = (labels_arr == 0).sum()
        n_neg = (labels_arr == 1).sum()
        print(f"Dataset: {len(samples)} samples")
        print(f"  Positive (live):   {n_pos} ({100*n_pos/len(samples):.1f}%)")
        print(f"  Negative (attack): {n_neg} ({100*n_neg/len(samples):.1f}%)")

        label_data = load_labels()
        sublabel_counts = {}
        for info in label_data.values():
            sl = info.get("sublabel", "unknown")
            sublabel_counts[sl] = sublabel_counts.get(sl, 0) + 1
        print("\nSub-label distribution:")
        for sl, cnt in sorted(sublabel_counts.items(), key=lambda x: -x[1]):
            print(f"  {sl:30s}: {cnt}")

        train_s, train_l, test_s, test_l = get_train_test_split(samples, labels)
        print(f"\nTrain/Test split: {len(train_s)} / {len(test_s)}")
        print(f"  Train pos/neg: {sum(1 for l in train_l if l==0)} / {sum(1 for l in train_l if l==1)}")
        print(f"  Test  pos/neg: {sum(1 for l in test_l if l==0)} / {sum(1 for l in test_l if l==1)}")
    else:
        print("Run with --stats to see dataset information.")
        print("This file provides utilities imported by analyze.py.")
