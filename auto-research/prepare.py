"""
prepare.py — Fixed evaluation harness for liveness auto-research.
DO NOT MODIFY. This file is part of the fixed infrastructure.

Provides:
- load_dataset(): returns train/test split with signatureIds and labels
- evaluate(): computes balanced_accuracy, accuracy, f1
- DATA_DIR, SAMPLES_DIR paths
"""
import json
import random
import numpy as np
from pathlib import Path
from collections import Counter
from sklearn.metrics import balanced_accuracy_score, accuracy_score, f1_score

DATA_DIR = Path("/mnt/nas/public2/simon/projects/auto_research/liveness-research/data")
SAMPLES_DIR = DATA_DIR / "samples"
SEED = 42


def load_dataset(test_ratio=0.2):
    """Load labeled dataset, return train/test split.
    
    Returns:
        train_ids: list of signatureIds for training
        test_ids: list of signatureIds for testing
        labels: dict {signatureId: {main_label, sublabel, property_name, ...}}
    """
    with open(str(DATA_DIR / "labels.json")) as f:
        labels = json.load(f)

    valid = []
    for sig in labels:
        far = SAMPLES_DIR / sig / "far.jpg"
        near = SAMPLES_DIR / sig / "near.jpg"
        if far.exists() and near.exists():
            valid.append(sig)

    random.seed(SEED)
    random.shuffle(valid)
    split = int(len(valid) * (1 - test_ratio))
    return valid[:split], valid[split:], labels


def evaluate(y_true, y_pred):
    """Compute standard metrics.
    
    Args:
        y_true: list of int (0=Positive, 1=Negative)
        y_pred: list of int (0=Positive, 1=Negative)
    
    Returns:
        dict with balanced_accuracy, accuracy, f1_score
    """
    return {
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_score": f1_score(y_true, y_pred, average="binary"),
    }


def print_metrics(metrics, num_params=0, training_seconds=0, approach=""):
    """Print metrics in the required output format."""
    print(f"\n---")
    print(f"balanced_accuracy: {metrics['balanced_accuracy']:.6f}")
    print(f"accuracy:          {metrics['accuracy']:.6f}")
    print(f"f1_score:          {metrics['f1_score']:.6f}")
    print(f"num_params:        {num_params}")
    print(f"training_seconds:  {training_seconds:.1f}")
    print(f"approach:          {approach}")
    print(f"---")


if __name__ == "__main__":
    train_ids, test_ids, labels = load_dataset()
    dist_train = Counter(labels[s]["main_label"] for s in train_ids)
    dist_test = Counter(labels[s]["main_label"] for s in test_ids)
    print(f"Dataset loaded:")
    print(f"  Train: {len(train_ids)} {dict(dist_train)}")
    print(f"  Test:  {len(test_ids)} {dict(dist_test)}")
    print(f"  Total: {len(train_ids) + len(test_ids)}")
    print(f"  Samples dir: {SAMPLES_DIR}")
