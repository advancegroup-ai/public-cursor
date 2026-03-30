"""
Liveness detection analysis — the file the agent modifies.
Experiment 15: Gradient mean only (2 features) + DecisionTree.

Simplify gradient approach to single feature per image.

Usage: python analyze.py
"""

import numpy as np
from sklearn.tree import DecisionTreeClassifier
from prepare import (
    load_dataset, load_image, get_train_test_split,
    evaluate, print_results, print_detailed_report,
)

# ---------------------------------------------------------------------------
# Feature extraction (MODIFY THIS)
# ---------------------------------------------------------------------------

def _gradient_features(img):
    """Single gradient feature: mean magnitude."""
    gray = img.mean(axis=2)
    gx = gray[:, 1:] - gray[:, :-1]
    gy = gray[1:, :] - gray[:-1, :]
    h, w = min(gx.shape[0], gy.shape[0]), min(gx.shape[1], gy.shape[1])
    mag = np.sqrt(gx[:h, :w] ** 2 + gy[:h, :w] ** 2)
    return [float(mag.mean())]


def extract_features(sample: dict) -> np.ndarray:
    """Extract gradient features from far and near images."""
    features = []
    far = load_image(sample["far"])
    near = load_image(sample["near"])

    for img in [far, near]:
        features.extend(_gradient_features(img))

    return np.array(features, dtype=np.float32)


# ---------------------------------------------------------------------------
# Classification (MODIFY THIS)
# ---------------------------------------------------------------------------

def build_classifier():
    """Build and return a classifier."""
    return DecisionTreeClassifier(
        max_depth=None,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
    )


# ---------------------------------------------------------------------------
# Main pipeline (MODIFY THIS)
# ---------------------------------------------------------------------------

def main():
    print("Loading dataset...")
    samples, labels = load_dataset()
    print(f"Loaded {len(samples)} samples")

    print("Splitting train/test...")
    train_samples, train_labels, test_samples, test_labels = get_train_test_split(
        samples, labels
    )
    print(f"Train: {len(train_samples)}, Test: {len(test_samples)}")

    print("Extracting features...")
    X_train = np.array([extract_features(s) for s in train_samples])
    X_test = np.array([extract_features(s) for s in test_samples])

    X_train = np.nan_to_num(X_train, nan=0.0, posinf=1.0, neginf=-1.0)
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=1.0, neginf=-1.0)

    num_features = X_train.shape[1]
    print(f"Features: {num_features}")

    print("Training classifier...")
    clf = build_classifier()
    clf.fit(X_train, train_labels)

    print("Evaluating...")
    y_pred = clf.predict(X_test)

    metrics = evaluate(test_labels, y_pred)
    print_results(metrics, num_features, len(samples))
    print_detailed_report(test_labels, y_pred)


if __name__ == "__main__":
    main()
