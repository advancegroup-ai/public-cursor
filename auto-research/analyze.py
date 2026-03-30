"""
Liveness detection analysis — the file the agent modifies.
Experiment 22: Haar wavelet high-frequency energy + DecisionTree.

Alternative to Laplacian: high-frequency energy via simple Haar decomposition.

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

def _haar_energy(img):
    """High-frequency energy from simple Haar-like decomposition."""
    gray = img.mean(axis=2)
    h, w = gray.shape
    h2, w2 = h // 2, w // 2
    even_rows = gray[:h2 * 2:2, :]
    odd_rows = gray[1:h2 * 2:2, :]
    detail_h = (even_rows - odd_rows)[:, :w2 * 2]
    even_cols = gray[:, :w2 * 2:2]
    odd_cols = gray[:, 1:w2 * 2:2]
    detail_v = (even_cols - odd_cols)[:h2 * 2, :]
    energy = float(np.mean(detail_h ** 2) + np.mean(detail_v ** 2))
    return [energy]


def extract_features(sample: dict) -> np.ndarray:
    """Extract wavelet features from far and near images."""
    features = []
    far = load_image(sample["far"])
    near = load_image(sample["near"])

    for img in [far, near]:
        features.extend(_haar_energy(img))

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
