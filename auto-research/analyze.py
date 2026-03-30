"""
Liveness detection analysis — the file the agent modifies.
Baseline: simple pixel statistics features + logistic regression.

Usage: python analyze.py
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from prepare import (
    load_dataset, load_image, get_train_test_split,
    evaluate, print_results, print_detailed_report,
)

# ---------------------------------------------------------------------------
# Feature extraction (MODIFY THIS)
# ---------------------------------------------------------------------------

def extract_features(sample: dict) -> np.ndarray:
    """
    Extract features from a single sample (3 images).
    Returns a 1D feature vector.
    """
    features = []

    far = load_image(sample["far"])
    near = load_image(sample["near"])

    for img in [far, near]:
        features.append(img.mean())
        features.append(img.std())
        for ch in range(3):
            features.append(img[:, :, ch].mean())
            features.append(img[:, :, ch].std())

    if sample["card"] is not None:
        card = load_image(sample["card"])
        features.append(card.mean())
        features.append(card.std())
    else:
        features.extend([0.0, 0.0])

    return np.array(features, dtype=np.float32)


# ---------------------------------------------------------------------------
# Classification (MODIFY THIS)
# ---------------------------------------------------------------------------

def build_classifier():
    """Build and return a classifier."""
    return LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")


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
