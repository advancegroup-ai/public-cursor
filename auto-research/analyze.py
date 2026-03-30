"""
Liveness detection analysis — the file the agent modifies.
Experiment 2: Simplified feature set targeting key attack signatures.

Usage: python analyze.py
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from prepare import (
    load_dataset, load_image, get_train_test_split,
    evaluate, print_results, print_detailed_report,
)

# ---------------------------------------------------------------------------
# Feature extraction (MODIFY THIS)
# ---------------------------------------------------------------------------

def _core_features(img):
    """Core per-image features: stats + noise + color + contrast."""
    features = []
    r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    gray = img.mean(axis=2)

    features.append(float(gray.mean()))
    features.append(float(gray.std()))

    for ch in [r, g, b]:
        features.append(float(ch.mean()))
        features.append(float(ch.std()))

    features.append(float(r.mean() - b.mean()))
    features.append(float(g.mean() - b.mean()))

    lap = np.zeros_like(gray)
    lap[1:-1, 1:-1] = (gray[:-2, 1:-1] + gray[2:, 1:-1] +
                        gray[1:-1, :-2] + gray[1:-1, 2:] -
                        4 * gray[1:-1, 1:-1])
    features.append(float(np.abs(lap).mean()))
    features.append(float(lap.var()))

    features.append(float(np.percentile(gray, 95) - np.percentile(gray, 5)))

    channel_std = np.std(img, axis=2)
    features.append(float(channel_std.mean()))
    features.append(float((channel_std < 0.01).mean()))

    row_means = gray.mean(axis=1)
    fft_row = np.abs(np.fft.rfft(row_means - row_means.mean()))
    n = len(fft_row)
    features.append(float(fft_row[n // 3]) if n > 3 else 0.0)
    features.append(float(fft_row[1:].max()) if n > 1 else 0.0)

    return features


def extract_features(sample: dict) -> np.ndarray:
    """Extract features from a single sample."""
    features = []

    far = load_image(sample["far"])
    near = load_image(sample["near"])

    for img in [far, near]:
        features.extend(_core_features(img))

    features.append(float(np.abs(far.mean() - near.mean())))
    features.append(float(np.abs(far.std() - near.std())))

    if sample["card"] is not None:
        card = load_image(sample["card"])
        features.append(float(card.mean()))
        features.append(float(card.std()))
        features.append(float(np.percentile(card.mean(axis=2), 95) -
                               np.percentile(card.mean(axis=2), 5)))
    else:
        features.extend([0.0, 0.0, 0.0])

    return np.array(features, dtype=np.float32)


# ---------------------------------------------------------------------------
# Classification (MODIFY THIS)
# ---------------------------------------------------------------------------

def build_classifier():
    """Build and return a classifier."""
    return RandomForestClassifier(
        n_estimators=100,
        max_depth=None,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
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
