"""
Generate synthetic liveness detection dataset for auto-research POC.
Uses vectorized numpy — runs in seconds.

Run once: python generate_synthetic_data.py
"""

import json
import random
import numpy as np
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
SAMPLES_DIR = DATA_DIR / "samples"
LABELS_FILE = DATA_DIR / "labels.json"

random.seed(42)
np.random.seed(42)

IMG_SIZE = 64


def make_image(is_live: bool, attack_type: str = "", view: str = "far") -> np.ndarray:
    """Generate synthetic face-like image with distinguishing features (vectorized)."""
    base_r = np.random.randint(160, 220)
    base_g = np.random.randint(120, 180)
    base_b = np.random.randint(90, 150)
    noise_std = np.random.uniform(4, 10) if is_live else np.random.uniform(1, 5)

    ys = np.arange(IMG_SIZE).reshape(-1, 1)
    gradient = (ys / IMG_SIZE) * 20 - 10
    noise = np.random.normal(0, noise_std, (IMG_SIZE, IMG_SIZE, 3))

    img = np.stack([
        np.full((IMG_SIZE, IMG_SIZE), base_r, dtype=float),
        np.full((IMG_SIZE, IMG_SIZE), base_g, dtype=float),
        np.full((IMG_SIZE, IMG_SIZE), base_b, dtype=float),
    ], axis=-1)
    img += gradient[:, :, None] + noise

    # Face oval
    yy, xx = np.mgrid[:IMG_SIZE, :IMG_SIZE]
    cy, cx = IMG_SIZE // 2, IMG_SIZE // 2
    dist = ((yy - cy) / (IMG_SIZE / 3)) ** 2 + ((xx - cx) / (IMG_SIZE / 4)) ** 2
    face_mask = (dist < 1.0).astype(float)
    img += (20 * (1 - dist) * face_mask)[:, :, None]

    if view == "near":
        img += 10
    elif view == "card":
        img -= 15
        img[:3, :] = 200; img[-3:, :] = 200
        img[:, :3] = 200; img[:, -3:] = 200

    if not is_live:
        if attack_type == "Screen":
            stripe = np.zeros((IMG_SIZE, IMG_SIZE, 3))
            stripe[::3, :, :] = 15
            img += stripe
            img[:, :, 2] += 12  # blue tint
        elif attack_type in ("Color_Photocopy", "B&W_Photocopy"):
            mean_val = img.mean(axis=(0, 1), keepdims=True)
            img = img * 0.6 + mean_val * 0.4
            if "B&W" in attack_type:
                gray = img.mean(axis=2, keepdims=True)
                img = np.broadcast_to(gray, img.shape).copy()
        elif attack_type == "Fake_Card":
            img = img * 0.5 + 128 * 0.5
        elif attack_type == "Screenshot":
            img[:, :, 1] += 8  # green tint
        elif attack_type == "Injection":
            from scipy.ndimage import uniform_filter
            for c in range(3):
                img[:, :, c] = uniform_filter(img[:, :, c], size=3)

    return np.clip(img, 0, 255).astype(np.uint8)


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)

    labels = {}
    pos_spec = [("Clear", 250), ("Blur", 40), ("Glossy", 30), ("Other", 20), ("Dim", 10)]
    neg_spec = [("Screen", 50), ("Color_Photocopy", 30), ("B&W_Photocopy", 20),
                ("Fake_Card", 15), ("Screenshot", 15), ("Injection", 10), ("Other", 10)]
    regions = ["INDONESIA", "THAILAND", "MALAYSIA", "BRAZIL"]

    idx = 0
    for sublabel, count in pos_spec:
        for _ in range(count):
            sig = f"pos_{idx:04d}"
            d = SAMPLES_DIR / sig; d.mkdir(exist_ok=True)
            for v in ["far", "near", "card"]:
                np.save(str(d / f"{v}.npy"), make_image(True, view=v))
            labels[sig] = {"label": f"Positive/*/{sublabel}", "sublabel": sublabel,
                           "main_label": "Positive", "region": random.choice(regions)}
            idx += 1

    neg_start = idx
    idx = 0
    for attack_type, count in neg_spec:
        for _ in range(count):
            sig = f"neg_{idx:04d}"
            d = SAMPLES_DIR / sig; d.mkdir(exist_ok=True)
            for v in ["far", "near", "card"]:
                np.save(str(d / f"{v}.npy"), make_image(False, attack_type, v))
            labels[sig] = {"label": f"Negative_Type/*/{attack_type}", "sublabel": attack_type,
                           "main_label": "Negative", "region": random.choice(regions)}
            idx += 1

    with open(LABELS_FILE, "w") as f:
        json.dump(labels, f, indent=2)
    print(f"Generated {neg_start} positive + {idx} negative = {neg_start + idx} samples")
    print(f"Data dir size: {sum(f.stat().st_size for f in SAMPLES_DIR.rglob('*') if f.is_file()) / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    main()
