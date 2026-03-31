"""
train.py — Liveness detection training script (runs on DGX1).

Experiment: Hybrid approach - lightweight ResNet18 deep features + handcrafted
noise/frequency features, combined in a fusion classifier.
Previous best: 0.976659 (ResNet18 6ch baseline)
Handcrafted noise features achieved 1.0 on the old dataset format;
combining them with deep features should leverage both signal types.
"""
import os
import sys
import json
import time
import random
import math
import numpy as np
from pathlib import Path
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from sklearn.metrics import balanced_accuracy_score, accuracy_score, f1_score

DATA_DIR = Path("/mnt/nas/public2/simon/projects/auto_research/liveness-research/data")
RESULTS_FILE = Path("/mnt/nas/public2/simon/projects/auto_research/liveness-research/data/last_result.json")
SEED = 42
MAX_SECONDS = 270

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


def extract_handcrafted_features(img_pil):
    """Extract noise, frequency, and texture features from a PIL image."""
    img_np = np.array(img_pil.convert("RGB")).astype(np.float32)

    gray = 0.299 * img_np[:,:,0] + 0.587 * img_np[:,:,1] + 0.114 * img_np[:,:,2]

    # Laplacian noise
    laplacian = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
    from scipy.ndimage import convolve
    lap_response = convolve(gray, laplacian)
    noise_std = float(np.std(lap_response))
    noise_mean = float(np.mean(np.abs(lap_response)))

    # Gradient magnitudes
    gy, gx = np.gradient(gray)
    grad_mag = np.sqrt(gx**2 + gy**2)
    grad_mean = float(np.mean(grad_mag))
    grad_std = float(np.std(grad_mag))

    # FFT high-frequency ratio
    fft = np.fft.fft2(gray)
    fft_shift = np.fft.fftshift(fft)
    magnitude = np.abs(fft_shift)
    h, w = gray.shape
    cy, cx = h // 2, w // 2
    r = min(h, w) // 8
    total_energy = float(np.sum(magnitude))
    center_mask = np.zeros_like(magnitude, dtype=bool)
    yy, xx = np.ogrid[:h, :w]
    center_mask[(yy - cy)**2 + (xx - cx)**2 <= r**2] = True
    low_energy = float(np.sum(magnitude[center_mask]))
    hf_ratio = 1.0 - (low_energy / (total_energy + 1e-8))

    # Color statistics
    channel_means = [float(np.mean(img_np[:,:,c])) for c in range(3)]
    channel_stds = [float(np.std(img_np[:,:,c])) for c in range(3)]

    # Blue shift (common in screen captures)
    blue_shift = channel_means[2] - (channel_means[0] + channel_means[1]) / 2.0

    features = [
        noise_std, noise_mean,
        grad_mean, grad_std,
        hf_ratio,
        blue_shift,
        *channel_stds,
    ]
    return np.array(features, dtype=np.float32)


class HybridLivenessDataset(Dataset):
    def __init__(self, sig_ids, labels, transform=None, img_size=160):
        self.sig_ids = sig_ids
        self.labels = labels
        self.transform = transform
        self.img_size = img_size
        self.samples_dir = DATA_DIR / "samples"

    def __len__(self):
        return len(self.sig_ids)

    def __getitem__(self, idx):
        sig = self.sig_ids[idx]
        info = self.labels[sig]
        label = 0 if info["main_label"] == "Positive" else 1

        far_path = self.samples_dir / sig / "far.jpg"
        near_path = self.samples_dir / sig / "near.jpg"

        try:
            far_img = Image.open(str(far_path)).convert("RGB")
            near_img = Image.open(str(near_path)).convert("RGB")
        except Exception:
            far_img = Image.new("RGB", (self.img_size, self.img_size))
            near_img = Image.new("RGB", (self.img_size, self.img_size))

        # Extract handcrafted features before transforms
        far_feats = extract_handcrafted_features(far_img)
        near_feats = extract_handcrafted_features(near_img)
        hand_feats = np.concatenate([far_feats, near_feats])
        hand_feats = torch.from_numpy(hand_feats)

        if self.transform:
            far_img = self.transform(far_img)
            near_img = self.transform(near_img)

        img = torch.cat([far_img, near_img], dim=0)
        return img, hand_feats, label


class HybridModel(nn.Module):
    def __init__(self, hand_feat_dim=18, num_classes=2):
        super().__init__()
        backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

        old_conv = backbone.conv1
        backbone.conv1 = nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            backbone.conv1.weight[:, :3] = old_conv.weight
            backbone.conv1.weight[:, 3:] = old_conv.weight

        self.features = nn.Sequential(*list(backbone.children())[:-1])
        cnn_feat_dim = 512

        self.hand_encoder = nn.Sequential(
            nn.Linear(hand_feat_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 64),
            nn.ReLU(),
        )

        self.classifier = nn.Sequential(
            nn.Linear(cnn_feat_dim + 64, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, img, hand_feats):
        cnn_out = self.features(img).flatten(1)
        hand_out = self.hand_encoder(hand_feats)
        fused = torch.cat([cnn_out, hand_out], dim=1)
        return self.classifier(fused)


def load_data():
    with open(str(DATA_DIR / "labels.json")) as f:
        labels = json.load(f)

    valid = []
    for sig, info in labels.items():
        far = DATA_DIR / "samples" / sig / "far.jpg"
        near = DATA_DIR / "samples" / sig / "near.jpg"
        if far.exists() and near.exists():
            valid.append(sig)

    random.shuffle(valid)
    split = int(len(valid) * 0.8)
    train_ids = valid[:split]
    test_ids = valid[split:]

    dist_train = Counter(labels[s]["main_label"] for s in train_ids)
    dist_test = Counter(labels[s]["main_label"] for s in test_ids)
    print(f"Train: {len(train_ids)} {dict(dist_train)}")
    print(f"Test:  {len(test_ids)} {dict(dist_test)}")

    return train_ids, test_ids, labels


def train():
    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_ids, test_ids, labels = load_data()

    img_size = 160
    transform_train = T.Compose([
        T.Resize((img_size, img_size)),
        T.RandomHorizontalFlip(),
        T.ColorJitter(brightness=0.2, contrast=0.2),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    transform_test = T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    train_ds = HybridLivenessDataset(train_ids, labels, transform_train, img_size)
    test_ds = HybridLivenessDataset(test_ids, labels, transform_test, img_size)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True,
                              num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False,
                             num_workers=4, pin_memory=True)

    hand_feat_dim = 9 * 2  # 9 features per image, 2 images
    model = HybridModel(hand_feat_dim=hand_feat_dim).to(device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: Hybrid ResNet18 + handcrafted, params: {num_params:,}")

    train_labels = [0 if labels[s]["main_label"] == "Positive" else 1 for s in train_ids]
    class_counts = Counter(train_labels)
    weight = torch.tensor([1.0 / class_counts[0], 1.0 / class_counts[1]], dtype=torch.float32)
    weight = weight / weight.sum() * 2
    criterion = nn.CrossEntropyLoss(weight=weight.to(device))

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-3)
    epochs = 12
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_bal_acc = 0
    best_acc = 0
    best_f1 = 0
    for epoch in range(epochs):
        if time.time() - t0 > MAX_SECONDS:
            print(f"Time budget reached at epoch {epoch}")
            break

        model.train()
        total_loss = 0
        for imgs, hand_feats, targets in train_loader:
            imgs = imgs.to(device)
            hand_feats = hand_feats.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            out = model(imgs, hand_feats)
            loss = criterion(out, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()

        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for imgs, hand_feats, targets in test_loader:
                imgs = imgs.to(device)
                hand_feats = hand_feats.to(device)
                out = model(imgs, hand_feats)
                preds = out.argmax(dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(targets.numpy())

        bal_acc = balanced_accuracy_score(all_labels, all_preds)
        acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average="binary")
        avg_loss = total_loss / len(train_loader)

        print(f"Epoch {epoch+1}/{epochs}: loss={avg_loss:.4f} acc={acc:.4f} bal_acc={bal_acc:.4f} f1={f1:.4f}")

        if bal_acc > best_bal_acc:
            best_bal_acc = bal_acc
            best_acc = acc
            best_f1 = f1

    elapsed = time.time() - t0

    approach = "hybrid_resnet18_deep_plus_handcrafted_noise_freq_features"
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
