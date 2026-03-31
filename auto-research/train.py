"""
train.py — Liveness detection training script (runs on DGX1).
Experiment: Hybrid ResNet18 deep features + handcrafted noise/freq/gradient features fusion
Key insight: handcrafted noise features (Laplacian variance) were perfect on smaller dataset,
             fusing them with deep features should give the best of both worlds.
"""
import os
import sys
import json
import time
import random
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
    """Extract noise, gradient, and frequency features from a PIL image."""
    img_np = np.array(img_pil.convert("L")).astype(np.float32)

    # Laplacian variance (noise estimation)
    from scipy.ndimage import laplace
    lap = laplace(img_np)
    noise_var = np.var(lap)
    noise_mean = np.abs(lap).mean()

    # Gradient magnitude (Sobel-like)
    gy = np.diff(img_np, axis=0)
    gx = np.diff(img_np, axis=1)
    grad_mean = (np.abs(gy).mean() + np.abs(gx).mean()) / 2.0
    grad_std = (np.std(gy) + np.std(gx)) / 2.0

    # FFT high-frequency ratio
    fft = np.fft.fft2(img_np)
    fft_shift = np.fft.fftshift(fft)
    mag = np.abs(fft_shift)
    h, w = mag.shape
    cy, cx = h // 2, w // 2
    r = min(h, w) // 8
    center_mask = np.zeros_like(mag, dtype=bool)
    Y, X = np.ogrid[:h, :w]
    center_mask[(Y - cy)**2 + (X - cx)**2 <= r**2] = True
    low_energy = mag[center_mask].sum()
    total_energy = mag.sum() + 1e-10
    hf_ratio = 1.0 - low_energy / total_energy

    # Color features from RGB
    img_rgb = np.array(img_pil.convert("RGB")).astype(np.float32)
    channel_means = img_rgb.mean(axis=(0, 1))
    blue_shift = channel_means[2] - (channel_means[0] + channel_means[1]) / 2.0

    return np.array([noise_var, noise_mean, grad_mean, grad_std, hf_ratio, blue_shift], dtype=np.float32)


class HybridLivenessDataset(Dataset):
    def __init__(self, sig_ids, labels, transform=None, img_size=224):
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

        # Handcrafted features (computed before transform)
        far_feats = extract_handcrafted_features(far_img)
        near_feats = extract_handcrafted_features(near_img)
        hand_feats = np.concatenate([far_feats, near_feats])  # 12 features

        if self.transform:
            far_img = self.transform(far_img)
            near_img = self.transform(near_img)

        img = torch.cat([far_img, near_img], dim=0)
        hand_feats_tensor = torch.from_numpy(hand_feats).float()

        return img, hand_feats_tensor, label


class HybridModel(nn.Module):
    def __init__(self, num_handcrafted=12, num_classes=2):
        super().__init__()
        backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

        old_conv = backbone.conv1
        backbone.conv1 = nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            backbone.conv1.weight[:, :3] = old_conv.weight
            backbone.conv1.weight[:, 3:] = old_conv.weight

        self.features = nn.Sequential(*list(backbone.children())[:-1])  # up to avgpool
        deep_dim = backbone.fc.in_features  # 512

        self.hand_proj = nn.Sequential(
            nn.Linear(num_handcrafted, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 64),
            nn.ReLU(),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(deep_dim + 64, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, img, hand_feats):
        deep = self.features(img).flatten(1)
        hand = self.hand_proj(hand_feats)
        fused = torch.cat([deep, hand], dim=1)
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


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce = F.cross_entropy(inputs, targets, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce)
        loss = ((1 - pt) ** self.gamma) * ce
        return loss.mean()


def train():
    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_ids, test_ids, labels = load_data()

    transform_train = T.Compose([
        T.Resize((256, 256)),
        T.RandomCrop(224),
        T.RandomHorizontalFlip(),
        T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    transform_test = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    train_ds = HybridLivenessDataset(train_ids, labels, transform_train)
    test_ds = HybridLivenessDataset(test_ids, labels, transform_test)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

    model = HybridModel(num_handcrafted=12, num_classes=2).to(device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: Hybrid ResNet18 + handcrafted (noise/grad/FFT/color) fusion, params: {num_params:,}")

    train_labels = [0 if labels[s]["main_label"] == "Positive" else 1 for s in train_ids]
    class_counts = Counter(train_labels)
    weight = torch.tensor([1.0 / class_counts[0], 1.0 / class_counts[1]], dtype=torch.float32)
    weight = weight / weight.sum() * 2
    criterion = FocalLoss(alpha=weight.to(device), gamma=2.0)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    epochs = 15
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=1e-3, steps_per_epoch=len(train_loader), epochs=epochs,
        pct_start=0.2, div_factor=10, final_div_factor=100
    )

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
            scheduler.step()
            total_loss += loss.item()

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
    approach = "hybrid_resnet18_handcrafted_noise_grad_fft_color_fusion"

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
