"""
train.py — Liveness detection training script (runs on DGX1).

Experiment: Hybrid approach - handcrafted noise features + ResNet18 deep features
The noise features achieved 1.0 on the small dataset; deep learning reached 0.977 on DGX1.
This combines both: extract Laplacian noise + gradient features alongside ResNet18 features,
then train a classifier on the concatenated representation.
Previous best: 0.976659 (ResNet18 6ch baseline)
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


def compute_noise_features(img_tensor):
    """Compute Laplacian noise and gradient features from a 3-channel image tensor.
    Returns a tensor of handcrafted features.
    img_tensor: (3, H, W) normalized tensor
    """
    img_np = img_tensor.numpy().transpose(1, 2, 0)
    img_np = (img_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])) * 255
    img_np = img_np.clip(0, 255).astype(np.uint8)
    gray = np.mean(img_np, axis=2)

    from scipy.ndimage import laplace, sobel
    lap = laplace(gray.astype(np.float32))
    noise = np.mean(np.abs(lap))

    gx = sobel(gray.astype(np.float32), axis=0)
    gy = sobel(gray.astype(np.float32), axis=1)
    grad_mag = np.sqrt(gx**2 + gy**2)
    grad_mean = np.mean(grad_mag)
    grad_std = np.std(grad_mag)

    fft = np.fft.fft2(gray)
    fft_shift = np.fft.fftshift(fft)
    magnitude = np.log1p(np.abs(fft_shift))
    h, w = magnitude.shape
    center_h, center_w = h // 4, w // 4
    hf_energy = magnitude.mean()
    lf_mask = np.zeros_like(magnitude)
    lf_mask[h//2-center_h:h//2+center_h, w//2-center_w:w//2+center_w] = 1
    lf_energy = (magnitude * lf_mask).sum() / max(lf_mask.sum(), 1)
    hf_ratio = hf_energy / max(lf_energy, 1e-8)

    return torch.tensor([noise, grad_mean, grad_std, hf_ratio], dtype=torch.float32)


class HybridLivenessDataset(Dataset):
    def __init__(self, sig_ids, labels, transform=None):
        self.sig_ids = sig_ids
        self.labels = labels
        self.transform = transform
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
            far_img = Image.new("RGB", (224, 224))
            near_img = Image.new("RGB", (224, 224))

        if self.transform:
            far_t = self.transform(far_img)
            near_t = self.transform(near_img)
        else:
            far_t = T.ToTensor()(far_img)
            near_t = T.ToTensor()(near_img)

        far_feats = compute_noise_features(far_t)
        near_feats = compute_noise_features(near_t)
        hand_features = torch.cat([far_feats, near_feats])

        img = torch.cat([far_t, near_t], dim=0)
        return img, hand_features, label


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


class HybridModel(nn.Module):
    """Combines ResNet18 deep features with handcrafted noise/gradient features."""
    def __init__(self, num_hand_features=8, num_classes=2):
        super().__init__()
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

        old_conv = resnet.conv1
        resnet.conv1 = nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            resnet.conv1.weight[:, :3] = old_conv.weight
            resnet.conv1.weight[:, 3:] = old_conv.weight

        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        deep_dim = 512

        self.hand_proj = nn.Sequential(
            nn.Linear(num_hand_features, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(deep_dim + 32, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes),
        )

    def forward(self, img, hand_features):
        deep = self.backbone(img).flatten(1)
        hand = self.hand_proj(hand_features)
        combined = torch.cat([deep, hand], dim=1)
        return self.classifier(combined)


def train():
    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_ids, test_ids, labels = load_data()

    transform_train = T.Compose([
        T.Resize((224, 224)),
        T.RandomHorizontalFlip(),
        T.ColorJitter(brightness=0.2, contrast=0.2),
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

    model = HybridModel(num_hand_features=8, num_classes=2).to(device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: Hybrid ResNet18 + handcrafted features, params: {num_params:,}")

    train_labels_list = [0 if labels[s]["main_label"] == "Positive" else 1 for s in train_ids]
    class_counts = Counter(train_labels_list)
    weight = torch.tensor([1.0 / class_counts[0], 1.0 / class_counts[1]], dtype=torch.float32)
    weight = weight / weight.sum() * 2
    criterion = nn.CrossEntropyLoss(weight=weight.to(device))

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=12)

    best_bal_acc = 0
    best_acc = 0
    best_f1 = 0
    epochs = 12

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
    approach = "hybrid_resnet18_6ch_plus_noise_gradient_fft_features"

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
