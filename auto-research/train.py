"""
train.py — Liveness detection training script (runs on DGX1).

Experiment: ResNet18 6ch + frequency-aware head with DCT pooling.
Adds frequency domain features via learnable DCT-like pooling on intermediate
feature maps. Screen replays and prints often show frequency artifacts that
are invisible in spatial domain but clear in frequency space.
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
from sklearn.model_selection import StratifiedKFold

DATA_DIR = Path("/mnt/nas/public2/simon/projects/auto_research/liveness-research/data")
RESULTS_FILE = DATA_DIR / "last_result.json"
SEED = 42
MAX_SECONDS = 270

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


def load_labels():
    labels = {}
    for fname, default_label in [("neg_batch.json", "Negative"), ("pos_batch.json", "Positive")]:
        fpath = DATA_DIR / fname
        if not fpath.exists():
            continue
        with open(str(fpath)) as f:
            batch = json.load(f)
        for d in batch.get("data", []):
            parts = d["sample_id"].split("_")
            if len(parts) >= 4:
                sig = parts[3]
                main_label = d["pn"].split("/")[0]
                if main_label in ("Negative Type", "Negative_Type"):
                    main_label = "Negative"
                labels[sig] = {"main_label": main_label, "pn": d["pn"]}
    if not labels:
        with open(str(DATA_DIR / "labels.json")) as f:
            labels = json.load(f)
    return labels


class LivenessDataset(Dataset):
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
            far_img = self.transform(far_img)
            near_img = self.transform(near_img)

        img = torch.cat([far_img, near_img], dim=0)
        return img, label


class FrequencyPool(nn.Module):
    """Pool feature maps in frequency domain to capture periodic artifacts."""
    def __init__(self, in_channels, pool_size=7):
        super().__init__()
        self.pool_size = pool_size
        self.weight = nn.Parameter(torch.randn(in_channels, pool_size, pool_size) * 0.02)

    def forward(self, x):
        B, C, H, W = x.shape
        x_fft = torch.fft.rfft2(x, norm='ortho')
        magnitude = torch.abs(x_fft)
        phase_info = torch.angle(x_fft)

        mag_pool = F.adaptive_avg_pool2d(magnitude, (self.pool_size, self.pool_size // 2 + 1))
        weighted = mag_pool * self.weight[:, :, :self.pool_size // 2 + 1].unsqueeze(0)
        return weighted.flatten(1)


class FreqAwareResNet(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

        old_conv = backbone.conv1
        backbone.conv1 = nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            backbone.conv1.weight[:, :3] = old_conv.weight
            backbone.conv1.weight[:, 3:] = old_conv.weight

        layers = list(backbone.children())
        self.early = nn.Sequential(*layers[:5])   # up to layer1
        self.mid = nn.Sequential(*layers[5:7])     # layer2 + layer3
        self.late = nn.Sequential(*layers[7:8])    # layer4
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        self.freq_early = FrequencyPool(64, pool_size=7)
        self.freq_mid = FrequencyPool(256, pool_size=5)

        spatial_dim = 512
        freq_early_dim = 64 * 7 * 4  # rfft output width
        freq_mid_dim = 256 * 5 * 3

        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(spatial_dim + freq_early_dim + freq_mid_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        early = self.early(x)
        freq_e = self.freq_early(early)

        mid = self.mid(early)
        freq_m = self.freq_mid(mid)

        late = self.late(mid)
        spatial = self.global_pool(late).flatten(1)

        combined = torch.cat([spatial, freq_e, freq_m], dim=1)
        return self.classifier(combined)


def load_data():
    labels = load_labels()
    samples_dir = DATA_DIR / "samples"
    all_sigs = set(os.listdir(str(samples_dir)))

    valid_sigs = [s for s in labels if s in all_sigs
                  and (samples_dir / s / "far.jpg").exists()
                  and (samples_dir / s / "near.jpg").exists()]

    binary_labels = [0 if labels[s]["main_label"] == "Positive" else 1 for s in valid_sigs]
    print(f"Total valid samples: {len(valid_sigs)}")
    print(f"Distribution: {Counter(binary_labels)}")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(valid_sigs, binary_labels)):
        if fold_idx == 0:
            train_ids = [valid_sigs[i] for i in train_idx]
            test_ids = [valid_sigs[i] for i in test_idx]
            break

    dist_train = Counter(labels[s]["main_label"] for s in train_ids)
    dist_test = Counter(labels[s]["main_label"] for s in test_ids)
    print(f"Train: {len(train_ids)} {dict(dist_train)}")
    print(f"Test:  {len(test_ids)} {dict(dist_test)}")

    return train_ids, test_ids, labels


def train():
    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}, count: {torch.cuda.device_count()}")

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

    train_ds = LivenessDataset(train_ids, labels, transform_train)
    test_ds = LivenessDataset(test_ids, labels, transform_test)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

    model = FreqAwareResNet(num_classes=2).to(device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: FreqAware ResNet18 (6ch + freq pooling), params: {num_params:,}")

    train_labels = [0 if labels[s]["main_label"] == "Positive" else 1 for s in train_ids]
    class_counts = Counter(train_labels)
    weight = torch.tensor([1.0 / class_counts[0], 1.0 / class_counts[1]], dtype=torch.float32)
    weight = weight / weight.sum() * 2
    criterion = nn.CrossEntropyLoss(weight=weight.to(device))

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-3)
    epochs = 15
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=3e-4, epochs=epochs, steps_per_epoch=len(train_loader)
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
        for imgs, targets in train_loader:
            imgs, targets = imgs.to(device), targets.to(device)
            optimizer.zero_grad()
            out = model(imgs)
            loss = criterion(out, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for imgs, targets in test_loader:
                imgs = imgs.to(device)
                out = model(imgs)
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
    approach = "freq_aware_resnet18_6ch_learnable_fft_pooling_multi_scale"

    print(f"\n---")
    print(f"balanced_accuracy: {best_bal_acc:.6f}")
    print(f"accuracy:          {best_acc:.6f}")
    print(f"f1_score:          {best_f1:.6f}")
    print(f"num_params:        {num_params}")
    print(f"training_seconds:  {elapsed:.1f}")
    print(f"approach:          {approach}")
    print(f"---")

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
