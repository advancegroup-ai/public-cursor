"""
train.py — Liveness detection training script (runs on DGX1).
Experiment: Multi-scale ResNet18 with resolution fusion. Processes each image
at 3 different scales (112, 224, 336) to capture both fine-grained artifacts
(large scale) and global patterns (small scale). Features from each scale
are concatenated before classification.
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


class MultiScaleDataset(Dataset):
    def __init__(self, sig_ids, labels, sizes=(112, 224, 336), is_train=True):
        self.sig_ids = sig_ids
        self.labels = labels
        self.sizes = sizes
        self.is_train = is_train
        self.samples_dir = DATA_DIR / "samples"
        self.normalize = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    def __len__(self):
        return len(self.sig_ids)

    def _load_img(self, path, size):
        try:
            img = Image.open(str(path)).convert("RGB")
        except Exception:
            img = Image.new("RGB", (size, size))
        img = T.Resize((size, size))(img)
        if self.is_train:
            if random.random() > 0.5:
                img = T.functional.hflip(img)
            img = T.ColorJitter(0.2, 0.2, 0.1, 0.05)(img)
        return self.normalize(T.ToTensor()(img))

    def __getitem__(self, idx):
        sig = self.sig_ids[idx]
        info = self.labels[sig]
        label = 0 if info["main_label"] == "Positive" else 1

        far_path = self.samples_dir / sig / "far.jpg"
        near_path = self.samples_dir / sig / "near.jpg"

        images = []
        for size in self.sizes:
            images.append(self._load_img(far_path, size))
            images.append(self._load_img(near_path, size))

        return images, label


def collate_multiscale(batch):
    """Custom collate for variable-size tensors."""
    num_scales = len(batch[0][0])
    batched_images = []
    labels = []
    for i in range(num_scales):
        imgs = torch.stack([item[0][i] for item in batch])
        batched_images.append(imgs)
    labels = torch.tensor([item[1] for item in batch], dtype=torch.long)
    return batched_images, labels


class MultiScaleModel(nn.Module):
    def __init__(self, num_classes=2, sizes=(112, 224, 336)):
        super().__init__()
        self.num_scales = len(sizes)

        backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.shared_encoder = nn.Sequential(*list(backbone.children())[:-1])  # up to avgpool
        feat_dim = 512

        self.scale_projections = nn.ModuleList([
            nn.Linear(feat_dim, 128) for _ in range(self.num_scales * 2)
        ])

        total_feat = 128 * self.num_scales * 2
        self.classifier = nn.Sequential(
            nn.Linear(total_feat, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes),
        )

    def forward(self, image_list):
        feats = []
        for i, img in enumerate(image_list):
            f = self.shared_encoder(img).flatten(1)
            f = self.scale_projections[i](f)
            feats.append(f)
        combined = torch.cat(feats, dim=1)
        return self.classifier(combined)


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

    sizes = (112, 224, 336)
    train_ds = MultiScaleDataset(train_ids, labels, sizes=sizes, is_train=True)
    test_ds = MultiScaleDataset(test_ids, labels, sizes=sizes, is_train=False)

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True,
                              num_workers=4, pin_memory=True, collate_fn=collate_multiscale)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False,
                             num_workers=4, pin_memory=True, collate_fn=collate_multiscale)

    model = MultiScaleModel(sizes=sizes).to(device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: Multi-scale ResNet18 (shared encoder, 3 scales), params: {num_params:,}")

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
        if time.time() - t0 > MAX_SECONDS - 30:
            print(f"Time budget approaching at epoch {epoch}")
            break

        model.train()
        total_loss = 0
        for image_list, targets in train_loader:
            image_list = [img.to(device) for img in image_list]
            targets = targets.to(device)
            optimizer.zero_grad()
            out = model(image_list)
            loss = criterion(out, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()

        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for image_list, targets in test_loader:
                image_list = [img.to(device) for img in image_list]
                out = model(image_list)
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

    approach = "multiscale_resnet18_shared_3scales_112_224_336_far_near"
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
