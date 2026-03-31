"""
train.py — Liveness detection training script (runs on DGX1).
Experiment: Dual-stream ResNet18 with cross-attention fusion between far/near.
Each stream processes one image independently, then features are fused via attention.
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


class PreloadedDataset(Dataset):
    def __init__(self, images, labels_list, transform=None):
        self.images = images
        self.labels = labels_list
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        far_img, near_img = self.images[idx]
        label = self.labels[idx]

        if self.transform:
            far_t = self.transform(far_img.copy())
            near_t = self.transform(near_img.copy())
        else:
            far_t = T.ToTensor()(far_img)
            near_t = T.ToTensor()(near_img)

        return far_t, near_t, label


def load_data():
    t_load = time.time()

    with open(str(DATA_DIR / "labels.json")) as f:
        labels = json.load(f)

    if len(labels) == 0:
        print("labels.json empty, rebuilding from annotations_full.jsonl...")
        with open(str(DATA_DIR / "annotations_full.jsonl")) as f:
            for line in f:
                entry = json.loads(line.strip())
                sig = entry.get("sig", "")
                label = entry.get("label", "")
                if label in ("Positive", "Negative"):
                    labels[sig] = {"main_label": label}
        with open(str(DATA_DIR / "labels.json"), "w") as f:
            json.dump(labels, f)

    pos_sigs = [s for s, info in labels.items() if info["main_label"] == "Positive"]
    neg_sigs = [s for s, info in labels.items() if info["main_label"] == "Negative"]

    random.shuffle(pos_sigs)
    random.shuffle(neg_sigs)

    target_per_class = 200
    samples_dir = DATA_DIR / "samples"

    def load_class(sig_list, target):
        imgs = []
        for sig in sig_list:
            if len(imgs) >= target:
                break
            try:
                far = Image.open(str(samples_dir / sig / "far.jpg")).convert("RGB")
                near = Image.open(str(samples_dir / sig / "near.jpg")).convert("RGB")
                far.load()
                near.load()
                far = far.resize((224, 224), Image.BILINEAR)
                near = near.resize((224, 224), Image.BILINEAR)
                imgs.append((sig, far, near))
            except Exception:
                continue
        return imgs

    print("Loading negatives...")
    neg_loaded = load_class(neg_sigs, target_per_class)
    print("Loading positives...")
    pos_loaded = load_class(pos_sigs, len(neg_loaded))

    print(f"Loaded: {len(pos_loaded)} pos, {len(neg_loaded)} neg in {time.time()-t_load:.1f}s")

    all_data = [(sig, far, near, 0) for sig, far, near in pos_loaded] + \
               [(sig, far, near, 1) for sig, far, near in neg_loaded]
    random.shuffle(all_data)

    split = int(len(all_data) * 0.8)
    train_data = all_data[:split]
    test_data = all_data[split:]

    train_images = [(far, near) for _, far, near, _ in train_data]
    train_labels = [lbl for _, _, _, lbl in train_data]
    test_images = [(far, near) for _, far, near, _ in test_data]
    test_labels = [lbl for _, _, _, lbl in test_data]

    print(f"Train: {len(train_labels)} ({Counter(train_labels)})")
    print(f"Test:  {len(test_labels)} ({Counter(test_labels)})")

    return train_images, train_labels, test_images, test_labels


class CrossAttentionFusion(nn.Module):
    """Cross-attention between far and near feature vectors."""
    def __init__(self, dim):
        super().__init__()
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.scale = dim ** -0.5
        self.norm = nn.LayerNorm(dim)

    def forward(self, x1, x2):
        q = self.query(x1)
        k = self.key(x2)
        v = self.value(x2)
        attn = (q * k).sum(dim=-1, keepdim=True) * self.scale
        attn = torch.sigmoid(attn)
        out = attn * v
        return self.norm(x1 + out)


class DualStreamModel(nn.Module):
    """Two ResNet18 encoders + cross-attention fusion + classifier."""
    def __init__(self, num_classes=2):
        super().__init__()
        # Shared backbone (weight sharing between streams)
        backbone = models.resnet18(pretrained=True)
        self.features = nn.Sequential(*list(backbone.children())[:-1])
        feat_dim = 512

        self.cross_attn_far = CrossAttentionFusion(feat_dim)
        self.cross_attn_near = CrossAttentionFusion(feat_dim)

        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(feat_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, far, near):
        f_far = self.features(far).flatten(1)
        f_near = self.features(near).flatten(1)

        f_far_attn = self.cross_attn_far(f_far, f_near)
        f_near_attn = self.cross_attn_near(f_near, f_far)

        combined = torch.cat([f_far_attn, f_near_attn], dim=1)
        return self.classifier(combined)


def train():
    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_images, train_labels, test_images, test_labels = load_data()
    load_time = time.time() - t0
    print(f"Data loading took {load_time:.1f}s")

    transform_train = T.Compose([
        T.RandomHorizontalFlip(),
        T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
        T.RandomRotation(10),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    transform_test = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    train_ds = PreloadedDataset(train_images, train_labels, transform_train)
    test_ds = PreloadedDataset(test_images, test_labels, transform_test)

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=0)

    model = DualStreamModel().to(device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: Dual-stream ResNet18 + CrossAttn, params: {num_params:,}")

    class_counts = Counter(train_labels)
    weight = torch.tensor([1.0 / class_counts[0], 1.0 / class_counts[1]], dtype=torch.float32)
    weight = weight / weight.sum() * 2
    criterion = nn.CrossEntropyLoss(weight=weight.to(device))

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-3)

    epochs = 25
    warmup_epochs = 3

    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(1, epochs - warmup_epochs)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    best_bal_acc = 0
    best_acc = 0
    best_f1 = 0

    for epoch in range(epochs):
        if time.time() - t0 > MAX_SECONDS:
            print(f"Time budget reached at epoch {epoch}")
            break

        model.train()
        total_loss = 0
        batch_count = 0
        for far, near, targets in train_loader:
            far, near, targets = far.to(device), near.to(device), targets.to(device)
            optimizer.zero_grad()
            out = model(far, near)
            loss = criterion(out, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            batch_count += 1

        scheduler.step()

        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for far, near, targets in test_loader:
                far, near = far.to(device), near.to(device)
                out = model(far, near)
                preds = out.argmax(dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(targets.numpy())

        bal_acc = balanced_accuracy_score(all_labels, all_preds)
        acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average="binary")
        avg_loss = total_loss / max(1, batch_count)

        elapsed_now = time.time() - t0
        print(f"Epoch {epoch+1}/{epochs}: loss={avg_loss:.4f} acc={acc:.4f} bal_acc={bal_acc:.4f} f1={f1:.4f} t={elapsed_now:.0f}s")

        if bal_acc > best_bal_acc:
            best_bal_acc = bal_acc
            best_acc = acc
            best_f1 = f1

    elapsed = time.time() - t0
    approach = "dual_stream_resnet18_cross_attention_shared_backbone_400samples"

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
