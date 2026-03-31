"""
train.py — Liveness detection training script (runs on DGX1).
Experiment: ResNet50 pretrained, 6-ch input, focal loss, strong augmentation,
            cosine warmup. Preload images into RAM. Small balanced dataset.
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


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()


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

        img = torch.cat([far_t, near_t], dim=0)
        return img, label


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
        print(f"Rebuilt labels.json: {len(labels)} entries")

    pos_sigs = [s for s, info in labels.items() if info["main_label"] == "Positive"]
    neg_sigs = [s for s, info in labels.items() if info["main_label"] == "Negative"]

    random.shuffle(pos_sigs)
    random.shuffle(neg_sigs)

    # Attempt to load up to target_per_class per class
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
                # Resize early to save RAM
                far = far.resize((224, 224), Image.BILINEAR)
                near = near.resize((224, 224), Image.BILINEAR)
                imgs.append((sig, far, near))
            except Exception:
                continue
            if len(imgs) % 50 == 0:
                print(f"  Loaded {len(imgs)}/{target} ({time.time()-t_load:.1f}s)")
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


def build_model(num_classes=2):
    model = models.resnet50(pretrained=True)

    old_conv = model.conv1
    model.conv1 = nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3, bias=False)
    with torch.no_grad():
        model.conv1.weight[:, :3] = old_conv.weight
        model.conv1.weight[:, 3:] = old_conv.weight

    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(model.fc.in_features, num_classes),
    )
    return model


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
        T.RandomRotation(15),
        T.RandomGrayscale(p=0.1),
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

    model = build_model().to(device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: ResNet50 (6-ch), params: {num_params:,}")

    class_counts = Counter(train_labels)
    weight = torch.tensor([1.0 / class_counts[0], 1.0 / class_counts[1]], dtype=torch.float32)
    weight = weight / weight.sum() * 2
    criterion = FocalLoss(alpha=weight.to(device), gamma=2.0)

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-3)

    remaining = MAX_SECONDS - (time.time() - t0)
    epochs = 20
    warmup_epochs = 2

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
        for imgs, targets in train_loader:
            imgs, targets = imgs.to(device), targets.to(device)
            optimizer.zero_grad()
            out = model(imgs)
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
            for imgs, targets in test_loader:
                imgs = imgs.to(device)
                out = model(imgs)
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
    approach = "resnet50_6ch_focal_loss_strong_aug_cosine_warmup_400samples"

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
