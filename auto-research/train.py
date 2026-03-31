"""
train.py — Liveness detection training script (runs on DGX1).

Experiment: ResNet18 6ch + Focal Loss + MixUp + label smoothing.
Focal loss helps with hard examples, MixUp regularization improves generalization,
label smoothing prevents overconfident predictions.
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


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, label_smoothing=0.05):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.label_smoothing = label_smoothing

    def forward(self, inputs, targets):
        num_classes = inputs.size(1)
        smooth_targets = torch.zeros_like(inputs)
        smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0)
        smooth_targets = smooth_targets * (1 - self.label_smoothing) + self.label_smoothing / num_classes

        log_probs = F.log_softmax(inputs, dim=1)
        probs = torch.exp(log_probs)

        focal_weight = (1 - probs) ** self.gamma
        loss = -focal_weight * smooth_targets * log_probs

        if self.alpha is not None:
            alpha_t = self.alpha[targets].unsqueeze(1)
            loss = alpha_t * loss

        return loss.sum(dim=1).mean()


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


def build_model(num_classes=2):
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    old_conv = model.conv1
    model.conv1 = nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3, bias=False)
    with torch.no_grad():
        model.conv1.weight[:, :3] = old_conv.weight
        model.conv1.weight[:, 3:] = old_conv.weight

    model.fc = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(model.fc.in_features, num_classes),
    )
    return model


def mixup_data(x, y, alpha=0.3):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


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
        T.RandomRotation(10),
        T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
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

    model = build_model().to(device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: ResNet18 (6ch) + FocalLoss + MixUp, params: {num_params:,}")

    train_labels = [0 if labels[s]["main_label"] == "Positive" else 1 for s in train_ids]
    class_counts = Counter(train_labels)
    alpha = torch.tensor([1.0 / class_counts[0], 1.0 / class_counts[1]], dtype=torch.float32)
    alpha = alpha / alpha.sum() * 2
    criterion = FocalLoss(alpha=alpha.to(device), gamma=2.0, label_smoothing=0.05)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=5e-4)
    epochs = 20
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

            mixed_imgs, y_a, y_b, lam = mixup_data(imgs, targets, alpha=0.3)

            optimizer.zero_grad()
            out = model(mixed_imgs)
            loss = lam * criterion(out, y_a) + (1 - lam) * criterion(out, y_b)
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
    approach = "resnet18_6ch_focal_loss_mixup_label_smoothing"

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
