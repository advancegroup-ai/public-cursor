"""
train.py — Liveness detection training script (runs on DGX1).
Experiment: ResNet50 6ch with multi-scale feature pyramid + gradient 
accumulation + cosine warmup + TTA at inference.
Key ideas:
- ResNet50 for deeper features vs ResNet18
- Extract features from multiple layers (layer2, layer3, layer4) and combine
- Gradient accumulation (4 steps) for effective batch size 128
- Warmup + cosine LR schedule
- Test-time augmentation (original + horizontal flip) for more robust predictions
Previous best on DGX1: 0.976659 (ResNet18 6ch baseline).
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
RESULTS_FILE = Path("/mnt/nas/public2/simon/projects/auto_research/liveness-research/data/last_result.json")
SEED = 42
MAX_SECONDS = 270

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


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


class MultiScaleResNet50(nn.Module):
    """ResNet50 with multi-scale feature pyramid network for 6-channel input."""
    def __init__(self, num_classes=2):
        super().__init__()
        base = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

        old_conv = base.conv1
        base.conv1 = nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            base.conv1.weight[:, :3] = old_conv.weight
            base.conv1.weight[:, 3:] = old_conv.weight

        self.stem = nn.Sequential(base.conv1, base.bn1, base.relu, base.maxpool)
        self.layer1 = base.layer1  # 256
        self.layer2 = base.layer2  # 512
        self.layer3 = base.layer3  # 1024
        self.layer4 = base.layer4  # 2048

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.proj2 = nn.Linear(512, 256)
        self.proj3 = nn.Linear(1024, 256)
        self.proj4 = nn.Linear(2048, 256)

        self.classifier = nn.Sequential(
            nn.LayerNorm(256 * 3),
            nn.Dropout(0.4),
            nn.Linear(256 * 3, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.stem(x)
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        f2 = self.pool(x2).flatten(1)
        f3 = self.pool(x3).flatten(1)
        f4 = self.pool(x4).flatten(1)

        p2 = F.gelu(self.proj2(f2))
        p3 = F.gelu(self.proj3(f3))
        p4 = F.gelu(self.proj4(f4))

        multi = torch.cat([p2, p3, p4], dim=1)
        return self.classifier(multi)


def load_data():
    with open(str(DATA_DIR / "labels.json")) as f:
        labels = json.load(f)

    valid = []
    valid_labels = []
    for sig, info in labels.items():
        far = DATA_DIR / "samples" / sig / "far.jpg"
        near = DATA_DIR / "samples" / sig / "near.jpg"
        if far.exists() and near.exists():
            valid.append(sig)
            valid_labels.append(0 if info["main_label"] == "Positive" else 1)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    train_idx, test_idx = next(skf.split(valid, valid_labels))
    train_ids = [valid[i] for i in train_idx]
    test_ids = [valid[i] for i in test_idx]

    dist_train = Counter(valid_labels[i] for i in train_idx)
    dist_test = Counter(valid_labels[i] for i in test_idx)
    print(f"Train: {len(train_ids)} {dict(dist_train)}")
    print(f"Test:  {len(test_ids)} {dict(dist_test)}")

    return train_ids, test_ids, labels


def train():
    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_ids, test_ids, labels = load_data()

    transform_train = T.Compose([
        T.Resize((256, 256)),
        T.RandomCrop(224),
        T.RandomHorizontalFlip(),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    transform_test = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    transform_tta_flip = T.Compose([
        T.Resize((224, 224)),
        T.RandomHorizontalFlip(p=1.0),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    train_ds = LivenessDataset(train_ids, labels, transform_train)
    test_ds = LivenessDataset(test_ids, labels, transform_test)
    test_ds_flip = LivenessDataset(test_ids, labels, transform_tta_flip)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    test_loader = DataLoader(test_ds, batch_size=48, shuffle=False, num_workers=4, pin_memory=True)
    test_loader_flip = DataLoader(test_ds_flip, batch_size=48, shuffle=False, num_workers=4, pin_memory=True)

    model = MultiScaleResNet50().to(device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: Multi-Scale ResNet50 (6ch), params: {num_params:,}")

    train_labels = [0 if labels[s]["main_label"] == "Positive" else 1 for s in train_ids]
    class_counts = Counter(train_labels)
    weight = torch.tensor([1.0 / class_counts[0], 1.0 / class_counts[1]], dtype=torch.float32)
    weight = weight / weight.sum() * 2
    criterion = nn.CrossEntropyLoss(weight=weight.to(device), label_smoothing=0.1)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=5e-4)
    epochs = 15
    accum_steps = 4

    def lr_lambda(epoch):
        warmup = 2
        if epoch < warmup:
            return (epoch + 1) / warmup
        progress = (epoch - warmup) / max(1, epochs - warmup)
        return max(1e-6 / 1e-4, 0.5 * (1 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    best_bal_acc = 0
    best_acc = 0
    best_f1 = 0
    best_state = None

    for epoch in range(epochs):
        if time.time() - t0 > MAX_SECONDS:
            print(f"Time budget reached at epoch {epoch}")
            break

        model.train()
        total_loss = 0
        optimizer.zero_grad()
        for step, (imgs, targets) in enumerate(train_loader):
            imgs, targets = imgs.to(device), targets.to(device)
            out = model(imgs)
            loss = criterion(out, targets) / accum_steps
            loss.backward()

            if (step + 1) % accum_steps == 0 or (step + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()

            total_loss += loss.item() * accum_steps

        scheduler.step()

        model.eval()
        all_probs = []
        all_probs_flip = []
        all_labels_list = []

        with torch.no_grad():
            for imgs, targets in test_loader:
                imgs = imgs.to(device)
                out = F.softmax(model(imgs), dim=1)
                all_probs.append(out.cpu())
                all_labels_list.extend(targets.numpy())

            for imgs, targets in test_loader_flip:
                imgs = imgs.to(device)
                out = F.softmax(model(imgs), dim=1)
                all_probs_flip.append(out.cpu())

        all_probs = torch.cat(all_probs, dim=0)
        all_probs_flip = torch.cat(all_probs_flip, dim=0)
        tta_probs = (all_probs + all_probs_flip) / 2.0
        tta_preds = tta_probs.argmax(dim=1).numpy()
        no_tta_preds = all_probs.argmax(dim=1).numpy()

        bal_acc = balanced_accuracy_score(all_labels_list, tta_preds)
        bal_acc_no_tta = balanced_accuracy_score(all_labels_list, no_tta_preds)
        acc = accuracy_score(all_labels_list, tta_preds)
        f1 = f1_score(all_labels_list, tta_preds, average="binary")
        avg_loss = total_loss / len(train_loader)

        print(f"Epoch {epoch+1}/{epochs}: loss={avg_loss:.4f} acc={acc:.4f} bal_acc={bal_acc:.4f}(TTA)/{bal_acc_no_tta:.4f}(no-TTA) f1={f1:.4f}")

        if bal_acc > best_bal_acc:
            best_bal_acc = bal_acc
            best_acc = acc
            best_f1 = f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    elapsed = time.time() - t0
    approach = "resnet50_multiscale_fpn_6ch_grad_accum4x_warmup_cosine_tta"

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
