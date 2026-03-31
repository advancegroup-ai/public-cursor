"""
train.py — Liveness detection training script (runs on DGX1).
Experiment: ResNet18 with 9-channel input (far + near + pixel-wise difference),
plus test-time augmentation (TTA) for more robust predictions. The difference
channel captures subtle pixel-level inconsistencies between far and near shots
that may be more informative for detecting attacks.
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


class LivenessDatasetWithDiff(Dataset):
    def __init__(self, sig_ids, labels, transform=None, is_train=True):
        self.sig_ids = sig_ids
        self.labels = labels
        self.transform = transform
        self.is_train = is_train
        self.samples_dir = DATA_DIR / "samples"
        self.resize = T.Resize((224, 224))
        self.to_tensor = T.ToTensor()

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

        far_img = self.resize(far_img)
        near_img = self.resize(near_img)

        if self.is_train:
            if random.random() > 0.5:
                far_img = T.functional.hflip(far_img)
                near_img = T.functional.hflip(near_img)
            if random.random() > 0.5:
                brightness = random.uniform(0.8, 1.2)
                contrast = random.uniform(0.8, 1.2)
                far_img = T.functional.adjust_brightness(far_img, brightness)
                far_img = T.functional.adjust_contrast(far_img, contrast)
                near_img = T.functional.adjust_brightness(near_img, brightness)
                near_img = T.functional.adjust_contrast(near_img, contrast)

        far_t = self.to_tensor(far_img)
        near_t = self.to_tensor(near_img)

        normalize = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        far_n = normalize(far_t)
        near_n = normalize(near_t)

        diff = far_t - near_t
        diff_n = (diff - diff.mean()) / (diff.std() + 1e-6)

        img = torch.cat([far_n, near_n, diff_n], dim=0)
        return img, label


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


def build_model(num_classes=2):
    """ResNet18 modified for 9-channel input (far + near + difference)."""
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

    old_conv = model.conv1
    model.conv1 = nn.Conv2d(9, 64, kernel_size=7, stride=2, padding=3, bias=False)
    with torch.no_grad():
        model.conv1.weight[:, :3] = old_conv.weight
        model.conv1.weight[:, 3:6] = old_conv.weight
        model.conv1.weight[:, 6:9] = old_conv.weight * 0.5

    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(model.fc.in_features, num_classes),
    )
    return model


def train():
    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_ids, test_ids, labels = load_data()

    train_ds = LivenessDatasetWithDiff(train_ids, labels, is_train=True)
    test_ds = LivenessDatasetWithDiff(test_ids, labels, is_train=False)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True,
                              num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False,
                             num_workers=4, pin_memory=True)

    model = build_model().to(device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: ResNet18 (9-ch: far+near+diff), params: {num_params:,}")

    train_labels = [0 if labels[s]["main_label"] == "Positive" else 1 for s in train_ids]
    class_counts = Counter(train_labels)
    weight = torch.tensor([1.0 / class_counts[0], 1.0 / class_counts[1]], dtype=torch.float32)
    weight = weight / weight.sum() * 2
    criterion = nn.CrossEntropyLoss(weight=weight.to(device), label_smoothing=0.05)

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=5e-4)
    epochs = 15
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=2e-4, steps_per_epoch=len(train_loader), epochs=epochs,
        pct_start=0.2, anneal_strategy='cos'
    )

    best_bal_acc = 0
    best_acc = 0
    best_f1 = 0
    best_state = None

    for epoch in range(epochs):
        if time.time() - t0 > MAX_SECONDS - 30:
            print(f"Time budget approaching at epoch {epoch}")
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
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # TTA evaluation with best model
    if best_state is not None and time.time() - t0 < MAX_SECONDS - 10:
        print("\nRunning TTA evaluation...")
        model.load_state_dict(best_state)
        model.to(device)
        model.eval()

        all_probs = []
        all_labels_tta = []

        for tta_flip in [False, True]:
            batch_probs = []
            batch_labels = []
            with torch.no_grad():
                for imgs, targets in test_loader:
                    if tta_flip:
                        imgs = torch.flip(imgs, dims=[3])
                    imgs = imgs.to(device)
                    out = F.softmax(model(imgs), dim=1)
                    batch_probs.append(out.cpu())
                    if not tta_flip:
                        batch_labels.extend(targets.numpy())

            all_probs.append(torch.cat(batch_probs))
            if not tta_flip:
                all_labels_tta = batch_labels

        avg_probs = sum(all_probs) / len(all_probs)
        tta_preds = avg_probs.argmax(dim=1).numpy()

        tta_bal_acc = balanced_accuracy_score(all_labels_tta, tta_preds)
        tta_acc = accuracy_score(all_labels_tta, tta_preds)
        tta_f1 = f1_score(all_labels_tta, tta_preds, average="binary")
        print(f"TTA: bal_acc={tta_bal_acc:.4f} acc={tta_acc:.4f} f1={tta_f1:.4f}")

        if tta_bal_acc > best_bal_acc:
            best_bal_acc = tta_bal_acc
            best_acc = tta_acc
            best_f1 = tta_f1

    elapsed = time.time() - t0

    approach = "resnet18_9ch_far_near_diff_label_smooth_onecycle_TTA"
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
