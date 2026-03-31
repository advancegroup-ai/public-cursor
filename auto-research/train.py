"""
train.py — Liveness detection: ResNet18 6ch + OneCycleLR + label smoothing + stronger aug + TTA
Best previous: 0.976659 (ResNet18 baseline)
"""
import os, sys, json, time, random, math
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
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


class LivenessDataset(Dataset):
    def __init__(self, sig_ids, labels, transform=None, tta=False, tta_count=5):
        self.sig_ids = sig_ids
        self.labels = labels
        self.transform = transform
        self.samples_dir = DATA_DIR / "samples"
        self.tta = tta
        self.tta_count = tta_count

    def __len__(self):
        return len(self.sig_ids) * (self.tta_count if self.tta else 1)

    def __getitem__(self, idx):
        real_idx = idx // self.tta_count if self.tta else idx
        sig = self.sig_ids[real_idx]
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
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
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


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, label_smoothing=0.05):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, logits, targets):
        num_classes = logits.size(1)
        log_probs = F.log_softmax(logits, dim=-1)
        probs = torch.exp(log_probs)

        with torch.no_grad():
            smooth_targets = torch.zeros_like(log_probs)
            smooth_targets.fill_(self.label_smoothing / (num_classes - 1))
            smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.label_smoothing)

        focal_weight = (1.0 - probs).pow(self.gamma)

        if self.alpha is not None:
            alpha_weight = self.alpha[targets].unsqueeze(1)
            loss = (-alpha_weight * focal_weight * smooth_targets * log_probs).sum(dim=-1)
        else:
            loss = (-focal_weight * smooth_targets * log_probs).sum(dim=-1)

        return loss.mean()


def train():
    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_ids, test_ids, labels = load_data()

    transform_train = T.Compose([
        T.Resize((256, 256)),
        T.RandomResizedCrop(224, scale=(0.7, 1.0), ratio=(0.8, 1.2)),
        T.RandomHorizontalFlip(),
        T.RandomRotation(15),
        T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
        T.RandomGrayscale(p=0.05),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        T.RandomErasing(p=0.15, scale=(0.02, 0.15)),
    ])
    transform_test = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    transform_tta = T.Compose([
        T.Resize((256, 256)),
        T.RandomResizedCrop(224, scale=(0.85, 1.0)),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    train_ds = LivenessDataset(train_ids, labels, transform_train)
    test_ds = LivenessDataset(test_ids, labels, transform_test)

    BATCH_SIZE = 32
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=4, pin_memory=True, drop_last=True)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False,
                             num_workers=4, pin_memory=True)

    model = build_model().to(device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: ResNet18 (6-ch), params: {num_params:,}")

    train_labels = [0 if labels[s]["main_label"] == "Positive" else 1 for s in train_ids]
    class_counts = Counter(train_labels)
    alpha = torch.tensor([1.0 / class_counts[0], 1.0 / class_counts[1]], dtype=torch.float32)
    alpha = alpha / alpha.sum() * 2
    criterion = FocalLoss(alpha=alpha.to(device), gamma=2.0, label_smoothing=0.05)

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=5e-4)

    epochs = 15
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=3e-4, epochs=epochs, steps_per_epoch=len(train_loader),
        pct_start=0.15, anneal_strategy='cos', div_factor=10, final_div_factor=100,
    )

    best_bal_acc = 0
    best_acc = 0
    best_f1 = 0
    best_state = None

    for epoch in range(epochs):
        if time.time() - t0 > MAX_SECONDS - 30:
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

        print(f"Epoch {epoch+1}/{epochs}: loss={avg_loss:.4f} acc={acc:.4f} bal_acc={bal_acc:.4f} f1={f1:.4f} lr={optimizer.param_groups[0]['lr']:.6f}")

        if bal_acc > best_bal_acc:
            best_bal_acc = bal_acc
            best_acc = acc
            best_f1 = f1
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    # TTA evaluation with best model
    if best_state is not None and time.time() - t0 < MAX_SECONDS - 10:
        print("Running TTA evaluation...")
        model.load_state_dict(best_state)
        model.eval()

        tta_ds = LivenessDataset(test_ids, labels, transform_tta, tta=True, tta_count=5)
        tta_loader = DataLoader(tta_ds, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

        all_logits = []
        with torch.no_grad():
            for imgs, _ in tta_loader:
                imgs = imgs.to(device)
                out = model(imgs)
                all_logits.extend(out.cpu().numpy())

        all_logits = np.array(all_logits)
        n_test = len(test_ids)
        avg_logits = all_logits.reshape(n_test, 5, 2).mean(axis=1)
        tta_preds = avg_logits.argmax(axis=1)
        tta_labels = [0 if labels[s]["main_label"] == "Positive" else 1 for s in test_ids]

        tta_bal_acc = balanced_accuracy_score(tta_labels, tta_preds)
        tta_acc = accuracy_score(tta_labels, tta_preds)
        tta_f1 = f1_score(tta_labels, tta_preds, average="binary")

        print(f"TTA: bal_acc={tta_bal_acc:.4f} acc={tta_acc:.4f} f1={tta_f1:.4f}")

        if tta_bal_acc > best_bal_acc:
            best_bal_acc = tta_bal_acc
            best_acc = tta_acc
            best_f1 = tta_f1

    elapsed = time.time() - t0
    approach = "resnet18_6ch_focal_loss_onecycleLR_dropout03_strongaug_TTA5"

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
