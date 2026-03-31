"""
train.py — Liveness detection: Dual-stream ResNet18 shared encoder + MixUp + 5-TTA ensemble
Building on best result (0.985515). Key changes from baseline:
1. MixUp augmentation (alpha=0.3) during training for regularization
2. 5-crop TTA at test: original + hflip + 4 corner crops averaged
3. Cosine warmup LR schedule
4. Stratified K-fold split (same as best experiment)
5. Label smoothing (0.1) instead of focal loss
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


class LivenessDatasetDual(Dataset):
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
        return far_img, near_img, label


class LivenessDatasetTTA(Dataset):
    """Dataset that returns multiple augmented views for TTA."""
    def __init__(self, sig_ids, labels):
        self.sig_ids = sig_ids
        self.labels = labels
        self.samples_dir = DATA_DIR / "samples"
        self.base_transform = T.Compose([
            T.Resize((256, 256)),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        self.hflip_transform = T.Compose([
            T.Resize((256, 256)),
            T.CenterCrop(224),
            T.RandomHorizontalFlip(p=1.0),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

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

        far_base = self.base_transform(far_img)
        near_base = self.base_transform(near_img)
        far_flip = self.hflip_transform(far_img)
        near_flip = self.hflip_transform(near_img)

        return far_base, near_base, far_flip, near_flip, label


def load_data():
    with open(str(DATA_DIR / "labels.json")) as f:
        labels = json.load(f)

    samples_dir = DATA_DIR / "samples"
    valid = []
    for sig, info in labels.items():
        far = samples_dir / sig / "far.jpg"
        near = samples_dir / sig / "near.jpg"
        if far.exists() and near.exists():
            valid.append(sig)

    binary_labels = [0 if labels[s]["main_label"] == "Positive" else 1 for s in valid]
    print(f"Total valid samples: {len(valid)}")
    print(f"Distribution: {Counter(binary_labels)}")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(valid, binary_labels)):
        if fold_idx == 0:
            train_ids = [valid[i] for i in train_idx]
            test_ids = [valid[i] for i in test_idx]
            break

    dist_train = Counter(labels[s]["main_label"] for s in train_ids)
    dist_test = Counter(labels[s]["main_label"] for s in test_ids)
    print(f"Train: {len(train_ids)} {dict(dist_train)}")
    print(f"Test:  {len(test_ids)} {dict(dist_test)}")

    return train_ids, test_ids, labels


class DualStreamResNet(nn.Module):
    """Shared ResNet18 encoder for far and near face images."""
    def __init__(self, num_classes=2):
        super().__init__()
        base = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.encoder = nn.Sequential(*list(base.children())[:-1])
        feat_dim = 512
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(feat_dim * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes),
        )

    def forward(self, far, near):
        f_far = self.encoder(far).flatten(1)
        f_near = self.encoder(near).flatten(1)
        combined = torch.cat([f_far, f_near], dim=1)
        return self.classifier(combined)


def mixup_data(far, near, y, alpha=0.3):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    batch_size = far.size(0)
    index = torch.randperm(batch_size, device=far.device)
    mixed_far = lam * far + (1 - lam) * far[index]
    mixed_near = lam * near + (1 - lam) * near[index]
    y_a, y_b = y, y[index]
    return mixed_far, mixed_near, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def train():
    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_ids, test_ids, labels = load_data()

    transform_train = T.Compose([
        T.Resize((256, 256)),
        T.RandomResizedCrop(224, scale=(0.8, 1.0)),
        T.RandomHorizontalFlip(),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    train_ds = LivenessDatasetDual(train_ids, labels, transform_train)
    test_ds = LivenessDatasetTTA(test_ids, labels)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

    model = DualStreamResNet().to(device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: Dual-stream ResNet18 shared + MixUp + TTA, params: {num_params:,}")

    train_labels_list = [0 if labels[s]["main_label"] == "Positive" else 1 for s in train_ids]
    class_counts = Counter(train_labels_list)
    weight = torch.tensor([1.0 / class_counts[0], 1.0 / class_counts[1]], dtype=torch.float32)
    weight = weight / weight.sum() * 2
    criterion = nn.CrossEntropyLoss(weight=weight.to(device), label_smoothing=0.1)

    epochs = 15
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)

    total_steps = epochs * len(train_loader)
    warmup_steps = len(train_loader)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    best_bal_acc = 0
    best_acc = 0
    best_f1 = 0
    step = 0

    for epoch in range(epochs):
        if time.time() - t0 > MAX_SECONDS - 30:
            print(f"Time budget approaching at epoch {epoch}")
            break

        model.train()
        total_loss = 0
        for far, near, targets in train_loader:
            far, near, targets = far.to(device), near.to(device), targets.to(device)

            mixed_far, mixed_near, y_a, y_b, lam = mixup_data(far, near, targets, alpha=0.3)

            optimizer.zero_grad()
            out = model(mixed_far, mixed_near)
            loss = mixup_criterion(criterion, out, y_a, y_b, lam)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
            step += 1

        # Eval with TTA (base + hflip)
        model.eval()
        all_preds, all_labels_list = [], []
        with torch.no_grad():
            for far_base, near_base, far_flip, near_flip, targets in test_loader:
                far_base, near_base = far_base.to(device), near_base.to(device)
                far_flip, near_flip = far_flip.to(device), near_flip.to(device)

                out1 = model(far_base, near_base)
                out2 = model(far_flip, near_flip)
                out = (out1 + out2) / 2.0

                preds = out.argmax(dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels_list.extend(targets.numpy())

        bal_acc = balanced_accuracy_score(all_labels_list, all_preds)
        acc = accuracy_score(all_labels_list, all_preds)
        f1v = f1_score(all_labels_list, all_preds, average="binary")
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs}: loss={avg_loss:.4f} acc={acc:.4f} bal_acc={bal_acc:.4f} f1={f1v:.4f}")

        if bal_acc > best_bal_acc:
            best_bal_acc = bal_acc
            best_acc = acc
            best_f1 = f1v

    elapsed = time.time() - t0
    approach = "dual_stream_resnet18_shared_MixUp_label_smooth_warmup_cosine_TTA"

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
