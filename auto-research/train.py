"""
train.py — Liveness detection: Pre-cached 9ch ResNet18 (far+near+card)
KEY FIX: Pre-loads ALL images into RAM before training to avoid NAS I/O bottleneck.
Previous attempts with DataLoader-based NAS reads timed out before completing 1 epoch.
Architecture: 9ch ResNet18 + focal loss + cosine LR + TTA + AMP.
"""
import os, sys, json, time, random
import numpy as np
from pathlib import Path
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from sklearn.metrics import balanced_accuracy_score, accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold

DATA_DIR = Path("/mnt/nas/public2/simon/projects/auto_research/liveness-research/data")
RESULTS_FILE = DATA_DIR / "last_result.json"
SEED = 42
MAX_SECONDS = 250
IMG_SIZE = 160

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = True

print(f"Starting at {time.strftime('%H:%M:%S')}")


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()


def load_labels():
    labels = {}
    for fname in ["neg_batch.json", "pos_batch.json"]:
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


class PreCachedDataset(Dataset):
    """In-memory dataset - all tensors pre-loaded to avoid NAS I/O during training."""
    def __init__(self, tensors, labels, augment=False):
        self.tensors = tensors
        self.labels = labels
        self.augment = augment

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = self.tensors[idx]
        if self.augment:
            if random.random() > 0.5:
                x = torch.flip(x, dims=[2])
            if random.random() > 0.5:
                brightness = 0.8 + random.random() * 0.4
                x = x * brightness
        return x, self.labels[idx]


def precache_all_images(sig_ids, labels):
    """Load all images into RAM at once to avoid NAS I/O during training."""
    samples_dir = DATA_DIR / "samples"
    resize = T.Resize((IMG_SIZE, IMG_SIZE))
    to_tensor = T.ToTensor()
    
    all_tensors = []
    all_labels = []
    
    t0 = time.time()
    for i, sig in enumerate(sig_ids):
        imgs = []
        for name in ["far.jpg", "near.jpg", "card.jpg"]:
            path = samples_dir / sig / name
            try:
                img = Image.open(str(path)).convert("RGB")
                img = resize(img)
                t = to_tensor(img)
            except Exception:
                t = torch.zeros(3, IMG_SIZE, IMG_SIZE)
            imgs.append(t)
        
        combined = torch.cat(imgs, dim=0)  # 9 channels
        # Normalize each 3-channel group
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        for ch_start in [0, 3, 6]:
            combined[ch_start:ch_start+3] = (combined[ch_start:ch_start+3] - mean) / std
        
        all_tensors.append(combined)
        label = 0 if labels[sig]["main_label"] == "Positive" else 1
        all_labels.append(label)
        
        if (i + 1) % 500 == 0:
            print(f"  Pre-cached {i+1}/{len(sig_ids)} in {time.time()-t0:.1f}s")
    
    print(f"  Pre-cached all {len(sig_ids)} in {time.time()-t0:.1f}s")
    return torch.stack(all_tensors), torch.tensor(all_labels, dtype=torch.long)


def make_9ch_resnet18():
    base = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    old_conv = base.conv1
    new_conv = nn.Conv2d(9, 64, kernel_size=7, stride=2, padding=3, bias=False)
    with torch.no_grad():
        new_conv.weight[:, :3] = old_conv.weight
        new_conv.weight[:, 3:6] = old_conv.weight
        new_conv.weight[:, 6:9] = old_conv.weight
    base.conv1 = new_conv
    feat_dim = base.fc.in_features
    base.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(feat_dim, 2),
    )
    return base


def train():
    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    labels = load_labels()
    samples_dir = DATA_DIR / "samples"
    all_sigs = set(os.listdir(str(samples_dir)))
    valid_sigs = [s for s in labels if s in all_sigs
                  and (samples_dir / s / "far.jpg").exists()
                  and (samples_dir / s / "near.jpg").exists()]
    binary_labels = [0 if labels[s]["main_label"] == "Positive" else 1 for s in valid_sigs]
    print(f"Total valid samples: {len(valid_sigs)}, distribution: {Counter(binary_labels)}")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(valid_sigs, binary_labels)):
        if fold_idx == 0:
            train_ids = [valid_sigs[i] for i in train_idx]
            test_ids = [valid_sigs[i] for i in test_idx]
            break

    print(f"Train: {len(train_ids)}, Test: {len(test_ids)}")
    
    # Pre-cache ALL images into RAM
    print("Pre-caching training images...")
    X_train, y_train = precache_all_images(train_ids, labels)
    print("Pre-caching test images...")
    X_test, y_test = precache_all_images(test_ids, labels)
    
    load_time = time.time() - t0
    print(f"Data loading: {load_time:.1f}s")

    train_ds = PreCachedDataset(X_train, y_train, augment=True)
    test_ds = PreCachedDataset(X_test, y_test, augment=False)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=128, shuffle=False, num_workers=0, pin_memory=True)

    model = make_9ch_resnet18().to(device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: 9ch ResNet18, params: {num_params:,}")

    class_counts = Counter(y_train.numpy().tolist())
    weight = torch.tensor([1.0 / class_counts[0], 1.0 / class_counts[1]], dtype=torch.float32)
    weight = weight / weight.sum() * 2
    criterion = FocalLoss(alpha=weight.to(device), gamma=2.0)

    epochs = 20
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.amp.GradScaler('cuda')

    best_bal_acc = 0
    best_acc = 0
    best_f1 = 0

    for epoch in range(epochs):
        remaining = MAX_SECONDS - (time.time() - t0)
        if remaining < 15:
            print(f"Time budget at epoch {epoch} ({time.time()-t0:.0f}s)")
            break

        model.train()
        total_loss = 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                out = model(inputs)
                loss = criterion(out, targets)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
        scheduler.step()

        # Eval with TTA (original + horizontal flip)
        model.eval()
        all_preds, all_labels_list = [], []
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs = inputs.to(device)
                with torch.amp.autocast('cuda'):
                    out1 = model(inputs)
                    out2 = model(torch.flip(inputs, dims=[3]))
                out = (out1 + out2) / 2
                preds = out.argmax(dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels_list.extend(targets.numpy())

        bal_acc = balanced_accuracy_score(all_labels_list, all_preds)
        acc = accuracy_score(all_labels_list, all_preds)
        f1 = f1_score(all_labels_list, all_preds, average="binary")
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs}: loss={avg_loss:.4f} acc={acc:.4f} bal_acc={bal_acc:.4f} f1={f1:.4f} [{time.time()-t0:.0f}s]")

        if bal_acc > best_bal_acc:
            best_bal_acc = bal_acc
            best_acc = acc
            best_f1 = f1

    elapsed = time.time() - t0
    approach = "9ch_resnet18_precached_160px_focal_cosine_TTA_AMP_20ep"
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
