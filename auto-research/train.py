"""
train.py — Liveness: EfficientNet-B0 + focal loss, local-cached images for speed.
Copies sampled data to /tmp then trains from local SSD.
"""
import os, sys, json, time, random, math, shutil
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

NAS_DIR = Path("/mnt/nas/public2/simon/projects/auto_research/liveness-research/data")
LOCAL_DIR = Path("/tmp/liveness_data")
RESULTS_FILE = NAS_DIR / "last_result.json"
SEED = 42
MAX_SECONDS = 200

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

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()


class LivenessDataset(Dataset):
    def __init__(self, sig_ids, label_map, samples_dir, transform=None):
        self.sig_ids = sig_ids
        self.label_map = label_map
        self.transform = transform
        self.samples_dir = samples_dir

    def __len__(self):
        return len(self.sig_ids)

    def __getitem__(self, idx):
        sig = self.sig_ids[idx]
        label = self.label_map[sig]

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


def load_and_cache_data():
    """Load labels from jsonl, sample balanced subset, copy to local SSD."""
    t0 = time.time()

    label_map = {}
    positives = []
    negatives = []

    with open(str(NAS_DIR / "annotations_full.jsonl")) as f:
        for line in f:
            rec = json.loads(line)
            sig = rec["sig"]
            lbl = rec.get("label", "")
            if lbl == "Positive":
                positives.append(sig)
                label_map[sig] = 0
            elif lbl == "Negative":
                negatives.append(sig)
                label_map[sig] = 1

    print(f"Labels: {len(positives)} pos, {len(negatives)} neg ({time.time()-t0:.1f}s)")

    random.shuffle(positives)
    random.shuffle(negatives)

    MAX_PER_CLASS = 1500
    positives = positives[:MAX_PER_CLASS]
    negatives = negatives[:MAX_PER_CLASS]

    all_ids = positives + negatives
    random.shuffle(all_ids)

    local_samples = LOCAL_DIR / "samples"
    local_samples.mkdir(parents=True, exist_ok=True)

    print(f"Copying {len(all_ids)} samples to local SSD...")
    copied = 0
    failed = 0
    for sig in all_ids:
        src = NAS_DIR / "samples" / sig
        dst = local_samples / sig
        if dst.exists():
            copied += 1
            continue
        try:
            shutil.copytree(str(src), str(dst))
            copied += 1
        except Exception:
            failed += 1

    print(f"Copied: {copied}, failed: {failed} ({time.time()-t0:.1f}s)")

    valid_ids = [s for s in all_ids if (local_samples / s / "far.jpg").exists() and (local_samples / s / "near.jpg").exists()]
    print(f"Valid after copy: {len(valid_ids)}")

    split = int(len(valid_ids) * 0.8)
    train_ids = valid_ids[:split]
    test_ids = valid_ids[split:]

    dist_train = Counter(label_map[s] for s in train_ids)
    dist_test = Counter(label_map[s] for s in test_ids)
    print(f"Train: {len(train_ids)} ({dict(dist_train)})")
    print(f"Test:  {len(test_ids)} ({dict(dist_test)})")

    return train_ids, test_ids, label_map, local_samples


def build_model(num_classes=2):
    """EfficientNet-B0 pretrained, modified for 6-channel input."""
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)

    old_conv = model.features[0][0]
    new_conv = nn.Conv2d(6, old_conv.out_channels,
                         kernel_size=old_conv.kernel_size,
                         stride=old_conv.stride,
                         padding=old_conv.padding,
                         bias=False)
    with torch.no_grad():
        new_conv.weight[:, :3] = old_conv.weight
        new_conv.weight[:, 3:] = old_conv.weight
    model.features[0][0] = new_conv

    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, num_classes)
    )
    return model


def train():
    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_ids, test_ids, label_map, samples_dir = load_and_cache_data()
    print(f"Data ready: {time.time()-t0:.1f}s")

    transform_train = T.Compose([
        T.Resize((240, 240)),
        T.RandomCrop(224),
        T.RandomHorizontalFlip(),
        T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
        T.RandomRotation(10),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        T.RandomErasing(p=0.15),
    ])
    transform_test = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    train_ds = LivenessDataset(train_ids, label_map, samples_dir, transform_train)
    test_ds = LivenessDataset(test_ids, label_map, samples_dir, transform_test)

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

    model = build_model().to(device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: EfficientNet-B0 (6-ch), params: {num_params:,}")

    train_labels = [label_map[s] for s in train_ids]
    class_counts = Counter(train_labels)
    weight = torch.tensor([1.0 / class_counts[0], 1.0 / class_counts[1]], dtype=torch.float32)
    weight = weight / weight.sum() * 2
    criterion = FocalLoss(alpha=weight.to(device), gamma=2.0)

    epochs = 15
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)

    def cosine_warmup(epoch):
        if epoch < 2:
            return (epoch + 1) / 2
        progress = (epoch - 2) / max(1, epochs - 2)
        return max(1e-6 / 3e-4, 0.5 * (1 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, cosine_warmup)

    best_bal_acc = 0
    best_acc = 0
    best_f1 = 0

    for epoch in range(epochs):
        elapsed = time.time() - t0
        if elapsed > MAX_SECONDS:
            print(f"Time budget reached at epoch {epoch} ({elapsed:.1f}s)")
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
            total_loss += loss.item()

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
        avg_loss = total_loss / len(train_loader)

        print(f"Epoch {epoch+1}/{epochs}: loss={avg_loss:.4f} bal_acc={bal_acc:.4f} f1={f1:.4f} [{time.time()-t0:.0f}s]")

        if bal_acc > best_bal_acc:
            best_bal_acc = bal_acc
            best_acc = acc
            best_f1 = f1

    elapsed = time.time() - t0

    approach = "efficientnet_b0_6ch_focal_balanced1500_local_cache_cosine_warmup"
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
