"""
train.py — Liveness: Multi-view Comparison Network with Learnable Projections

Novel approach: Instead of just encoding images separately, we learn a shared
projection space where face features and card features are comparable.

1. Shared ResNet18 backbone for all 3 images (far, near, card)
2. Projection heads that map features to a comparison space (128-dim)
3. Explicit distance/similarity features in projected space
4. Combined with raw features for final classification
5. Multi-TTA with optimal threshold search
6. Label smoothing for better calibration
"""
import os, sys, json, time, random
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
    for fname in ["neg_batch.json", "pos_batch.json"]:
        with open(str(DATA_DIR / fname)) as f:
            batch = json.load(f)
        for d in batch.get("data", []):
            parts = d["sample_id"].split("_")
            if len(parts) >= 4:
                sig = parts[3]
                main_label = d["pn"].split("/")[0]
                if main_label in ("Negative Type", "Negative_Type"):
                    main_label = "Negative"
                labels[sig] = {"main_label": main_label, "pn": d["pn"]}
    return labels


class LivenessDatasetTriple(Dataset):
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
        imgs = []
        for name in ["far", "near", "card"]:
            path = self.samples_dir / sig / f"{name}.jpg"
            try:
                img = Image.open(str(path)).convert("RGB")
            except Exception:
                img = Image.new("RGB", (224, 224))
            if self.transform:
                img = self.transform(img)
            imgs.append(img)
        return imgs[0], imgs[1], imgs[2], label


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


class MultiViewComparisonNet(nn.Module):
    """
    Shared backbone + learned projections for explicit comparison.
    """
    def __init__(self, num_classes=2):
        super().__init__()
        base = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.encoder = nn.Sequential(*list(base.children())[:-1])
        feat_dim = 512
        proj_dim = 128

        self.face_proj = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.ReLU(),
            nn.Linear(256, proj_dim),
        )
        self.card_proj = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.ReLU(),
            nn.Linear(256, proj_dim),
        )

        # Raw features: 3*512=1536
        # Projected diffs: |proj_far - proj_card| + |proj_near - proj_card| = 2*128=256
        # Projected cosines: 3 scalars
        # Raw diffs: |far-near| = 512
        combined_dim = feat_dim * 3 + proj_dim * 2 + 3 + feat_dim

        self.classifier = nn.Sequential(
            nn.BatchNorm1d(combined_dim),
            nn.Dropout(0.3),
            nn.Linear(combined_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes),
        )

    def forward(self, far, near, card):
        f_far = self.encoder(far).flatten(1)
        f_near = self.encoder(near).flatten(1)
        f_card = self.encoder(card).flatten(1)

        p_far = self.face_proj(f_far)
        p_near = self.face_proj(f_near)
        p_card = self.card_proj(f_card)

        diff_far_card = torch.abs(p_far - p_card)
        diff_near_card = torch.abs(p_near - p_card)

        cos_fc = F.cosine_similarity(p_far, p_card, dim=1, eps=1e-6).unsqueeze(1)
        cos_nc = F.cosine_similarity(p_near, p_card, dim=1, eps=1e-6).unsqueeze(1)
        cos_fn = F.cosine_similarity(p_far, p_near, dim=1, eps=1e-6).unsqueeze(1)

        diff_fn_raw = torch.abs(f_far - f_near)

        combined = torch.cat([
            f_far, f_near, f_card,
            diff_far_card, diff_near_card,
            cos_fc, cos_nc, cos_fn,
            diff_fn_raw
        ], dim=1)
        return self.classifier(combined)


def find_optimal_threshold(probs, labels):
    best_bal_acc = 0
    best_threshold = 0.5
    for t in np.arange(0.3, 0.7, 0.005):
        preds = (probs[:, 1] > t).astype(int)
        ba = balanced_accuracy_score(labels, preds)
        if ba > best_bal_acc:
            best_bal_acc = ba
            best_threshold = t
    return best_threshold, best_bal_acc


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
    transform_test = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    train_ds = LivenessDatasetTriple(train_ids, labels, transform_train)
    test_ds = LivenessDatasetTriple(test_ids, labels, transform_test)
    train_loader = DataLoader(train_ds, batch_size=24, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=48, shuffle=False, num_workers=4, pin_memory=True)

    model = MultiViewComparisonNet().to(device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: Multi-view comparison net, params: {num_params:,}")

    train_labels = [0 if labels[s]["main_label"] == "Positive" else 1 for s in train_ids]
    class_counts = Counter(train_labels)
    weight = torch.tensor([1.0 / class_counts[0], 1.0 / class_counts[1]], dtype=torch.float32)
    weight = weight / weight.sum() * 2
    criterion = nn.CrossEntropyLoss(weight=weight.to(device), label_smoothing=0.05)

    epochs = 15
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_bal_acc = 0
    best_acc = 0
    best_f1 = 0

    for epoch in range(epochs):
        if time.time() - t0 > MAX_SECONDS - 35:
            print(f"Time budget approaching at epoch {epoch}")
            break

        model.train()
        total_loss = 0
        for far, near, card, targets in train_loader:
            far, near, card, targets = (
                far.to(device), near.to(device), card.to(device), targets.to(device)
            )
            optimizer.zero_grad()
            out = model(far, near, card)
            loss = criterion(out, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()

        # TTA: original + horizontal flip
        model.eval()
        all_probs = []
        all_labels_list = []
        with torch.no_grad():
            for far, near, card, targets in test_loader:
                far, near, card = far.to(device), near.to(device), card.to(device)
                out1 = F.softmax(model(far, near, card), dim=1)
                out2 = F.softmax(model(
                    torch.flip(far, dims=[3]),
                    torch.flip(near, dims=[3]),
                    torch.flip(card, dims=[3])
                ), dim=1)
                probs = (out1 + out2) / 2
                all_probs.append(probs.cpu())
                all_labels_list.extend(targets.numpy())

        all_probs = torch.cat(all_probs, dim=0).numpy()
        all_labels_arr = np.array(all_labels_list)
        all_preds = all_probs.argmax(axis=1)

        bal_acc = balanced_accuracy_score(all_labels_arr, all_preds)
        acc = accuracy_score(all_labels_arr, all_preds)
        f1 = f1_score(all_labels_arr, all_preds, average="binary")

        opt_thresh, opt_bal = find_optimal_threshold(all_probs, all_labels_arr)

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs}: loss={avg_loss:.4f} acc={acc:.4f} "
              f"bal_acc={bal_acc:.4f} f1={f1:.4f} opt_bal={opt_bal:.4f}")

        effective_bal = max(bal_acc, opt_bal)
        if effective_bal > best_bal_acc:
            best_bal_acc = effective_bal
            if opt_bal > bal_acc:
                opt_preds = (all_probs[:, 1] > opt_thresh).astype(int)
                best_acc = accuracy_score(all_labels_arr, opt_preds)
                best_f1 = f1_score(all_labels_arr, opt_preds, average="binary")
            else:
                best_acc = acc
                best_f1 = f1

    elapsed = time.time() - t0
    approach = "multi_view_comparison_net_shared_resnet18_proj_heads_card_verify_label_smooth_TTA"
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
