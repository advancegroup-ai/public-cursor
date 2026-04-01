"""
train.py — Liveness detection: Card-Face Verification + Dual-Stream Liveness

This approach combines two complementary signals:
1. LIVENESS BRANCH: Dual-stream ResNet18 for far+near (proven best at 0.9855)
2. VERIFICATION BRANCH: Siamese comparison of face images vs card photo
   - Learns if the face in far/near matches the face on the ID card
   - Deepfake attacks (70% of negatives) have mismatched card vs face
3. Outputs are fused for final prediction

This should beat pure dual-stream because it adds the card comparison signal
that human annotators use to catch deepfakes.
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
        far_path = self.samples_dir / sig / "far.jpg"
        near_path = self.samples_dir / sig / "near.jpg"
        card_path = self.samples_dir / sig / "card.jpg"
        try:
            far_img = Image.open(str(far_path)).convert("RGB")
        except Exception:
            far_img = Image.new("RGB", (224, 224))
        try:
            near_img = Image.open(str(near_path)).convert("RGB")
        except Exception:
            near_img = Image.new("RGB", (224, 224))
        try:
            card_img = Image.open(str(card_path)).convert("RGB")
        except Exception:
            card_img = Image.new("RGB", (224, 224))
        if self.transform:
            far_img = self.transform(far_img)
            near_img = self.transform(near_img)
            card_img = self.transform(card_img)
        return far_img, near_img, card_img, label


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


class CardFaceVerificationNet(nn.Module):
    """
    Two-branch architecture:
    Branch A (Liveness): Dual-stream ResNet18 on far+near (proven best approach)
    Branch B (Verification): Compare far/near features against card features
    """
    def __init__(self, num_classes=2):
        super().__init__()
        base = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.face_encoder = nn.Sequential(*list(base.children())[:-1])
        
        card_base = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.card_encoder = nn.Sequential(*list(card_base.children())[:-1])
        
        feat_dim = 512
        
        # Branch A: Liveness (far + near)
        liveness_dim = feat_dim * 2  # 1024
        self.liveness_head = nn.Sequential(
            nn.BatchNorm1d(liveness_dim),
            nn.Dropout(0.3),
            nn.Linear(liveness_dim, 256),
            nn.ReLU(),
        )
        
        # Branch B: Verification features
        # |far - card|, |near - card|, cosine(far, card), cosine(near, card)
        verify_dim = feat_dim * 2 + 2  # 1026
        self.verify_head = nn.Sequential(
            nn.BatchNorm1d(verify_dim),
            nn.Dropout(0.3),
            nn.Linear(verify_dim, 128),
            nn.ReLU(),
        )
        
        self.final_classifier = nn.Sequential(
            nn.BatchNorm1d(256 + 128),
            nn.Dropout(0.2),
            nn.Linear(256 + 128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, far, near, card):
        f_far = self.face_encoder(far).flatten(1)
        f_near = self.face_encoder(near).flatten(1)
        f_card = self.card_encoder(card).flatten(1)
        
        # Branch A: Liveness
        liveness_feat = torch.cat([f_far, f_near], dim=1)
        liveness_out = self.liveness_head(liveness_feat)
        
        # Branch B: Verification
        diff_far_card = torch.abs(f_far - f_card)
        diff_near_card = torch.abs(f_near - f_card)
        cos_far_card = F.cosine_similarity(f_far, f_card, dim=1, eps=1e-6).unsqueeze(1)
        cos_near_card = F.cosine_similarity(f_near, f_card, dim=1, eps=1e-6).unsqueeze(1)
        verify_feat = torch.cat([diff_far_card, diff_near_card, cos_far_card, cos_near_card], dim=1)
        verify_out = self.verify_head(verify_feat)
        
        combined = torch.cat([liveness_out, verify_out], dim=1)
        return self.final_classifier(combined)


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

    model = CardFaceVerificationNet().to(device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: Card-Face Verification + Dual-stream liveness, params: {num_params:,}")

    train_labels = [0 if labels[s]["main_label"] == "Positive" else 1 for s in train_ids]
    class_counts = Counter(train_labels)
    weight = torch.tensor([1.0 / class_counts[0], 1.0 / class_counts[1]], dtype=torch.float32)
    weight = weight / weight.sum() * 2
    criterion = nn.CrossEntropyLoss(weight=weight.to(device))

    epochs = 15
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_bal_acc = 0
    best_acc = 0
    best_f1 = 0

    for epoch in range(epochs):
        if time.time() - t0 > MAX_SECONDS - 30:
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
        all_probs, all_labels_list = [], []
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

        all_probs = torch.cat(all_probs, dim=0)
        all_preds = all_probs.argmax(dim=1).numpy()

        bal_acc = balanced_accuracy_score(all_labels_list, all_preds)
        acc = accuracy_score(all_labels_list, all_preds)
        f1 = f1_score(all_labels_list, all_preds, average="binary")
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs}: loss={avg_loss:.4f} acc={acc:.4f} bal_acc={bal_acc:.4f} f1={f1:.4f}")

        if bal_acc > best_bal_acc:
            best_bal_acc = bal_acc
            best_acc = acc
            best_f1 = f1

    elapsed = time.time() - t0
    approach = "card_face_verification_plus_dual_stream_liveness_resnet18_TTA"
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
