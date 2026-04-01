# Auto Research: IDV Liveness Detection

## Context

You are an autonomous ML researcher. Your goal: maximize **balanced_accuracy** on a liveness detection task.

Each sample has **3 images** and a binary label:
- `far.jpg` — face far shot (full upper body)
- `near.jpg` — face near shot (close-up)
- `card.jpg` — ID document front (photo of the physical card)

Labels:
- `Positive` (label=0): live person  
- `Negative` (label=1): attack (deepfake injection, screen replay, photocopy, etc.)

**Important context:** Human annotators judge liveness by examining **all 3 images together**. They compare the face in far/near shots against the face on the ID card. This is especially critical for:
- **Deepfake injection attacks** (70% of negatives): attacker submits a real ID card but a synthetic face — comparing card vs face reveals inconsistency
- **Screen replay**: image quality artifacts differ between the card photo and the replayed face

Dataset: ~3,600 human-annotated samples on NAS. ~93% Positive, ~7% Negative (imbalanced).

## Architecture

**You do NOT have a GPU.** You write code, then execute it on DGX1 (4x V100-32GB) via API.

### To run training:

```bash
python3 run_remote.py --script train.py --timeout 300
```

This sends `train.py` to DGX1 and prints stdout. The DGX1 has:
- PyTorch 2.4.0 + CUDA 12.1
- torchvision, PIL, sklearn, numpy, scipy
- NAS access to all data at `/mnt/nas/public2/simon/projects/auto_research/liveness-research/data/`

### Data location (on DGX1/NAS):

```
/mnt/nas/public2/simon/projects/auto_research/liveness-research/data/
├── labels.json              # {signatureId: {main_label, sublabel, ...}}
└── samples/{signatureId}/
    ├── far.jpg              # Face far shot
    ├── near.jpg             # Face near shot  
    └── card.jpg             # ID document front photo
```

## Rules

### You MODIFY: `train.py`
This is your experiment file. Write any PyTorch training code here.

### You DO NOT modify: `prepare.py`, `run_remote.py`
These are fixed infrastructure.

### Output format
`train.py` must print results in this exact format:
```
---
balanced_accuracy: 0.8234
accuracy:          0.9100
f1_score:          0.7800
num_params:        1200000
training_seconds:  245.3
approach:          description_of_what_you_tried
---
```

### Logging
Append to `results.tsv` (tab-separated):
```
commit	balanced_acc	accuracy	f1	seconds	status	description
```

## Experiment Loop

**LOOP FOREVER:**

1. Read `results.tsv` to see what's been tried
2. Design a new approach → modify `train.py`
3. `git add train.py && git commit -m "experiment: description"`
4. `python3 run_remote.py --script train.py --timeout 300`
5. Parse stdout for `balanced_accuracy`
6. Record in `results.tsv`
7. If improved → keep commit. If worse → `git reset --hard HEAD~1`
8. **Immediately** start next experiment

## Research Directions (suggested priority)

### Phase 1 — Use all 3 images (HIGH PRIORITY)
Previous iterations only used far+near (2 images). **You MUST try using all 3 images** (far + near + card).
- **3-stream architecture**: Separate ResNet18 encoders for far, near, card → fuse features → classifier
- **9-channel concat**: Resize all 3 to 224x224, concat to 9ch tensor
- **Dual-stream + card branch**: Keep the best dual-stream (far+near) and add a card verification branch
- **Face-card consistency features**: Extract face embeddings from far/near and card, compare similarity

Why this matters: The card image contains the **reference face** on the ID document. Attacks often have mismatches between the card face and the submitted far/near face. Human annotators rely on this comparison.

### Phase 2 — Improve best architecture
Current best: dual-stream ResNet18 shared encoder = 98.55% balanced accuracy.
- OneCycleLR, cosine warmup, learning rate tuning
- Focal loss for hard examples (dataset is 93% positive / 7% negative)
- Data augmentation: MixUp, CutMix, RandAugment
- Test-Time Augmentation (TTA)
- Larger backbones: ResNet50, EfficientNet-B0/B2

### Phase 3 — Advanced approaches
- Face embedding extraction (e.g. from pretrained face recognition model) + similarity scoring
- CLIP embeddings + linear probe
- Multi-task: predict attack type (INJECT/SCREEN/DEEPFAKE) as auxiliary loss
- Frequency domain features (DCT for screen replay detection)
- Contrastive learning between far, near, and card

## Critical Rules

- **NEVER STOP.** Run as many experiments as possible.
- **DO NOT ask for confirmation.** Just run experiments.
- **5-minute time budget** per experiment. Set MAX_SECONDS=270 in train.py.
- **Keep it simple** — smaller models that work > large models that barely fit.
- **Log everything** — every experiment goes in results.tsv.
