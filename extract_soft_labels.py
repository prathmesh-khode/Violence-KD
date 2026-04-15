"""
extract_soft_labels.py
──────────────────────
Runs your ALREADY TRAINED ViolenceModel (teacher) over all training clips
and saves softmax probabilities at temperature T=4 to disk.

These soft labels are then loaded by train_student.py for distillation.

Saves:
  outputs/soft_labels_T4.pt   ← dict { video_path (str) → [p0, p1] list }

Usage:
  python extract_soft_labels.py

Run time : ~15-30 min depending on dataset size.
Run ONCE.  No need to rerun unless you retrain the teacher.
"""

import os
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import (
    TRAIN_PATH, OUTPUT_DIR,
    BEST_MODEL_PATH, SOFT_LABELS_PATH,
    NUM_FRAMES, IMG_SIZE,
    DISTILL_TEMPERATURE, NUM_WORKERS,
)
from models import ViolenceModel
from dataset import VideoDataset


def load_teacher(path, device):
    """Load ViolenceModel from checkpoint — handles both raw and dict formats."""
    model = ViolenceModel().to(device)
    # Disable freeze so all weights load cleanly
    for p in model.parameters():
        p.requires_grad = False

    raw = torch.load(path, map_location=device, weights_only=False)

    if isinstance(raw, dict):
        if "model_state" in raw:
            model.load_state_dict(raw["model_state"])
            print(f"[INFO] Loaded checkpoint dict  (epoch {raw.get('epoch', '?')}, "
                  f"best_acc={raw.get('best_acc', '?')})")
        elif "state_dict" in raw:
            model.load_state_dict(raw["state_dict"])
            print("[INFO] Loaded state_dict key")
        else:
            # The dict IS the state dict (most common case from your train.py)
            model.load_state_dict(raw)
            print("[INFO] Loaded raw state dict")
    else:
        raise ValueError(f"Unexpected checkpoint format: {type(raw)}")

    model.eval()
    return model


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\n" + "="*55)
    print("  EXTRACTING SOFT LABELS FROM TEACHER")
    print("="*55)
    print(f"  Device      : {device}")
    print(f"  Temperature : {DISTILL_TEMPERATURE}")
    print(f"  Checkpoint  : {BEST_MODEL_PATH}")
    print(f"  Train data  : {TRAIN_PATH}")

    if not os.path.exists(BEST_MODEL_PATH):
        raise FileNotFoundError(
            f"\n[ERROR] Checkpoint not found: {BEST_MODEL_PATH}\n"
            "Make sure outputs/best_model.pt exists."
        )

    # ── Load teacher ──────────────────────────────────────────
    model = load_teacher(BEST_MODEL_PATH, device)

    # ── Dataset (val mode = NO augmentation = deterministic) ──
    dataset = VideoDataset(
        TRAIN_PATH,
        mode="val",           # deterministic — no random augmentation
        img_size=IMG_SIZE,
        num_frames=NUM_FRAMES,
    )

    loader = DataLoader(
        dataset,
        batch_size=16,        # safe for most GPUs
        shuffle=False,        # MUST be False to keep index mapping correct
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    print(f"\n[INFO] Extracting soft labels for {len(dataset)} clips...\n")

    soft_labels = {}
    T = DISTILL_TEMPERATURE

    with torch.no_grad():
        for batch_idx, (clips, _) in enumerate(tqdm(loader, desc="Extracting")):
            clips = clips.to(device)
            logits = model(clips)

            # Temperature scaling → softer probability distribution
            soft_probs = F.softmax(logits / T, dim=1).cpu().numpy()  # (B, 2)

            start = batch_idx * loader.batch_size
            for i, prob in enumerate(soft_probs):
                idx = start + i
                if idx >= len(dataset):
                    break
                video_path, _ = dataset.samples[idx]
                soft_labels[video_path] = prob.tolist()

    print(f"\n[INFO] Extracted soft labels for {len(soft_labels)} clips")

    # ── Save ──────────────────────────────────────────────────
    torch.save(soft_labels, SOFT_LABELS_PATH)
    print(f"[INFO] Saved → {SOFT_LABELS_PATH}")

    # ── Sanity check ──────────────────────────────────────────
    sample_path  = list(soft_labels.keys())[0]
    sample_probs = soft_labels[sample_path]
    print(f"\n[CHECK] Sample entry:")
    print(f"  Video           : {os.path.basename(sample_path)}")
    print(f"  p(non-violence) : {sample_probs[0]:.4f}")
    print(f"  p(violence)     : {sample_probs[1]:.4f}")
    print(f"  Sum             : {sum(sample_probs):.6f}  (must be ~1.0)")

    v_probs = np.array([v[1] for v in soft_labels.values()])
    print(f"\n[STATS] Teacher confidence on training set (p_violence):")
    print(f"  Mean : {v_probs.mean():.4f}")
    print(f"  Std  : {v_probs.std():.4f}")
    print(f"  Min  : {v_probs.min():.4f}")
    print(f"  Max  : {v_probs.max():.4f}")
    high = (v_probs > 0.8).mean() * 100
    low  = (v_probs < 0.2).mean() * 100
    print(f"  >0.8 (confident violence)    : {high:.1f}%")
    print(f"  <0.2 (confident non-violence): {low:.1f}%")
    ambiguous = ((v_probs >= 0.2) & (v_probs <= 0.8)).mean() * 100
    print(f"  0.2-0.8 (ambiguous)          : {ambiguous:.1f}%")

    print(f"\n✓ Done. Next step: python train_student.py")


if __name__ == "__main__":
    main()
