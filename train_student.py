"""
train_student.py
────────────────
Trains the lightweight MobileNet3D student via knowledge distillation
from your trained ViolenceModel (R3D-18) teacher.

How distillation works here:
  1. Teacher already extracted soft labels (soft_labels_T4.pt).
     Each training clip has a probability distribution like [0.13, 0.87].
  2. Student trains on BOTH:
       Hard CE loss  → standard cross-entropy against ground truth labels
       Soft KD loss  → KL divergence against teacher's soft distributions
  3. Combined loss:
       L = alpha * CE(student_logits, hard_labels)
         + (1 - alpha) * T² * KL(student_soft || teacher_soft)

Saves:
  outputs/student_best.pt
  outputs/student_last.pt
  outputs/student_train_log.csv
  outputs/student_metrics.png
  outputs/student_* (all plots)

Usage:
  python train_student.py
"""

import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from config import (
    TRAIN_PATH, VAL_PATH, OUTPUT_DIR, SOFT_LABELS_PATH, STUDENT_MODEL_PATH,
    STUDENT_IMG_SIZE, STUDENT_NUM_FRAMES, STUDENT_BATCH_SIZE,
    STUDENT_EPOCHS, STUDENT_LR, STUDENT_PATIENCE,
    DISTILL_TEMPERATURE, DISTILL_ALPHA,
    NUM_WORKERS,
)
from models import MobileNet3D
from dataset import DistillDataset, VideoDataset
from utils import (
    ensure_dir, save_csv_log, plot_metrics, save_metrics_csv,
    full_evaluation, plot_teacher_vs_student,
)


# ══════════════════════════════════════════════════════════════
#  DISTILLATION LOSS
# ══════════════════════════════════════════════════════════════

class DistillationLoss(nn.Module):
    """
    Combined hard-label CE + soft-label KL divergence.

    Args:
        T     : temperature (must match extract_soft_labels.py)
        alpha : weight on hard CE loss; (1-alpha) on KD loss
    """
    def __init__(self, T=4.0, alpha=0.4):
        super().__init__()
        self.T     = T
        self.alpha = alpha
        self.ce    = nn.CrossEntropyLoss(label_smoothing=0.05)

    def forward(self, student_logits, hard_labels, soft_teacher_probs):
        # Hard-label cross-entropy
        loss_ce = self.ce(student_logits, hard_labels)

        # Soft-label KL divergence
        # student_logits scaled by T, log-softmax
        log_soft_student = F.log_softmax(student_logits / self.T, dim=1)
        # soft_teacher_probs are already probabilities (not logits)
        loss_kd = F.kl_div(
            log_soft_student,
            soft_teacher_probs,
            reduction="batchmean",
        ) * (self.T ** 2)

        return self.alpha * loss_ce + (1.0 - self.alpha) * loss_kd


# ══════════════════════════════════════════════════════════════
#  VALIDATION (plain CE, no soft labels needed)
# ══════════════════════════════════════════════════════════════

def validate(model, loader, device, threshold=0.5):
    model.eval()
    ce = nn.CrossEntropyLoss()
    total_loss = 0
    all_preds, all_probs, all_targets = [], [], []

    with torch.no_grad():
        for clips, labels in loader:
            clips, labels = clips.to(device), labels.to(device)
            out   = model(clips)
            loss  = ce(out, labels)
            total_loss += loss.item()

            probs = torch.softmax(out, dim=1)[:, 1]
            preds = (probs > threshold).long()

            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader)
    acc      = np.mean(np.array(all_preds) == np.array(all_targets))
    return avg_loss, acc, np.array(all_preds), np.array(all_probs), np.array(all_targets)


# ══════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════

def main():
    ensure_dir(OUTPUT_DIR)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\n" + "="*60)
    print("  STUDENT TRAINING — Knowledge Distillation")
    print("="*60)
    print(f"  Device      : {device}")
    print(f"  Input size  : {STUDENT_IMG_SIZE}×{STUDENT_IMG_SIZE}")
    print(f"  Batch size  : {STUDENT_BATCH_SIZE}")
    print(f"  Epochs      : {STUDENT_EPOCHS}")
    print(f"  LR          : {STUDENT_LR}")
    print(f"  Temperature : {DISTILL_TEMPERATURE}")
    print(f"  Alpha       : {DISTILL_ALPHA}")
    print("="*60)

    # ── Check soft labels exist ───────────────────────────────
    if not os.path.exists(SOFT_LABELS_PATH):
        raise FileNotFoundError(
            f"\n[ERROR] Soft labels not found: {SOFT_LABELS_PATH}\n"
            "Run first: python extract_soft_labels.py"
        )

    # ── Datasets ──────────────────────────────────────────────
    train_set = DistillDataset(
        TRAIN_PATH,
        soft_labels_path=SOFT_LABELS_PATH,
        mode="train",
        img_size=STUDENT_IMG_SIZE,
        num_frames=STUDENT_NUM_FRAMES,
    )
    val_set = VideoDataset(
        VAL_PATH,
        mode="val",
        img_size=STUDENT_IMG_SIZE,
        num_frames=STUDENT_NUM_FRAMES,
    )

    # Weighted sampler to handle class imbalance
    sample_weights = train_set.get_class_weights()
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )

    train_loader = DataLoader(
        train_set, batch_size=STUDENT_BATCH_SIZE,
        sampler=sampler,
        num_workers=NUM_WORKERS, pin_memory=True,
    )
    val_loader = DataLoader(
        val_set, batch_size=STUDENT_BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True,
    )

    print(f"\n[INFO] Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    # ── Model ─────────────────────────────────────────────────
    student = MobileNet3D(num_classes=2).to(device)
    params  = sum(p.numel() for p in student.parameters()) / 1e6
    print(f"[INFO] Student parameters: {params:.2f}M")

    # ── Loss, optimizer, scheduler ────────────────────────────
    distill_criterion = DistillationLoss(T=DISTILL_TEMPERATURE, alpha=DISTILL_ALPHA)
    ce_criterion      = nn.CrossEntropyLoss()

    optimizer = optim.AdamW(student.parameters(),
                            lr=STUDENT_LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=STUDENT_EPOCHS
    )
    scaler = torch.amp.GradScaler()

    # ── CSV log ───────────────────────────────────────────────
    log_path = os.path.join(OUTPUT_DIR, "student_train_log.csv")
    log_fields = ["epoch", "train_loss", "train_acc",
                  "val_loss", "val_acc", "lr", "time_s"]

    # ── Training loop ─────────────────────────────────────────
    history = {k: [] for k in ["train_loss", "train_acc", "val_loss", "val_acc"]}
    best_val_acc    = 0.0
    patience_count  = 0
    first_row       = True

    for epoch in range(1, STUDENT_EPOCHS + 1):
        t0 = time.time()
        student.train()
        train_loss = 0.0
        correct = total = 0

        for clips, hard_labels, soft_labels in tqdm(
            train_loader, desc=f"Epoch {epoch}/{STUDENT_EPOCHS}", leave=False
        ):
            clips       = clips.to(device)
            hard_labels = hard_labels.to(device)
            soft_labels = soft_labels.to(device)

            optimizer.zero_grad()

            with torch.amp.autocast(device_type="cuda"):
                logits = student(clips)
                loss   = distill_criterion(logits, hard_labels, soft_labels)

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            preds = logits.argmax(1)
            correct += (preds == hard_labels).sum().item()
            total   += hard_labels.size(0)

        train_loss /= len(train_loader)
        train_acc   = correct / total

        # Validate
        val_loss, val_acc, _, _, _ = validate(student, val_loader, device)

        scheduler.step()
        elapsed    = time.time() - t0
        current_lr = scheduler.get_last_lr()[0]

        # History
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        # CSV log
        row = {
            "epoch": epoch,
            "train_loss": f"{train_loss:.4f}",
            "train_acc":  f"{train_acc:.4f}",
            "val_loss":   f"{val_loss:.4f}",
            "val_acc":    f"{val_acc:.4f}",
            "lr":         f"{current_lr:.2e}",
            "time_s":     f"{elapsed:.1f}",
        }
        save_csv_log(log_path, row, write_header=first_row)
        first_row = False

        print(f"\nEpoch {epoch:03d}/{STUDENT_EPOCHS} | "
              f"Train Loss: {train_loss:.4f}  Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}  Acc: {val_acc:.4f} | "
              f"LR: {current_lr:.2e}  Time: {elapsed:.1f}s")

        # Save best
        if val_acc > best_val_acc:
            best_val_acc  = val_acc
            patience_count = 0
            torch.save(student.state_dict(), STUDENT_MODEL_PATH)
            print(f"  🔥 New best student saved  (val_acc={best_val_acc:.4f})")
        else:
            patience_count += 1
            if patience_count >= STUDENT_PATIENCE:
                print(f"\n[INFO] Early stopping after {STUDENT_PATIENCE} epochs "
                      "without improvement.")
                break

    # Last checkpoint
    torch.save(student.state_dict(),
               os.path.join(OUTPUT_DIR, "student_last.pt"))

    # ── Training curves ───────────────────────────────────────
    plot_metrics(
        history["train_loss"], history["val_loss"],
        history["train_acc"],  history["val_acc"],
        output_dir=OUTPUT_DIR,
    )
    # Rename to student-specific filenames
    import shutil
    for src, dst in [
        ("outputs/loss_curve.png",     "outputs/student_loss_curve.png"),
        ("outputs/accuracy_curve.png", "outputs/student_accuracy_curve.png"),
    ]:
        if os.path.exists(src):
            shutil.copy2(src, dst)

    save_metrics_csv(
        history["train_loss"], history["val_loss"],
        history["train_acc"],  history["val_acc"],
        output_dir=OUTPUT_DIR,
    )

    # Teacher vs student comparison (if teacher log exists)
    plot_teacher_vs_student(
        os.path.join(OUTPUT_DIR, "metrics.csv"),
        os.path.join(OUTPUT_DIR, "student_train_log.csv"),
        output_dir=OUTPUT_DIR,
    )

    # ── Final evaluation ──────────────────────────────────────
    print("\n[INFO] Loading best student for final evaluation...")
    student.load_state_dict(
        torch.load(STUDENT_MODEL_PATH, map_location=device, weights_only=False)
    )
    _, _, preds, probs, targets = validate(student, val_loader, device)

    # Get video paths for error analysis
    val_paths = [p for p, _ in val_set.samples]

    full_evaluation(
        targets, preds, probs,
        paths=val_paths,
        output_dir=OUTPUT_DIR,
        prefix="student_",
    )

    print(f"\n✓ Student training complete.")
    print(f"  Best val accuracy : {best_val_acc:.4f}")
    print(f"  Checkpoint        : {STUDENT_MODEL_PATH}")
    print(f"  All plots         : {OUTPUT_DIR}/")
    print(f"\nNext step: python test.py  (to evaluate both models side by side)")


if __name__ == "__main__":
    main()
