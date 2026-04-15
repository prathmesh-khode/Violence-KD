"""
test.py
───────
Evaluates teacher (ViolenceModel) or student (MobileNet3D) on the val set.
Produces every metric and saves every plot.

Usage:
  python test.py                          # evaluate teacher
  python test.py --model student          # evaluate student
  python test.py --model both             # evaluate both + comparison table
  python test.py --model teacher --threshold 0.6
"""

import os
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import (
    VAL_PATH, OUTPUT_DIR,
    BEST_MODEL_PATH, STUDENT_MODEL_PATH,
    NUM_FRAMES, IMG_SIZE, STUDENT_IMG_SIZE,
    NUM_WORKERS,
)
from models import ViolenceModel, MobileNet3D
from dataset import VideoDataset
from utils import full_evaluation, ensure_dir, plot_teacher_vs_student


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model",      default="teacher",
                   choices=["teacher", "student", "both"])
    p.add_argument("--threshold",  type=float, default=0.5)
    p.add_argument("--batch_size", type=int,   default=16)
    return p.parse_args()


def load_model_weights(model, path, device):
    """
    Robust loader — handles:
      1. Raw state dict (most common — saved with torch.save(model.state_dict(), path))
      2. Checkpoint dict with 'model_state' key
      3. Checkpoint dict with 'state_dict' key
    """
    raw = torch.load(path, map_location=device, weights_only=False)

    if isinstance(raw, dict):
        if "model_state" in raw:
            model.load_state_dict(raw["model_state"])
        elif "state_dict" in raw:
            model.load_state_dict(raw["state_dict"])
        else:
            # The dict itself is the state dict
            model.load_state_dict(raw)
    else:
        raise ValueError(f"Unexpected checkpoint format: {type(raw)}")

    model.eval()
    return model


def run_inference(model, loader, device, threshold=0.5):
    model.eval()
    all_preds, all_probs, all_targets = [], [], []

    with torch.no_grad():
        for clips, labels in tqdm(loader, desc="Evaluating", leave=False):
            clips = clips.to(device)
            probs = torch.softmax(model(clips), dim=1)[:, 1].cpu().numpy()
            preds = (probs > threshold).astype(int)
            all_preds.extend(preds.tolist())
            all_probs.extend(probs.tolist())
            all_targets.extend(labels.numpy().tolist())

    return np.array(all_preds), np.array(all_probs), np.array(all_targets)


def eval_teacher(device, batch_size, threshold):
    print("\n" + "="*55)
    print("  EVALUATING: Teacher (ViolenceModel / R3D-18)")
    print("="*55)

    if not os.path.exists(BEST_MODEL_PATH):
        raise FileNotFoundError(
            f"[ERROR] Teacher checkpoint not found: {BEST_MODEL_PATH}"
        )

    model = ViolenceModel().to(device)
    model = load_model_weights(model, BEST_MODEL_PATH, device)

    params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[INFO] Params: {params:.1f}M  |  Input: {IMG_SIZE}×{IMG_SIZE}  "
          f"|  Threshold: {threshold}")

    val_set = VideoDataset(VAL_PATH, mode="val",
                           img_size=IMG_SIZE, num_frames=NUM_FRAMES)
    loader  = DataLoader(val_set, batch_size=batch_size,
                         shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    paths   = [p for p, _ in val_set.samples]

    preds, probs, targets = run_inference(model, loader, device, threshold)

    summary = full_evaluation(
        targets, preds, probs,
        paths=paths,
        output_dir=OUTPUT_DIR,
        prefix="teacher_",
    )
    return summary


def eval_student(device, batch_size, threshold):
    print("\n" + "="*55)
    print("  EVALUATING: Student (MobileNet3D)")
    print("="*55)

    if not os.path.exists(STUDENT_MODEL_PATH):
        raise FileNotFoundError(
            f"[ERROR] Student checkpoint not found: {STUDENT_MODEL_PATH}\n"
            "Run first: python train_student.py"
        )

    model = MobileNet3D(num_classes=2).to(device)
    model = load_model_weights(model, STUDENT_MODEL_PATH, device)

    params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[INFO] Params: {params:.2f}M  |  Input: {STUDENT_IMG_SIZE}×{STUDENT_IMG_SIZE}  "
          f"|  Threshold: {threshold}")

    val_set = VideoDataset(VAL_PATH, mode="val",
                           img_size=STUDENT_IMG_SIZE, num_frames=NUM_FRAMES)
    loader  = DataLoader(val_set, batch_size=batch_size,
                         shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    paths   = [p for p, _ in val_set.samples]

    preds, probs, targets = run_inference(model, loader, device, threshold)

    summary = full_evaluation(
        targets, preds, probs,
        paths=paths,
        output_dir=OUTPUT_DIR,
        prefix="student_",
    )
    return summary


def main():
    args = get_args()
    ensure_dir(OUTPUT_DIR)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    if args.model == "teacher":
        eval_teacher(device, args.batch_size, args.threshold)

    elif args.model == "student":
        eval_student(device, args.batch_size, args.threshold)

    elif args.model == "both":
        t_summary = eval_teacher(device, args.batch_size, args.threshold)
        s_summary = eval_student(device, args.batch_size, args.threshold)

        print("\n" + "="*58)
        print("  COMPARISON: Teacher vs Student")
        print("="*58)
        metrics = ["accuracy", "f1", "roc_auc", "pr_auc",
                   "mcc", "balanced_accuracy", "recall", "precision"]
        print(f"  {'Metric':<24} {'Teacher':>10} {'Student':>10} {'Diff':>8}")
        print(f"  {'-'*54}")
        for m in metrics:
            tv = t_summary.get(m, 0)
            sv = s_summary.get(m, 0)
            diff = sv - tv
            sign = "+" if diff >= 0 else ""
            print(f"  {m:<24} {tv:>10.4f} {sv:>10.4f}  {sign}{diff:>6.4f}")

        plot_teacher_vs_student(
            os.path.join(OUTPUT_DIR, "metrics.csv"),
            os.path.join(OUTPUT_DIR, "student_train_log.csv"),
            output_dir=OUTPUT_DIR,
        )

    print(f"\n✓ Evaluation complete. All plots saved to: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
