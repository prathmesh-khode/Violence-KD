"""
utils.py
────────
All helper functions for checkpointing, metrics, and plotting.
Your original functions are preserved exactly.
New additions are clearly marked with # ── NEW ──
"""

import os
import csv
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
    auc,
    classification_report,
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
    matthews_corrcoef,
    balanced_accuracy_score,
    average_precision_score,
    cohen_kappa_score,
    log_loss,
)


# ══════════════════════════════════════════════════════════════
#  YOUR ORIGINAL FUNCTIONS  (unchanged)
# ══════════════════════════════════════════════════════════════

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def save_checkpoint(model, optimizer, epoch, best_acc, path):
    ensure_dir(os.path.dirname(path) if os.path.dirname(path) else ".")
    torch.save({
        "epoch":           epoch,
        "model_state":     model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "best_acc":        best_acc,
    }, path)


def load_checkpoint(model, path, device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    print(f"[INFO] Loaded checkpoint from {path}")
    return model


def save_metrics_csv(train_losses, val_losses, train_accs, val_accs,
                     output_dir="outputs"):
    ensure_dir(output_dir)
    df = pd.DataFrame({
        "train_loss": train_losses,
        "val_loss":   val_losses,
        "train_acc":  train_accs,
        "val_acc":    val_accs,
    })
    df.to_csv(f"{output_dir}/metrics.csv", index=False)
    print(f"[INFO] Metrics CSV saved → {output_dir}/metrics.csv")


def plot_metrics(train_losses, val_losses, train_accs, val_accs,
                 output_dir="outputs"):
    ensure_dir(output_dir)

    plt.figure(figsize=(7, 4))
    plt.plot(train_losses, label="Train Loss", color="#1f77b4")
    plt.plot(val_losses,   label="Val Loss",   color="#ff7f0e")
    plt.xlabel("Epoch"); plt.ylabel("Loss")
    plt.title("Loss Curve"); plt.legend(); plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/loss_curve.png", dpi=150); plt.close()

    plt.figure(figsize=(7, 4))
    plt.plot(train_accs, label="Train Acc", color="#1f77b4")
    plt.plot(val_accs,   label="Val Acc",   color="#ff7f0e")
    plt.xlabel("Epoch"); plt.ylabel("Accuracy")
    plt.title("Accuracy Curve"); plt.legend(); plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/accuracy_curve.png", dpi=150); plt.close()


def plot_confusion_matrix(targets, preds, output_dir="outputs",
                          prefix="", title="Confusion Matrix"):
    ensure_dir(output_dir)
    cm = confusion_matrix(targets, preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Non-Violence", "Violence"],
                yticklabels=["Non-Violence", "Violence"],
                annot_kws={"size": 14})
    # Add percentage annotations
    total = cm.sum()
    for i in range(2):
        for j in range(2):
            pct = cm[i, j] / total * 100
            plt.text(j + 0.5, i + 0.72, f"({pct:.1f}%)",
                     ha="center", va="center", fontsize=9, color="gray")
    plt.title(title, fontsize=13)
    plt.xlabel("Predicted"); plt.ylabel("Actual")
    plt.tight_layout()
    fname = f"{output_dir}/{prefix}confusion_matrix.png"
    plt.savefig(fname, dpi=150); plt.close()
    print(f"[INFO] Saved: {fname}")
    return cm


def plot_roc_curve(targets, probs, output_dir="outputs", prefix=""):
    ensure_dir(output_dir)
    fpr, tpr, _ = roc_curve(targets, probs)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color="#1f77b4", lw=2, label=f"AUC = {roc_auc:.4f}")
    plt.fill_between(fpr, tpr, alpha=0.08, color="#1f77b4")
    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlim([0, 1]); plt.ylim([0, 1.02])
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve (AUC={roc_auc:.4f})", fontsize=13)
    plt.legend(loc="lower right"); plt.grid(alpha=0.3)
    plt.tight_layout()
    fname = f"{output_dir}/{prefix}roc_curve.png"
    plt.savefig(fname, dpi=150); plt.close()
    print(f"[INFO] Saved: {fname}")
    return roc_auc


def plot_pr_curve(targets, probs, output_dir="outputs", prefix=""):
    ensure_dir(output_dir)
    precision, recall, _ = precision_recall_curve(targets, probs)
    pr_auc = auc(recall, precision)
    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, color="#ff7f0e", lw=2, label=f"AUC = {pr_auc:.4f}")
    plt.fill_between(recall, precision, alpha=0.08, color="#ff7f0e")
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title("Precision-Recall Curve", fontsize=13)
    plt.legend(loc="upper right"); plt.grid(alpha=0.3)
    plt.tight_layout()
    fname = f"{output_dir}/{prefix}pr_curve.png"
    plt.savefig(fname, dpi=150); plt.close()
    print(f"[INFO] Saved: {fname}")
    return pr_auc


def save_classification_report(targets, preds, output_dir="outputs", prefix=""):
    ensure_dir(output_dir)
    report = classification_report(
        targets, preds,
        target_names=["Non-Violence", "Violence"],
        digits=4
    )
    fname = f"{output_dir}/{prefix}classification_report.txt"
    with open(fname, "w") as f:
        f.write(report)
    print("\n📄 Classification Report:\n")
    print(report)
    return report


def per_class_accuracy(cm):
    acc = cm.diagonal() / (cm.sum(axis=1) + 1e-9)
    print("\n📊 Per-Class Accuracy:")
    print(f"  Non-Violence : {acc[0]:.4f}")
    print(f"  Violence     : {acc[1]:.4f}")
    return acc


def save_error_analysis(targets, preds, probs, paths, output_dir="outputs", prefix=""):
    ensure_dir(output_dir)
    errors = []
    for t, p, prob, path in zip(targets, preds, probs, paths):
        if int(t) != int(p):
            error_type = "False Positive" if int(p) == 1 else "False Negative"
            errors.append({
                "video_path": path,
                "true_label": int(t),
                "pred_label": int(p),
                "violence_prob": float(prob),
                "error_type": error_type,
            })
    df = pd.DataFrame(errors)
    fname = f"{output_dir}/{prefix}errors.csv"
    df.to_csv(fname, index=False)
    print(f"[INFO] Saved {len(errors)} misclassified samples → {fname}")


def compute_metrics(targets, preds):
    acc = accuracy_score(targets, preds)
    f1  = f1_score(targets, preds, zero_division=0)
    return acc, f1


class EarlyStopping:
    def __init__(self, patience=12, delta=0.0):
        self.patience   = patience
        self.delta      = delta
        self.counter    = 0
        self.best_loss  = np.Inf
        self.early_stop = False

    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter   = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


# ══════════════════════════════════════════════════════════════
#  ── NEW ── EXTENDED METRICS + PLOTS
# ══════════════════════════════════════════════════════════════

def plot_class_accuracy_bar(cm, output_dir="outputs", prefix=""):
    """Bar chart of per-class accuracy."""
    ensure_dir(output_dir)
    class_acc = cm.diagonal() / (cm.sum(axis=1) + 1e-9)
    colors = ["#2196F3", "#F44336"]
    plt.figure(figsize=(6, 4))
    bars = plt.bar(["Non-Violence", "Violence"], class_acc, color=colors, width=0.4)
    for bar, acc in zip(bars, class_acc):
        plt.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.01, f"{acc:.4f}",
                 ha="center", va="bottom", fontsize=12, fontweight="bold")
    plt.ylim(0, 1.15); plt.ylabel("Accuracy")
    plt.title("Per-Class Accuracy", fontsize=13)
    plt.grid(alpha=0.3, axis="y"); plt.tight_layout()
    fname = f"{output_dir}/{prefix}class_accuracy.png"
    plt.savefig(fname, dpi=150); plt.close()
    print(f"[INFO] Saved: {fname}")


def plot_confidence_distribution(probs, targets, output_dir="outputs", prefix=""):
    """Histogram of predicted violence probability, split by true class."""
    ensure_dir(output_dir)
    probs = np.array(probs); targets = np.array(targets)
    plt.figure(figsize=(8, 4))
    plt.hist(probs[targets == 0], bins=40, alpha=0.65,
             color="#2196F3", label="True Non-Violence", density=True)
    plt.hist(probs[targets == 1], bins=40, alpha=0.65,
             color="#F44336", label="True Violence", density=True)
    plt.axvline(0.5, color="k", linestyle="--", lw=1.5, label="Threshold 0.5")
    plt.xlabel("Predicted Violence Probability"); plt.ylabel("Density")
    plt.title("Confidence Distribution by True Class", fontsize=13)
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    fname = f"{output_dir}/{prefix}confidence_dist.png"
    plt.savefig(fname, dpi=150); plt.close()
    print(f"[INFO] Saved: {fname}")


def plot_threshold_scan(probs, targets, output_dir="outputs", prefix=""):
    """Sweep threshold and plot F1, Precision, Recall — finds optimal threshold."""
    ensure_dir(output_dir)
    thresholds = np.arange(0.10, 0.95, 0.05)
    f1s, precs, recs = [], [], []
    for t in thresholds:
        p = (np.array(probs) > t).astype(int)
        f1s.append(f1_score(targets, p, zero_division=0))
        precs.append(precision_score(targets, p, zero_division=0))
        recs.append(recall_score(targets, p, zero_division=0))

    best_idx = int(np.argmax(f1s))
    plt.figure(figsize=(8, 5))
    plt.plot(thresholds, f1s,   color="#4caf50", lw=2, label="F1")
    plt.plot(thresholds, precs, color="#2196F3", lw=2, label="Precision")
    plt.plot(thresholds, recs,  color="#F44336", lw=2, label="Recall")
    plt.axvline(thresholds[best_idx], color="#4caf50", linestyle="--", lw=1.5,
                label=f"Best F1={f1s[best_idx]:.3f} @ {thresholds[best_idx]:.2f}")
    plt.xlabel("Decision Threshold"); plt.ylabel("Score")
    plt.title("F1 / Precision / Recall vs. Threshold", fontsize=13)
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    fname = f"{output_dir}/{prefix}threshold_scan.png"
    plt.savefig(fname, dpi=150); plt.close()
    print(f"[INFO] Saved: {fname}")
    print(f"[INFO] Optimal threshold: {thresholds[best_idx]:.2f}  "
          f"(F1={f1s[best_idx]:.4f})")
    return float(thresholds[best_idx])


def plot_teacher_vs_student(teacher_csv, student_csv,
                             output_dir="outputs"):
    """Overlay teacher and student val accuracy on one plot."""
    ensure_dir(output_dir)
    if not (os.path.exists(teacher_csv) and os.path.exists(student_csv)):
        return

    def read(path):
        df = pd.read_csv(path)
        return df["val_acc"].tolist()

    t = read(teacher_csv)
    s = read(student_csv)
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(t)+1), t,
             label=f"Teacher (R3D-18)  best={max(t):.4f}",
             color="#1f77b4", lw=2)
    plt.plot(range(1, len(s)+1), s,
             label=f"Student (MobileNet3D)  best={max(s):.4f}",
             color="#9c27b0", lw=2, linestyle="--")
    plt.xlabel("Epoch"); plt.ylabel("Val Accuracy")
    plt.title("Teacher vs Student Val Accuracy", fontsize=13)
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    fname = f"{output_dir}/teacher_vs_student.png"
    plt.savefig(fname, dpi=150); plt.close()
    print(f"[INFO] Saved: {fname}")


def save_csv_log(log_path, row_dict, write_header=False):
    """Append one row to a CSV training log."""
    ensure_dir(os.path.dirname(log_path) if os.path.dirname(log_path) else ".")
    mode = "w" if write_header else "a"
    with open(log_path, mode, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row_dict.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row_dict)


def full_evaluation(targets, preds, probs, paths=None,
                    output_dir="outputs", prefix=""):
    """
    Run every metric and save every plot in one call.
    Returns dict of key scalar metrics.
    """
    ensure_dir(output_dir)
    targets = np.array(targets)
    preds   = np.array(preds)
    probs   = np.array(probs)

    # ── Text report ──
    save_classification_report(targets, preds, output_dir, prefix)

    # ── Scalar metrics ──
    acc     = accuracy_score(targets, preds)
    f1      = f1_score(targets, preds, zero_division=0)
    prec    = precision_score(targets, preds, zero_division=0)
    rec     = recall_score(targets, preds, zero_division=0)
    mcc     = matthews_corrcoef(targets, preds)
    bal     = balanced_accuracy_score(targets, preds)
    kappa   = cohen_kappa_score(targets, preds)
    ll      = log_loss(targets, probs)
    ap      = average_precision_score(targets, probs)

    cm = confusion_matrix(targets, preds)
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp + 1e-9)
    sensitivity = tp / (tp + fn + 1e-9)

    fpr, tpr, _ = roc_curve(targets, probs)
    roc_auc = auc(fpr, tpr)

    print("\n" + "="*55)
    print(f"  Accuracy          : {acc:.4f}")
    print(f"  Balanced Accuracy : {bal:.4f}")
    print(f"  F1 (Violence)     : {f1:.4f}")
    print(f"  Precision         : {prec:.4f}")
    print(f"  Recall/Sensitivity: {sensitivity:.4f}")
    print(f"  Specificity       : {specificity:.4f}")
    print(f"  ROC AUC           : {roc_auc:.4f}")
    print(f"  PR AUC            : {ap:.4f}")
    print(f"  MCC               : {mcc:.4f}")
    print(f"  Cohen's Kappa     : {kappa:.4f}")
    print(f"  Log Loss          : {ll:.4f}")
    print(f"  TN={tn}  FP={fp}  FN={fn}  TP={tp}")
    print("="*55)

    # ── Plots ──
    plot_confusion_matrix(targets, preds, output_dir, prefix,
                          title=f"{prefix.strip('_').title()} Confusion Matrix")
    plot_roc_curve(targets, probs, output_dir, prefix)
    plot_pr_curve(targets, probs, output_dir, prefix)
    plot_class_accuracy_bar(cm, output_dir, prefix)
    plot_confidence_distribution(probs, targets, output_dir, prefix)
    optimal_t = plot_threshold_scan(probs, targets, output_dir, prefix)

    # ── Error analysis ──
    if paths is not None:
        save_error_analysis(targets, preds, probs, paths, output_dir, prefix)

    # ── Save scalar summary CSV ──
    summary = {
        "accuracy": acc, "balanced_accuracy": bal,
        "f1": f1, "precision": prec, "recall": rec,
        "specificity": specificity, "roc_auc": roc_auc,
        "pr_auc": ap, "mcc": mcc, "kappa": kappa,
        "log_loss": ll, "optimal_threshold": optimal_t,
    }
    df = pd.DataFrame([summary])
    df.to_csv(f"{output_dir}/{prefix}metrics_summary.csv", index=False)
    print(f"[INFO] Metrics summary saved → {output_dir}/{prefix}metrics_summary.csv")

    return summary
