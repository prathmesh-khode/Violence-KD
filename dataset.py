"""
dataset.py
──────────
VideoDataset     — for teacher training + student evaluation.
                   Uses your existing augmentation pipeline (from your code),
                   extended with temporal multi-scale sampling and color jitter.

DistillDataset   — wraps VideoDataset, loads pre-saved soft labels from
                   teacher for knowledge distillation training of student.
"""

import os
import cv2
import random
import numpy as np
import torch
from torch.utils.data import Dataset

from config import NUM_FRAMES, IMG_SIZE

# ── Normalization ────────────────────────────────────────────
MEAN = np.array([0.45, 0.45, 0.45], dtype=np.float32)
STD  = np.array([0.225, 0.225, 0.225], dtype=np.float32)


# ══════════════════════════════════════════════════════════════
#  AUGMENTATION HELPERS  (applied per-frame consistently)
# ══════════════════════════════════════════════════════════════

def cutout(frame):
    h, w, _ = frame.shape
    ch, cw = int(h * 0.2), int(w * 0.2)
    y = random.randint(0, h - ch)
    x = random.randint(0, w - cw)
    frame[y:y+ch, x:x+cw] = 0
    return frame


def apply_augmentation(frame):
    """Your original per-frame augmentation pipeline, intact."""
    # Horizontal flip
    if random.random() < 0.5:
        frame = cv2.flip(frame, 1)

    # Brightness + Contrast
    if random.random() < 0.5:
        alpha = 1.0 + random.uniform(-0.25, 0.25)
        beta  = random.uniform(-30, 30)
        frame = np.clip(frame * alpha + beta, 0, 255).astype(np.uint8)

    # Gaussian Blur
    if random.random() < 0.2:
        k = random.choice([3, 5])
        frame = cv2.GaussianBlur(frame, (k, k), 0)

    # Random noise
    if random.random() < 0.3:
        noise = np.random.normal(0, 10, frame.shape).astype(np.float32)
        frame = np.clip(frame.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    # Color jitter (HSV)
    if random.random() < 0.3:
        hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[..., 0] = (hsv[..., 0] + random.uniform(-10, 10)) % 180
        hsv[..., 1] = np.clip(hsv[..., 1] * random.uniform(0.8, 1.2), 0, 255)
        hsv[..., 2] = np.clip(hsv[..., 2] * random.uniform(0.8, 1.2), 0, 255)
        frame = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

    # Cutout
    if random.random() < 0.2:
        frame = cutout(frame)

    return frame


# ══════════════════════════════════════════════════════════════
#  FRAME SAMPLING
# ══════════════════════════════════════════════════════════════

def sample_frames(total_frames, num_frames, mode="train"):
    """
    train : random start + occasional multi-scale temporal sampling
    val   : deterministic uniform sampling
    """
    if total_frames <= 0:
        return np.zeros(num_frames, dtype=int)

    if mode == "train":
        # Standard random-start clip
        start = random.randint(0, max(0, total_frames - num_frames))
        indices = np.arange(start, start + num_frames)

        # 30% chance of multi-scale sampling (captures faster / slower motion)
        if random.random() < 0.3 and total_frames >= num_frames * 2:
            step = random.randint(2, 3)
            indices = np.arange(0, num_frames * step, step)
            if len(indices) > num_frames:
                indices = indices[:num_frames]
    else:
        indices = np.linspace(0, total_frames - 1, num_frames)

    return np.clip(indices, 0, total_frames - 1).astype(int)


# ══════════════════════════════════════════════════════════════
#  VIDEO DATASET
# ══════════════════════════════════════════════════════════════

class VideoDataset(Dataset):
    """
    Loads videos from:
      root_dir/0/*.mp4   (non-violence)
      root_dir/1/*.mp4   (violence)

    Args:
        root_dir  : e.g. 'data/processed/train'
        mode      : 'train' (augment) or 'val' (no augment)
        img_size  : spatial resize target (224 for teacher, 112 for student)
        num_frames: temporal clip length (16 for both)
    """

    def __init__(self, root_dir, mode="train", img_size=IMG_SIZE, num_frames=NUM_FRAMES):
        self.root_dir   = root_dir
        self.mode       = mode
        self.img_size   = img_size
        self.num_frames = num_frames
        self.samples    = []

        for label_str in ["0", "1"]:
            class_path = os.path.join(root_dir, label_str)
            if not os.path.isdir(class_path):
                continue
            label = int(label_str)
            for fname in os.listdir(class_path):
                if fname.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
                    self.samples.append((os.path.join(class_path, fname), label))

        if len(self.samples) == 0:
            raise ValueError(f"[ERROR] No videos found in {root_dir}")

        n0 = sum(1 for _, l in self.samples if l == 0)
        n1 = sum(1 for _, l in self.samples if l == 1)
        print(f"[INFO] {root_dir} ({mode}) → total={len(self.samples)}  "
              f"non-violence={n0}  violence={n1}")

    def get_class_weights(self):
        """Per-sample weights for WeightedRandomSampler (handles class imbalance)."""
        counts = [0, 0]
        for _, label in self.samples:
            counts[label] += 1
        total   = sum(counts)
        weights = [total / (2.0 * c + 1e-9) for c in counts]
        return [weights[label] for _, label in self.samples]

    def _load_video(self, path):
        cap    = cv2.VideoCapture(path)
        n      = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if n <= 0:
            cap.release()
            raise ValueError(f"Empty video: {path}")

        indices = sample_frames(n, self.num_frames, self.mode)
        frames  = []

        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()

            if not ret or frame is None:
                # Duplicate last valid frame instead of black frame
                if frames:
                    frame_rgb = frames[-1].copy()
                    frames.append(frame_rgb)
                    continue
                else:
                    frame = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)

            frame = cv2.resize(frame, (self.img_size, self.img_size))

            if self.mode == "train":
                frame = apply_augmentation(frame)

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = frame.astype(np.float32) / 255.0
            frame = (frame - MEAN) / STD
            frames.append(frame)

        cap.release()

        # Pad if video shorter than num_frames
        while len(frames) < self.num_frames:
            frames.append(frames[-1].copy())

        frames = np.stack(frames, axis=0).astype(np.float32)   # (T,H,W,C)
        return torch.from_numpy(frames).permute(3, 0, 1, 2)    # (C,T,H,W)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_path, label = self.samples[idx]
        try:
            clip = self._load_video(video_path)
            return clip, torch.tensor(label, dtype=torch.long)
        except Exception as e:
            print(f"[WARN] Skipping corrupted video: {video_path} | {e}")
            fallback = (idx + 1) % len(self.samples)
            return self.__getitem__(fallback)


# ══════════════════════════════════════════════════════════════
#  DISTILLATION DATASET
# ══════════════════════════════════════════════════════════════

class DistillDataset(Dataset):
    """
    Wraps VideoDataset and attaches pre-saved teacher soft labels.

    Returns (clip, hard_label, soft_label) per sample.
    soft_label is a 2-element float tensor [prob_nonviolence, prob_violence]
    saved at temperature T=4 by extract_soft_labels.py.
    """

    def __init__(self, root_dir, soft_labels_path,
                 mode="train", img_size=112, num_frames=NUM_FRAMES):
        self.base = VideoDataset(
            root_dir, mode=mode, img_size=img_size, num_frames=num_frames
        )
        print(f"[INFO] Loading soft labels from: {soft_labels_path}")
        self.soft_labels = torch.load(soft_labels_path, weights_only=False)
        matched = sum(1 for p, _ in self.base.samples if p in self.soft_labels)
        print(f"[INFO] Soft labels loaded: {len(self.soft_labels)} total, "
              f"{matched}/{len(self.base)} matched to current dataset")

    def __len__(self):
        return len(self.base)

    def get_class_weights(self):
        return self.base.get_class_weights()

    def __getitem__(self, idx):
        clip, hard_label = self.base[idx]
        video_path, _    = self.base.samples[idx]

        if video_path in self.soft_labels:
            soft = torch.tensor(self.soft_labels[video_path], dtype=torch.float32)
        else:
            # Fallback: one-hot if teacher soft label missing for this path
            soft = torch.zeros(2, dtype=torch.float32)
            soft[hard_label.item()] = 1.0

        return clip, hard_label, soft


# ══════════════════════════════════════════════════════════════
#  QUICK CHECK
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    print("dataset.py loaded successfully.")
    print("Expected structure:")
    print("  data/processed/train/0/  ← non-violence")
    print("  data/processed/train/1/  ← violence")
    print("  data/processed/val/0/")
    print("  data/processed/val/1/")
