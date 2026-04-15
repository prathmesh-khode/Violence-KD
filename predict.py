"""
predict.py
──────────
Run violence detection on a video file or live camera.
Works with both teacher (224×224) and student (112×112).

Usage:
  python predict.py --input data/processed/val/1/V_3.mp4
  python predict.py --input data/processed/val/1/V_3.mp4 --model student
  python predict.py --input 0 --model student --live
  python predict.py --input rtsp://user:pass@ip/stream --model student --live
"""

import os
import cv2
import time
import argparse
import numpy as np
import torch
from collections import deque

from config import (
    OUTPUT_DIR, BEST_MODEL_PATH, STUDENT_MODEL_PATH,
    NUM_FRAMES, IMG_SIZE, STUDENT_IMG_SIZE,
)
from models import ViolenceModel, MobileNet3D

MEAN = np.array([0.45, 0.45, 0.45], dtype=np.float32)
STD  = np.array([0.225, 0.225, 0.225], dtype=np.float32)


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input",      default="data/processed/val/1/V_3.mp4")
    p.add_argument("--model",      default="teacher", choices=["teacher", "student"])
    p.add_argument("--output",     default="outputs/predicted_output.mp4")
    p.add_argument("--threshold",  type=float, default=0.5)
    p.add_argument("--stride",     type=int,   default=4,
                   help="Run inference every N frames")
    p.add_argument("--smooth",     type=int,   default=5,
                   help="Temporal smoothing window size")
    p.add_argument("--live",       action="store_true",
                   help="Live display mode — don't save output file")
    p.add_argument("--no_display", action="store_true",
                   help="No window — for headless servers")
    return p.parse_args()


def load_model_weights(model, path, device):
    raw = torch.load(path, map_location=device, weights_only=False)
    if isinstance(raw, dict):
        if "model_state" in raw:
            model.load_state_dict(raw["model_state"])
        elif "state_dict" in raw:
            model.load_state_dict(raw["state_dict"])
        else:
            model.load_state_dict(raw)
    else:
        raise ValueError(f"Unexpected checkpoint format: {type(raw)}")
    model.eval()
    return model


def preprocess(frame, img_size):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (img_size, img_size)).astype(np.float32) / 255.0
    return (rgb - MEAN) / STD


def build_clip(buf):
    clip = np.stack(buf, axis=0).transpose(3, 0, 1, 2)[np.newaxis]
    return torch.from_numpy(clip.astype(np.float32))


def draw_overlay(frame, raw_p, smooth_p, pred, thr, fps_str, fidx, motion):
    H, W = frame.shape[:2]
    ov = frame.copy()
    cv2.rectangle(ov, (0, 0), (W, 72), (15, 15, 15), -1)
    cv2.addWeighted(ov, 0.55, frame, 0.45, 0, frame)

    is_v  = (pred == 1)
    label = "VIOLENCE DETECTED" if is_v else "Non-Violence"
    color = (0, 0, 255) if is_v else (50, 210, 50)
    cv2.putText(frame, label, (14, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, color, 2, cv2.LINE_AA)
    cv2.putText(frame, f"Conf: {smooth_p:.3f}  (raw: {raw_p:.3f})",
                (14, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (200, 200, 200), 1, cv2.LINE_AA)

    bx, by, bw, bh = W-222, 16, 202, 20
    cv2.rectangle(frame, (bx, by), (bx+bw, by+bh), (55, 55, 55), -1)
    cv2.rectangle(frame, (bx, by), (bx+int(smooth_p*bw), by+bh),
                  (0, 0, 255) if smooth_p > thr else (50, 210, 50), -1)
    tx = bx + int(thr * bw)
    cv2.line(frame, (tx, by-4), (tx, by+bh+4), (255, 255, 255), 1)

    cv2.rectangle(frame, (0, H-26), (W, H), (15, 15, 15), -1)
    info = f"Frame: {fidx}  |  {fps_str}"
    if motion:
        info += "  |  Motion"
    cv2.putText(frame, info, (10, H-9),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (160, 160, 160), 1, cv2.LINE_AA)

    if is_v:
        cv2.rectangle(frame, (0, 0), (W-1, H-1), (0, 0, 255), 3)
    return frame


def main():
    args   = get_args()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    if args.model == "teacher":
        model    = load_model_weights(ViolenceModel().to(device), BEST_MODEL_PATH, device)
        img_size = IMG_SIZE
        print(f"[INFO] Teacher loaded | input: {img_size}×{img_size}")
    else:
        model    = load_model_weights(MobileNet3D(num_classes=2).to(device),
                                       STUDENT_MODEL_PATH, device)
        img_size = STUDENT_IMG_SIZE
        print(f"[INFO] Student loaded | input: {img_size}×{img_size}")

    try:
        source = int(args.input)
    except ValueError:
        source = args.input

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open: {source}")

    fps    = cap.get(cv2.CAP_PROP_FPS) or 25
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[INFO] {width}×{height}  FPS:{fps:.1f}  Total frames:{total}")

    writer = None
    if not args.live:
        os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True)
        writer = cv2.VideoWriter(args.output,
                                 cv2.VideoWriter_fourcc(*"mp4v"),
                                 fps, (width, height))

    bg_sub = cv2.createBackgroundSubtractorMOG2(
        history=100, varThreshold=50, detectShadows=False)

    frame_buf    = []
    smooth_buf   = deque(maxlen=args.smooth)
    raw_p = smooth_p = 0.0
    pred  = 0
    fidx  = 0
    motion = False
    t0     = time.time()

    print("[INFO] Processing... Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        fidx += 1

        motion = cv2.countNonZero(bg_sub.apply(frame)) > 500

        proc = preprocess(frame, img_size)
        frame_buf.append(proc)
        if len(frame_buf) > NUM_FRAMES:
            frame_buf.pop(0)

        if len(frame_buf) == NUM_FRAMES and fidx % args.stride == 0:
            clip = build_clip(frame_buf).to(device)
            with torch.no_grad():
                raw_p = float(torch.softmax(model(clip), dim=1)[0, 1].cpu())
            smooth_buf.append(raw_p)
            smooth_p = float(np.mean(smooth_buf))
            pred = int(smooth_p > args.threshold)

        fps_str  = f"FPS: {fidx / max(time.time()-t0, 1e-9):.1f}"
        annotated = draw_overlay(frame.copy(), raw_p, smooth_p, pred,
                                 args.threshold, fps_str, fidx, motion)

        if writer:
            writer.write(annotated)

        if not args.no_display:
            cv2.imshow("Violence Detection", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("\n[INFO] Stopped by user.")
                break

        if fidx % 200 == 0:
            print(f"  Frame {fidx}/{total if total>0 else '?'}  "
                  f"Score:{smooth_p:.3f}  {'⚠ VIOLENCE' if pred else 'Normal'}")

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()

    if not args.live and writer:
        print(f"\n✓ Output saved: {args.output}")
    print(f"  Total frames: {fidx}")


if __name__ == "__main__":
    main()
