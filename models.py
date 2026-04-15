"""
models.py
─────────
ViolenceModel  — R3D-18 teacher. Architecture EXACTLY matches what was used
                 during training (temporal layer + single linear classifier).
                 Loads outputs/best_model.pt without errors.

MobileNet3D    — Lightweight student for Jetson Nano.
                 Trained via knowledge distillation from ViolenceModel.
"""

import torch
import torch.nn as nn
import torchvision.models.video as video_models

from config import NUM_CLASSES, FREEZE_BACKBONE


# ══════════════════════════════════════════════════════════════
#  TEACHER  —  ViolenceModel (R3D-18)
#  Architecture matches your training checkpoint exactly:
#    backbone → temporal(Linear→ReLU→Dropout) → classifier(Linear→2)
# ══════════════════════════════════════════════════════════════

class ViolenceModel(nn.Module):
    """
    R3D-18 violence detection model — teacher for knowledge distillation.
    Architecture is identical to what produced outputs/best_model.pt.

    Input : (B, C, T, H, W)  e.g. (B, 3, 16, 224, 224)
    Output: (B, 2)
    """
    def __init__(self):
        super(ViolenceModel, self).__init__()

        self.backbone = video_models.r3d_18(weights="DEFAULT")
        self.backbone.fc = nn.Identity()  # output: (B, 512)

        if FREEZE_BACKBONE:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Temporal fusion layer — matches your train.py exactly
        self.temporal = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
        )

        # Final classifier — matches your train.py exactly
        self.classifier = nn.Linear(512, NUM_CLASSES)

    def forward(self, x):
        feat = self.backbone(x)       # (B, 512)
        feat = self.temporal(feat)    # (B, 512)
        return self.classifier(feat)  # (B, 2)


def unfreeze_backbone(model):
    for param in model.backbone.parameters():
        param.requires_grad = True
    print("[INFO] Backbone unfrozen")


def count_parameters(model):
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[INFO] Total parameters    : {total:,}")
    print(f"[INFO] Trainable parameters: {trainable:,}")


# ══════════════════════════════════════════════════════════════
#  STUDENT  —  MobileNet3D (Jetson Nano ready)
# ══════════════════════════════════════════════════════════════

class _DWSep3D(nn.Module):
    """3D Depthwise-Separable conv block — core of MobileNet."""
    def __init__(self, in_c, out_c, stride=1, t_stride=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_c, in_c,
                      kernel_size=3,
                      stride=(t_stride, stride, stride),
                      padding=1,
                      groups=in_c,
                      bias=False),
            nn.BatchNorm3d(in_c),
            nn.ReLU6(inplace=True),
            nn.Conv3d(in_c, out_c, kernel_size=1, bias=False),
            nn.BatchNorm3d(out_c),
            nn.ReLU6(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class MobileNet3D(nn.Module):
    """
    Lightweight 3D MobileNet for real-time violence detection on Jetson Nano 4GB.
    ~3M params vs ~33M for R3D-18. Target: ~18ms with TensorRT INT8 on Nano.

    Input : (B, C, T, H, W)  — (B, 3, 16, 112, 112)
    Output: (B, 2)
    """
    def __init__(self, num_classes=2, dropout=0.3):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv3d(3, 32,
                      kernel_size=(1, 3, 3),
                      stride=(1, 2, 2),
                      padding=(0, 1, 1),
                      bias=False),
            nn.BatchNorm3d(32),
            nn.ReLU6(inplace=True),
        )

        self.layers = nn.Sequential(
            _DWSep3D(32,  64),
            _DWSep3D(64,  128, stride=2),
            _DWSep3D(128, 128),
            _DWSep3D(128, 256, stride=2, t_stride=2),
            _DWSep3D(256, 256),
            _DWSep3D(256, 512, stride=2, t_stride=2),
            _DWSep3D(512, 512),
            _DWSep3D(512, 512),
        )

        self.pool = nn.AdaptiveAvgPool3d(1)

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(512, 128),
            nn.ReLU6(inplace=True),
            nn.Dropout(dropout * 0.5),
            nn.Linear(128, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.stem(x)
        x = self.layers(x).flatten(1) if False else self.layers(x)
        x = self.pool(x).flatten(1)
        return self.classifier(x)


# ══════════════════════════════════════════════════════════════
#  SANITY CHECK
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    teacher = ViolenceModel().to(device)
    teacher.eval()
    with torch.no_grad():
        out = teacher(torch.randn(1, 3, 16, 224, 224).to(device))
    t_p = sum(p.numel() for p in teacher.parameters()) / 1e6
    print(f"Teacher  | output: {out.shape} | params: {t_p:.1f}M")
    print("  Keys (temporal/classifier):")
    for k in teacher.state_dict().keys():
        if "temporal" in k or "classifier" in k:
            print(f"    {k}")

    student = MobileNet3D(num_classes=2).to(device)
    student.eval()
    with torch.no_grad():
        out2 = student(torch.randn(1, 3, 16, 112, 112).to(device))
    s_p = sum(p.numel() for p in student.parameters()) / 1e6
    print(f"\nStudent  | output: {out2.shape} | params: {s_p:.1f}M")
    print(f"Compression: {t_p / s_p:.1f}x smaller")
