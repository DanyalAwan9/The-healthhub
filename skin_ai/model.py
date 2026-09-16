"""
model.py - multi-task skin-analysis network.

  EfficientNet-B0 or ResNet-50 transfer-learning backbone
  + an optional 4th input channel carrying 20-px local contrast (texture/blemish)
  + five task heads:
        skin_type          4-way  Normal / Dry / Oily / Combination
        redness            4-way  clear / mild / moderate / severe
        hyperpigmentation  4-way  clear / mild / moderate / severe
        blemish            1  regression, trained in log1p(count) space
        texture            1  regression, 0 (smooth) .. 1 (rough), sigmoid at read-out

Public API
----------
  build_model(arch="efficientnet_b0", pretrained=True, contrast_channel=True) -> SkinAnalysisNet
  SkinLoss()                     -> (total_loss, {per_head_loss})   NaN / -1 targets are ignored
  postprocess(raw_out, temperature=1.0) -> friendly probs / counts
  local_contrast_map(x, radius=20)      -> B,1,H,W   (differentiable local sigma)
  SKIN_TYPES, SEVERITY
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

SKIN_TYPES = ["Normal", "Dry", "Oily", "Combination"]
SEVERITY = ["clear", "mild", "moderate", "severe"]
CLASS_HEADS = ("skin_type", "redness", "hyperpigmentation")
REG_HEADS = ("blemish", "texture")


# --------------------------------------------------------- contrast-aware analysis
def local_contrast_map(x: torch.Tensor, radius: int = 20) -> torch.Tensor:
    """Per-pixel local standard deviation of luminance over a (2r+1) window."""
    w = x.new_tensor([0.299, 0.587, 0.114])[: x.shape[1]].view(1, -1, 1, 1)
    g = (x * w).sum(1, keepdim=True)
    k = int(radius) * 2 + 1
    mean = F.avg_pool2d(g, k, 1, radius, count_include_pad=False)
    sq = F.avg_pool2d(g * g, k, 1, radius, count_include_pad=False)
    std = (sq - mean * mean).clamp_min(1e-6).sqrt()
    return (std / 0.5).clamp(0.0, 1.0)


def _inflate_first_conv(conv: nn.Conv2d, extra: int = 1) -> nn.Conv2d:
    """Widen a pretrained first conv by `extra` input channels (zero-initialised)."""
    new = nn.Conv2d(conv.in_channels + extra, conv.out_channels,
                    kernel_size=conv.kernel_size, stride=conv.stride,
                    padding=conv.padding, dilation=conv.dilation,
                    groups=conv.groups, bias=conv.bias is not None)
    with torch.no_grad():
        new.weight[:, : conv.in_channels].copy_(conv.weight)
        new.weight[:, conv.in_channels:].zero_()
        if conv.bias is not None:
            new.bias.copy_(conv.bias)
    return new


def _head(d_in: int, d_out: int, p: float = 0.4, hidden: int = 256) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(d_in, hidden), nn.LayerNorm(hidden), nn.ReLU(inplace=True),
        nn.Dropout(p), nn.Linear(hidden, d_out),
    )


class SkinAnalysisNet(nn.Module):
    def __init__(self, arch: str = "efficientnet_b0", pretrained: bool = True,
                 contrast_channel: bool = True, dropout: float = 0.4, radius: int = 20):
        super().__init__()
        self.arch = arch
        self.contrast_channel = bool(contrast_channel)
        self.radius = int(radius)
        extra = 1 if self.contrast_channel else 0

        if arch == "efficientnet_b0":
            weights = None
            if pretrained:
                try:
                    weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
                except Exception:
                    weights = None
            m = models.efficientnet_b0(weights=weights)
            if extra:
                m.features[0][0] = _inflate_first_conv(m.features[0][0], extra)
            self.features = m.features
            feat_dim = m.classifier[1].in_features            # 1280
        elif arch == "resnet50":
            weights = None
            if pretrained:
                try:
                    weights = models.ResNet50_Weights.IMAGENET1K_V2
                except Exception:
                    weights = None
            m = models.resnet50(weights=weights)
            if extra:
                m.conv1 = _inflate_first_conv(m.conv1, extra)
            self.features = nn.Sequential(
                m.conv1, m.bn1, m.relu, m.maxpool,
                m.layer1, m.layer2, m.layer3, m.layer4,
            )
            feat_dim = m.fc.in_features                        # 2048
        else:
            raise ValueError(f"unknown arch {arch!r} (efficientnet_b0 | resnet50)")

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.feat_dim = feat_dim
        self.heads = nn.ModuleDict({
            "skin_type": _head(feat_dim, 4, dropout),
            "redness": _head(feat_dim, 4, dropout),
            "hyperpigmentation": _head(feat_dim, 4, dropout),
            "blemish": _head(feat_dim, 1, dropout),
            "texture": _head(feat_dim, 1, dropout),
        })

    def freeze_backbone(self, frozen: bool = True) -> None:
        for p in self.features.parameters():
            p.requires_grad = not frozen

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        if self.contrast_channel:
            x = torch.cat([x, local_contrast_map(x, self.radius)], dim=1)
        return torch.flatten(self.pool(self.features(x)), 1)

    def forward(self, x: torch.Tensor) -> dict:
        f = self.forward_features(x)
        return {k: h(f) for k, h in self.heads.items()}


class SkinLoss(nn.Module):
    """Weighted multi-task loss. Missing labels (class == -1, reg == NaN) are skipped."""

    def __init__(self, weights: dict | None = None):
        super().__init__()
        self.w = weights or {"skin_type": 1.0, "redness": 0.7,
                             "hyperpigmentation": 0.7, "blemish": 0.5, "texture": 0.3}
        self.ce = nn.CrossEntropyLoss(reduction="none")
        self.reg = nn.SmoothL1Loss(reduction="none")

    def forward(self, out: dict, tgt: dict):
        parts, total = {}, out["skin_type"].new_zeros(())
        for k in CLASS_HEADS:
            y = tgt[k].long()
            mask = y >= 0
            if mask.any():
                loss = self.ce(out[k][mask], y[mask]).mean()
            else:
                loss = out[k].new_zeros(())
            parts[k] = loss
            total = total + self.w[k] * loss

        b = tgt["blemish"].float()
        mb = ~torch.isnan(b)
        if mb.any():
            pred = out["blemish"].squeeze(1)[mb]
            lb = self.reg(pred, torch.log1p(b[mb].clamp(min=0))).mean()
        else:
            lb = out["blemish"].new_zeros(())
        parts["blemish"] = lb
        total = total + self.w["blemish"] * lb

        t = tgt["texture"].float()
        mt = ~torch.isnan(t)
        if mt.any():
            pred = torch.sigmoid(out["texture"].squeeze(1)[mt])
            lt = self.reg(pred, t[mt].clamp(0, 1)).mean()
        else:
            lt = out["texture"].new_zeros(())
        parts["texture"] = lt
        total = total + self.w["texture"] * lt

        return total, {k: float(v.detach()) for k, v in parts.items()}


@torch.no_grad()
def postprocess(out: dict, temperature: float = 1.0) -> dict:
    t = max(float(temperature), 0.05)
    return {
        "skin_type": torch.softmax(out["skin_type"] / t, dim=1),
        "redness": torch.softmax(out["redness"], dim=1),
        "hyperpigmentation": torch.softmax(out["hyperpigmentation"], dim=1),
        "blemish_count": torch.expm1(out["blemish"].squeeze(1)).clamp(min=0.0),
        "texture": torch.sigmoid(out["texture"].squeeze(1)),
    }


def build_model(arch: str = "efficientnet_b0", pretrained: bool = True,
                contrast_channel: bool = True, dropout: float = 0.4,
                radius: int = 20) -> SkinAnalysisNet:
    return SkinAnalysisNet(arch, pretrained, contrast_channel, dropout, radius)
