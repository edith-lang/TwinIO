"""Causal CNN-LSTM body-frame velocity network (Sec. IV-B, Fig. 4)."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalDWBlock(nn.Module):
    """Causal depthwise conv (k, dilation d) -> 1x1 conv -> GELU."""

    def __init__(self, ch: int, k: int, d: int):
        super().__init__()
        self.pad = (k - 1) * d
        self.dw = nn.Conv1d(ch, ch, k, dilation=d, groups=ch)
        self.pw = nn.Conv1d(ch, ch, 1)

    def forward(self, x):
        return F.gelu(self.pw(self.dw(F.pad(x, (self.pad, 0)))))


class VelocityNet(nn.Module):
    def __init__(self, hidden: int = 64, feat: int = 128, k: int = 5, dilations=(1, 2, 4)):
        super().__init__()
        # input normalisation lives inside the model so Jacobians are w.r.t. raw units
        self.register_buffer("mu", torch.zeros(6))
        self.register_buffer("sigma", torch.ones(6))
        self.inp = nn.Conv1d(6, hidden, 1)
        self.blocks = nn.Sequential(*[CausalDWBlock(hidden, k, d) for d in dilations])
        self.out = nn.Conv1d(hidden, feat, 1)
        self.lstm = nn.LSTM(feat, feat, batch_first=True)
        self.norm = nn.LayerNorm(feat)
        self.head = nn.Linear(feat, 3)
        self.receptive_field = 1 + (k - 1) * sum(dilations)

    def forward(self, u):
        """u: B x T x 6 raw IMU [gyro, acc] -> B x T x 3 body velocity."""
        x = ((u - self.mu) / self.sigma).transpose(1, 2)
        x = self.out(self.blocks(self.inp(x))).transpose(1, 2)
        h, _ = self.lstm(x)                    # state reset every window
        return self.head(self.norm(h))


def n_params(m):
    return sum(p.numel() for p in m.parameters())
