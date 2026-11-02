#!/usr/bin/env python3
"""Residual 3D CNN for three-channel MRI and six-channel early fusion."""

from __future__ import annotations

import torch
from cnn_config import require
from torch import nn


class ConvBlock3D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride=1, dropout: float = 0.0):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(
                in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False
            ),
            nn.InstanceNorm3d(out_ch, affine=True),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm3d(out_ch, affine=True),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Dropout3d(p=dropout) if dropout and dropout > 0 else nn.Identity(),
        )
        self.shortcut = nn.Identity()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.InstanceNorm3d(out_ch, affine=True),
            )

    def forward(self, x):
        return self.block(x) + self.shortcut(x)


class FeatureEncoder3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        base_channels: int,
        dropout: float,
        stem_kernel_z: int,
        z_stride_pattern,
    ):
        super().__init__()
        c = int(base_channels)
        stem_kernel_z = int(stem_kernel_z)
        z_stride_pattern = tuple(int(v) for v in z_stride_pattern)
        if stem_kernel_z not in (3, 5, 7):
            raise ValueError(
                f"stem_kernel_z must be one of 3, 5, 7; got {stem_kernel_z}."
            )
        if len(z_stride_pattern) != 5:
            raise ValueError(
                "z_stride_pattern must contain five values: stem, layer1, layer2, layer3, layer4."
            )
        if any(v not in (1, 2) for v in z_stride_pattern):
            raise ValueError(f"Each Z stride must be 1 or 2; got {z_stride_pattern}.")
        stem_z, z1, z2, z3, z4 = z_stride_pattern
        self.out_channels = c * 16
        self.stem = nn.Sequential(
            nn.Conv3d(
                in_channels,
                c,
                kernel_size=(7, 7, stem_kernel_z),
                stride=(2, 2, stem_z),
                padding=(3, 3, stem_kernel_z // 2),
                bias=False,
            ),
            nn.InstanceNorm3d(c, affine=True),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
        )
        self.layer1 = ConvBlock3D(c, c * 2, stride=(2, 2, z1), dropout=dropout)
        self.layer2 = ConvBlock3D(c * 2, c * 4, stride=(2, 2, z2), dropout=dropout)
        self.layer3 = ConvBlock3D(c * 4, c * 8, stride=(2, 2, z3), dropout=dropout)
        self.layer4 = ConvBlock3D(c * 8, c * 16, stride=(2, 2, z4), dropout=dropout)
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.flatten = nn.Flatten()

    def forward_spatial(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def pool_spatial(self, x):
        return self.flatten(self.pool(x))

    def forward(self, x):
        return self.pool_spatial(self.forward_spatial(x))


class MLPHead(nn.Module):
    def __init__(
        self, in_features: int, hidden_features: int, num_classes: int, dropout: float
    ):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_features, num_classes),
        )

    def forward(self, x):
        return self.head(x)


class SingleBranch3DCNN(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        base_channels: int,
        dropout: float,
        stem_kernel_z: int,
        z_stride_pattern,
    ):
        super().__init__()
        c = int(base_channels)
        self.encoder = FeatureEncoder3D(
            in_channels=in_channels,
            base_channels=c,
            dropout=dropout,
            stem_kernel_z=stem_kernel_z,
            z_stride_pattern=z_stride_pattern,
        )
        self.classifier = MLPHead(
            in_features=self.encoder.out_channels,
            hidden_features=c * 8,
            num_classes=num_classes,
            dropout=dropout,
        )
        init_weights(self)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        feature = self.encoder(x)
        logits = self.classifier(feature)
        return {"logits": logits, "feature": feature}


def init_weights(module: nn.Module):
    for submodule in module.modules():
        if isinstance(submodule, nn.Conv3d):
            nn.init.kaiming_normal_(submodule.weight, nonlinearity="leaky_relu")
        elif isinstance(submodule, nn.Linear):
            nn.init.xavier_uniform_(submodule.weight)
            if submodule.bias is not None:
                nn.init.zeros_(submodule.bias)


def model_parameters(stage, config):
    require(stage in ("static", "fusion"), "INVALID_MODEL")
    return dict(
        in_channels=3 if stage == "static" else 6,
        num_classes=2,
        base_channels=config.base_channels,
        dropout=config.static_dropout if stage == "static" else config.fusion_dropout,
        stem_kernel_z=5,
        z_stride_pattern=(1, 1, 2, 2, 2),
    )
