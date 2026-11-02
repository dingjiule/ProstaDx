#!/usr/bin/env python3
"""CNN settings shared by training, dataset loading, and model construction."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

VERSION = "1.2.0"
CHANNELS = ("T2WI", "DWI", "ADC", "DCE +0s", "DCE +10s", "DCE +100s")


class ReleaseError(Exception):
    def __init__(self, code):
        if not re.fullmatch("[A-Z][A-Z0-9_]+", code):
            code = "PROCESSING_FAILED"
        self.code = code
        super().__init__(code)


def require(condition, code):
    if not condition:
        raise ReleaseError(code)


@dataclass(frozen=True)
class TrainConfig:
    size: tuple = (128, 128, 32)
    base_channels: int = 16
    static_dropout: float = 0.4
    fusion_dropout: float = 0.3
    epochs: int = 200
    batch_size: int = 8
    learning_rate: float = 5e-05
    weight_decay: float = 0.003
    patience: int = 20
    flip_probability: float = 0.3
    seed: int | None = None
    workers: int = 2
    threads: int = 1
    amp: bool = True

    def validate(self):
        require(
            len(self.size) == 3
            and all(isinstance(v, int) and v > 0 for v in self.size),
            "INVALID_TENSOR_SIZE",
        )
        final = (
            math.ceil(self.size[0] / 32)
            * math.ceil(self.size[1] / 32)
            * math.ceil(self.size[2] / 8)
        )
        require(final > 1, "TENSOR_TOO_SMALL_FOR_INSTANCE_NORM")
        require(
            all(
                v > 0
                for v in (
                    self.base_channels,
                    self.epochs,
                    self.batch_size,
                    self.patience,
                    self.threads,
                )
            ),
            "INVALID_TRAINING_CONFIG",
        )
        require(
            self.workers >= 0
            and (
                self.seed is None
                or (isinstance(self.seed, int) and 0 <= self.seed < 2**32)
            ),
            "INVALID_TRAINING_CONFIG",
        )
        require(
            math.isfinite(self.learning_rate)
            and self.learning_rate > 0
            and math.isfinite(self.weight_decay)
            and (self.weight_decay >= 0),
            "INVALID_TRAINING_CONFIG",
        )
        require(
            0 <= self.flip_probability <= 1
            and 0 <= self.static_dropout < 1
            and (0 <= self.fusion_dropout < 1),
            "INVALID_TRAINING_CONFIG",
        )
