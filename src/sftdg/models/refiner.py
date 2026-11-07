"""Landscape refiner network producing soft labels."""

from __future__ import annotations

from typing import Sequence

import torch
from torch import nn


class LabelRefiner(nn.Module):
    """Lightweight MLP that maps logits/features to soft-label corrections."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dims: Sequence[int] = (256, 256),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = input_dim
        for width in hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev, width),
                    nn.ReLU(inplace=True),
                    nn.Dropout(dropout),
                ]
            )
            prev = width
        layers.append(nn.Linear(prev, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """Return unnormalized logits for the soft-label distribution."""
        return self.net(logits)
