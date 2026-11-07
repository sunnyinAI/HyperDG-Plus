"""Base class for domain generalization algorithms."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable, Sequence, Tuple

import torch


class Algorithm(ABC, torch.nn.Module):
    """Abstract base for algorithms.

    Combines `torch.nn.Module` to leverage parameter management and checkpointing.
    """

    def __init__(self) -> None:
        super().__init__()

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super().__init__()
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.num_domains = num_domains
        self.hparams = hparams

    @abstractmethod
    def update(
        self,
        minibatches: Sequence[Tuple[torch.Tensor, torch.Tensor]],
        unlabeled: Iterable[torch.Tensor] | None = None,
    ) -> dict:
        """Perform a single optimization step across domains."""

    @abstractmethod
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Return logits for evaluation."""
