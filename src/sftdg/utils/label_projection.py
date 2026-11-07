"""Implements Algorithm 2 (KL divergence minimization) from the SFT paper."""

from __future__ import annotations

from typing import Tuple

import torch


def project_distribution(
    probs: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 1.5,
) -> torch.Tensor:
    """Project refiner output to the constrained label simplex.

    Parameters
    ----------
    probs : torch.Tensor
        Softmax probabilities from the refiner (shape: [B, C]).
    targets : torch.Tensor
        Ground-truth class indices (shape: [B]).
    alpha : float
        Minimum ratio between correct-class probability and any other class.

    Returns
    -------
    torch.Tensor
        Projected probability distributions (shape: [B, C]).
    """
    if probs.dim() != 2:
        raise ValueError("Expected probs to be of shape [B, C].")

    batch, num_classes = probs.shape
    device = probs.device
    projected = torch.zeros_like(probs)

    for b in range(batch):
        p = probs[b]
        y = targets[b].item()

        perm = [y] + [k for k in range(num_classes) if k != y]
        inv_perm = torch.empty(num_classes, dtype=torch.long, device=device)
        for idx, orig in enumerate(perm):
            inv_perm[idx] = orig
        p_perm = p[perm]

        p1 = p_perm[0]
        violating = [
            j for j in range(1, num_classes) if alpha * p_perm[j] > p1
        ]
        violating.sort(key=lambda j: p_perm[j], reverse=True)

        active: list[int] = []
        for j in violating:
            denom = 1.0
            for idx in active:
                denom *= (alpha * p_perm[idx]) ** (1.0 / (len(active) + alpha))
            numerator = p1 ** (1.0 / (len(active) + alpha))
            threshold = numerator / (denom + 1e-12)
            if threshold < alpha * p_perm[j]:
                active.append(j)
            else:
                break

        exp_factor = 1.0 / (len(active) + alpha)
        q1 = p1 ** (alpha * exp_factor)
        for idx in active:
            q1 /= (alpha * p_perm[idx]) ** exp_factor

        q_perm = p_perm.clone()
        q_perm[0] = q1
        for idx in range(1, num_classes):
            if idx in active:
                q_perm[idx] = q1 / alpha
            else:
                q_perm[idx] = p_perm[idx]

        q_perm = q_perm / q_perm.sum()
        restored = torch.empty_like(q_perm)
        for idx, orig in enumerate(perm):
            restored[orig] = q_perm[idx]
        projected[b] = restored

    return projected
