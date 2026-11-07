"""FlatLand-style curvature-alignment algorithm for domain generalization."""

from __future__ import annotations

from typing import List, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from domainbed import networks

from .base import Algorithm
from .registry import register_algorithm
from sftdg.models import build_featurizer


@register_algorithm("flatland")
class FlatLandLorentz(Algorithm):
    """Implements curvature-aligned federated domain generalization.

    This adapts the FlatLand / Tailored Lorentz Space idea to DomainBed by
    learning per-domain curvature parameters and aligning their tangent
    representations in a shared Lorentz manifold.
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super().__init__(input_shape, num_classes, num_domains, hparams)

        self.num_domains = num_domains
        self.featurizer = build_featurizer(input_shape, hparams)
        self.classifier = networks.Classifier(
            self.featurizer.n_outputs,
            num_classes,
            hparams.get("nonlinear_classifier", False),
        )
        self.network = nn.Sequential(self.featurizer, self.classifier)

        self.log_curvatures = nn.Parameter(torch.zeros(num_domains))

        self.grad_clip = hparams.get("grad_clip_norm", 5.0)
        self.alpha_curv = hparams.get("alpha_curv", 0.1)
        self.beta_align = hparams.get("beta_align", 0.1)
        self.gamma_sharp = hparams.get("gamma_sharp", 0.0)
        self.eps = 1e-8

        self.optimizer_main = torch.optim.Adam(
            self.network.parameters(),
            lr=hparams["lr"],
            weight_decay=hparams["weight_decay"],
        )
        self.optimizer_curv = torch.optim.Adam(
            [self.log_curvatures],
            lr=hparams.get("curvature_lr", hparams["lr"]),
            weight_decay=0.0,
        )

    def _curvatures(self) -> torch.Tensor:
        return torch.nn.functional.softplus(self.log_curvatures) + 1e-4

    def _tangent_embedding(self, feats: torch.Tensor, curvature: torch.Tensor) -> torch.Tensor:
        """Project features to a curvature-aware tangent representation."""
        scale = torch.sqrt(curvature).unsqueeze(0)
        return feats / (scale + self.eps)

    def _alignment_loss(self, means: List[torch.Tensor], weights: torch.Tensor) -> torch.Tensor:
        stacked = torch.stack(means)
        weighted_mean = (stacked * weights[:, None]).sum(dim=0)
        align = ((stacked - weighted_mean) ** 2).sum(dim=1).mean()
        return align

    def _sharpness_proxy(self, tangent_norms: List[torch.Tensor], curvatures: torch.Tensor) -> torch.Tensor:
        if not tangent_norms:
            return torch.tensor(0.0, device=curvatures.device)
        losses = []
        for norms, K in zip(tangent_norms, curvatures):
            if norms.numel() < 2:
                continue
            losses.append(norms.var(unbiased=False) * K)
        if not losses:
            return torch.tensor(0.0, device=curvatures.device)
        return torch.stack(losses).mean()

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

    def update(
        self,
        minibatches: Sequence[Tuple[torch.Tensor, torch.Tensor]],
        unlabeled=None,
    ) -> dict:
        device = next(self.network.parameters()).device
        batches = [(x.to(device), y.to(device)) for x, y in minibatches]

        curvatures = self._curvatures()
        sup_losses = []
        tangent_means: List[torch.Tensor] = []
        tangent_norms: List[torch.Tensor] = []
        weights = []

        for env_idx, (x, y) in enumerate(batches):
            feats = self.featurizer(x)
            logits = self.classifier(feats)
            sup_loss = F.cross_entropy(logits, y)
            sup_losses.append(sup_loss)

            tangent = self._tangent_embedding(feats, curvatures[env_idx])
            tangent_means.append(tangent.mean(dim=0))
            tangent_norms.append(tangent.norm(dim=1))
            weights.append(y.size(0))

        sup_loss_total = torch.stack(sup_losses).mean()
        weight_tensor = torch.tensor(weights, device=device, dtype=torch.float32)
        weight_tensor = weight_tensor / (weight_tensor.sum() + self.eps)

        align_loss = self._alignment_loss(tangent_means, weight_tensor)

        curv_matrix = (curvatures.unsqueeze(0) - curvatures.unsqueeze(1)) ** 2
        curv_loss = curv_matrix.mean()

        sharp_loss = self._sharpness_proxy(tangent_norms, curvatures)

        total_loss = (
            sup_loss_total
            + self.alpha_curv * curv_loss
            + self.beta_align * align_loss
            + self.gamma_sharp * sharp_loss
        )

        self.optimizer_main.zero_grad()
        self.optimizer_curv.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.grad_clip)
        torch.nn.utils.clip_grad_norm_([self.log_curvatures], self.grad_clip)
        self.optimizer_main.step()
        self.optimizer_curv.step()

        with torch.no_grad():
            total_correct = 0
            total_seen = 0
            for x, y in batches:
                preds = self.network(x)
                total_correct += (preds.argmax(dim=1) == y).sum().item()
                total_seen += y.size(0)
            train_acc = total_correct / max(total_seen, 1)

        metrics = {
            "loss_sup": sup_loss_total.item(),
            "loss_curv": curv_loss.item(),
            "loss_align": align_loss.item(),
            "loss_sharp": sharp_loss.item(),
            "loss_total": total_loss.item(),
            "train_acc": train_acc,
            "mean_curvature": curvatures.mean().item(),
        }
        return metrics
