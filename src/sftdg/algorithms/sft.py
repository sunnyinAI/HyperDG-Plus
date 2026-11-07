"""Self-Feedback Training (SFT) implementation."""

from __future__ import annotations

from typing import Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from domainbed import networks

from .base import Algorithm
from .registry import register_algorithm
from sftdg.models import LabelRefiner, build_featurizer
from sftdg.utils import project_distribution


def _grad_norm(params) -> torch.Tensor:
    norms = []
    device = None
    for p in params:
        if p.grad is None:
            continue
        if device is None:
            device = p.grad.device
        norms.append(p.grad.norm(p=2))
    if not norms:
        return torch.tensor(0.0, device=device or torch.device("cpu"))
    return torch.norm(torch.stack(norms))


def _soft_cross_entropy(
    logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-8
) -> torch.Tensor:
    log_probs = torch.log_softmax(logits, dim=1)
    loss = -(targets * log_probs).sum(dim=1).mean()
    return loss


@register_algorithm("sft")
class SFTAlgorithm(Algorithm):
    """Approximate Algorithm 1 from the SFT paper."""

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super().__init__(input_shape, num_classes, num_domains, hparams)

        self.featurizer = build_featurizer(input_shape, hparams)
        self.classifier = networks.Classifier(
            self.featurizer.n_outputs,
            num_classes,
            hparams.get("nonlinear_classifier", False),
        )
        self.network = nn.Sequential(self.featurizer, self.classifier)

        self.rho = hparams.get("rho", 0.05)
        self.alpha = hparams.get("alpha", 1.5)
        self.lambda1 = hparams.get("lambda1", 0.5)
        self.lambda2 = hparams.get("lambda2", 0.5)
        self.label_mode = hparams.get("label_mode", "refined").lower()
        self.label_smoothing = float(hparams.get("label_smoothing", 0.0))
        valid_modes = {"one_hot", "smoothed", "refined"}
        if self.label_mode not in valid_modes:
            raise ValueError(
                f"Unsupported label_mode '{self.label_mode}'. "
                f"Expected one of {sorted(valid_modes)}"
            )
        self.enable_refiner = self.label_mode == "refined"
        self.use_pce = bool(hparams.get("use_pce", True))
        self.grad_clip = hparams.get("grad_clip_norm", 5.0)
        hidden_dim = hparams.get("refiner_hidden_dim", 256)
        hidden_layers = hparams.get("refiner_layers", 2)
        refiner_dropout = hparams.get("refiner_dropout", 0.1)
        hidden_dims = [hidden_dim for _ in range(hidden_layers)]

        self.refiner = LabelRefiner(
            input_dim=num_classes,
            num_classes=num_classes,
            hidden_dims=hidden_dims,
            dropout=refiner_dropout,
        )

        self.optimizer_main = torch.optim.Adam(
            self.network.parameters(),
            lr=hparams["lr"],
            weight_decay=hparams["weight_decay"],
        )
        self.optimizer_refiner = torch.optim.Adam(
            self.refiner.parameters(),
            lr=hparams.get("refiner_lr", hparams["lr"]),
            weight_decay=hparams.get("refiner_weight_decay", 0.0),
        )

        self.eps = 1e-8

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

    def _one_hot(self, labels: torch.Tensor) -> torch.Tensor:
        return F.one_hot(labels, num_classes=self.num_classes).to(dtype=torch.float32)

    def _smoothed_targets(self, labels: torch.Tensor) -> torch.Tensor:
        if self.label_smoothing <= 0.0:
            return self._one_hot(labels)
        smooth = self.label_smoothing / max(self.num_classes, 1)
        return self._one_hot(labels) * (1.0 - self.label_smoothing) + smooth

    def _generate_soft_targets(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Generate projected soft labels (stop-gradient for model step)."""
        if self.label_mode == "one_hot":
            return self._one_hot(labels)
        if self.label_mode == "smoothed":
            return self._smoothed_targets(labels)
        if not self.enable_refiner:
            raise RuntimeError("Refiner disabled but refined labels requested.")
        with torch.no_grad():
            ref_logits = self.refiner(logits)
            probs = torch.softmax(ref_logits, dim=1)
            soft = project_distribution(probs, labels, self.alpha)
        return soft

    def _sam_step(self, x: torch.Tensor, y: torch.Tensor, step: bool = True):
        logits = self.network(x)
        soft_targets = self._generate_soft_targets(logits.detach(), y)
        loss_clean = _soft_cross_entropy(logits, soft_targets, self.eps)

        self.optimizer_main.zero_grad()
        loss_clean.backward()
        grad_norm = _grad_norm(list(self.network.parameters()))
        scale = self.rho / (grad_norm + self.eps)

        perturbations = []
        with torch.no_grad():
            for p in self.network.parameters():
                if p.grad is None:
                    perturbations.append(None)
                    continue
                e = p.grad * scale
                p.add_(e)
                perturbations.append(e)

        self.optimizer_main.zero_grad()
        loss_sharp = _soft_cross_entropy(self.network(x), soft_targets, self.eps)
        if step:
            loss_sharp.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.grad_clip)
            self.optimizer_main.step()
        else:
            loss_sharp.backward()
            self.optimizer_main.zero_grad()

        with torch.no_grad():
            for p, e in zip(self.network.parameters(), perturbations):
                if e is not None:
                    p.sub_(e)

        self.optimizer_main.zero_grad()
        sharpness = (loss_sharp - loss_clean).detach()
        return loss_clean.detach(), loss_sharp.detach(), sharpness, soft_targets

    def _refiner_update(
        self,
        logits_a: torch.Tensor,
        labels_a: torch.Tensor,
        logits_b: torch.Tensor,
        labels_b: torch.Tensor,
        sharpness_gap: torch.Tensor,
    ) -> torch.Tensor:
        if not self.enable_refiner:
            return torch.tensor(0.0, device=logits_a.device)

        logits_a = logits_a.detach()
        logits_b = logits_b.detach()

        probs_a = torch.softmax(self.refiner(logits_a), dim=1)
        probs_b = torch.softmax(self.refiner(logits_b), dim=1)

        proj_a = project_distribution(probs_a.detach(), labels_a, self.alpha)
        proj_b = project_distribution(probs_b.detach(), labels_b, self.alpha)

        if self.use_pce:
            lpce_a = -(proj_a * torch.log(probs_a + self.eps)).sum(dim=1).mean()
            lpce_b = -(proj_b * torch.log(probs_b + self.eps)).sum(dim=1).mean()
            lpce = 0.5 * (lpce_a + lpce_b)
        else:
            lpce = torch.tensor(0.0, device=logits_a.device)

        ce_a = -(proj_a * torch.log_softmax(logits_a, dim=1)).sum(dim=1).mean()
        ce_b = -(proj_b * torch.log_softmax(logits_b, dim=1)).sum(dim=1).mean()
        ce_loss = 0.5 * (ce_a + ce_b)

        refine_loss = lpce + self.lambda1 * ce_loss + self.lambda2 * sharpness_gap.detach()

        self.optimizer_refiner.zero_grad()
        refine_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.refiner.parameters(), self.grad_clip)
        self.optimizer_refiner.step()

        return refine_loss.detach()

    def update(
        self,
        minibatches: Sequence[Tuple[torch.Tensor, torch.Tensor]],
        unlabeled=None,
    ) -> dict:
        device = next(self.network.parameters()).device
        batches = [(x.to(device), y.to(device)) for x, y in minibatches]
        n_envs = len(batches)
        if n_envs < 1:
            raise ValueError("SFT requires at least one source domain minibatch.")

        perm = torch.randperm(n_envs)
        idx_a = perm[0].item()
        idx_b = perm[1].item() if n_envs > 1 else perm[0].item()

        xa, ya = batches[idx_a]
        xb, yb = batches[idx_b]

        loss_clean_a, loss_sharp_a, sharpness_a, _ = self._sam_step(xa, ya, step=True)
        with torch.no_grad():
            logits_a_detach = self.network(xa)

        loss_clean_b, loss_sharp_b, sharpness_b, _ = self._sam_step(xb, yb, step=False)
        with torch.no_grad():
            logits_b_detach = self.network(xb)

        sharp_gap = torch.abs(sharpness_a - sharpness_b)
        if self.enable_refiner:
            refine_loss = self._refiner_update(
                logits_a_detach, ya, logits_b_detach, yb, sharp_gap
            )
        else:
            refine_loss = torch.tensor(0.0, device=device)

        with torch.no_grad():
            total_correct = 0
            total = 0
            for x, y in batches:
                preds = self.network(x)
                total_correct += (preds.argmax(dim=1) == y).sum().item()
                total += y.size(0)
            train_acc = total_correct / max(total, 1)

        metrics = {
            "loss_clean": loss_clean_a.item(),
            "loss_sharp": loss_sharp_a.item(),
            "sharpness_a": sharpness_a.item(),
            "sharpness_b": sharpness_b.item(),
            "sharpness_gap": sharp_gap.item(),
            "refine_loss": refine_loss.item(),
            "train_acc": train_acc,
            "active_env": idx_a,
            "feedback_env": idx_b,
        }
        return metrics
