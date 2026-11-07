"""HyperDG: Curvature-refined federated domain generalization."""

from __future__ import annotations

from typing import List, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from domainbed import networks

from .base import Algorithm
from .registry import register_algorithm
from sftdg.models import build_featurizer


def _softplus_curvature(params: torch.Tensor, eps: float) -> torch.Tensor:
    return torch.nn.functional.softplus(params) + eps


@register_algorithm("hyperdg")
class HyperDG(Algorithm):
    """Implements curvature alignment + flatness regularisation."""

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super().__init__(input_shape, num_classes, num_domains, hparams)

        self.featurizer = build_featurizer(input_shape, hparams)
        feat_dim = self.featurizer.n_outputs
        self.classifier = networks.Classifier(
            feat_dim,
            num_classes,
            hparams.get("nonlinear_classifier", False),
        )

        self.num_domains = num_domains
        self.curv_eps = hparams.get("curvature_epsilon", 1e-4)
        self.grad_clip = hparams.get("grad_clip_norm", 5.0)
        self.alpha_curv = hparams.get("alpha_curv", 0.2)
        self.beta_align = hparams.get("beta_align", 0.5)
        self.gamma_sharp = hparams.get("gamma_sharp", 0.05)
        self.delta_local = hparams.get("delta_local_flat", 0.1)
        self.delta_global = hparams.get("delta_global_flat", 0.1)
        curvature_lr = hparams.get("curvature_lr", hparams["lr"])

        # single_curvature=True collapses all per-domain α_i into one shared α
        self.single_curvature = bool(hparams.get("single_curvature", False))
        if self.single_curvature:
            self.log_curvatures = nn.Parameter(torch.zeros(1))
        else:
            self.log_curvatures = nn.Parameter(torch.zeros(num_domains))
        self.log_global_curvature = nn.Parameter(torch.tensor(0.0))

        self.optimizer_main = torch.optim.Adam(
            list(self.network_parameters()),
            lr=hparams["lr"],
            weight_decay=hparams["weight_decay"],
        )
        self.optimizer_curv = torch.optim.Adam(
            [self.log_curvatures, self.log_global_curvature],
            lr=curvature_lr,
            weight_decay=0.0,
        )

    def network_parameters(self):
        yield from self.featurizer.parameters()
        yield from self.classifier.parameters()

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.featurizer(x)
        return self.classifier(feats)

    def _tangent_embedding(self, feats: torch.Tensor, curvature: torch.Tensor) -> torch.Tensor:
        scale = torch.sqrt(curvature).clamp(min=self.curv_eps)
        return feats / scale

    def _alignment_loss(self, means: List[torch.Tensor], weights: torch.Tensor) -> torch.Tensor:
        stacked = torch.stack(means)
        weighted_mean = (stacked * weights[:, None]).sum(dim=0)
        return ((stacked - weighted_mean) ** 2).sum(dim=1).mean()

    def _sharpness_proxy(self, tangent_norms: List[torch.Tensor], curvatures: torch.Tensor) -> torch.Tensor:
        losses = []
        for norms, K in zip(tangent_norms, curvatures):
            if norms.numel() < 2:
                continue
            losses.append(norms.var(unbiased=False) * K)
        if not losses:
            return torch.tensor(0.0, device=curvatures.device)
        return torch.stack(losses).mean()

    def update(
        self,
        minibatches: Sequence[Tuple[torch.Tensor, torch.Tensor]],
        unlabeled=None,
    ) -> dict:
        device = next(self.featurizer.parameters()).device
        batches = [(x.to(device), y.to(device)) for x, y in minibatches]
        if not batches:
            raise ValueError("HyperDG requires at least one minibatch.")

        curvatures = _softplus_curvature(self.log_curvatures, self.curv_eps)
        global_curv = _softplus_curvature(self.log_global_curvature, self.curv_eps)

        sup_losses = []
        global_losses = []
        tangent_means: List[torch.Tensor] = []
        tangent_norms: List[torch.Tensor] = []
        local_grad_terms = []
        weights = []

        for env_idx, (x, y) in enumerate(batches):
            feats = self.featurizer(x)

            # For single_curvature, all domains share curvatures[0]
            curv_param = self.log_curvatures[0] if self.single_curvature else self.log_curvatures[env_idx]
            curv_val = curvatures[0] if self.single_curvature else curvatures[env_idx]

            tangent_local = self._tangent_embedding(feats, curv_val)
            logits_local = self.classifier(tangent_local)
            sup_loss = F.cross_entropy(logits_local, y)
            sup_losses.append(sup_loss)

            tangent_means.append(tangent_local.mean(dim=0))
            tangent_norms.append(tangent_local.norm(dim=1))
            weights.append(float(y.size(0)))

            grad_log = torch.autograd.grad(
                sup_loss,
                curv_param,
                retain_graph=True,
                create_graph=True,
                allow_unused=True,
            )[0]
            if grad_log is None:
                grad_val = torch.zeros_like(curv_param)
            else:
                grad_val = grad_log
            local_grad_terms.append(grad_val.abs())

            tangent_global = self._tangent_embedding(feats, global_curv)
            logits_global = self.classifier(tangent_global)
            global_loss = F.cross_entropy(logits_global, y)
            global_losses.append(global_loss)

        sup_loss_total = torch.stack(sup_losses).mean()

        weight_tensor = torch.tensor(weights, device=device, dtype=torch.float32)
        weight_tensor = weight_tensor / (weight_tensor.sum() + 1e-8)
        align_loss = self._alignment_loss(tangent_means, weight_tensor)

        # Curvature variance loss: zero when single_curvature (nothing to penalise)
        if self.single_curvature:
            curv_loss = torch.tensor(0.0, device=device)
        else:
            curv_diff = curvatures.unsqueeze(0) - curvatures.unsqueeze(1)
            curv_loss = (curv_diff ** 2).mean()

        sharp_curv = curvatures.expand(len(tangent_norms)) if self.single_curvature else curvatures
        sharp_loss = self._sharpness_proxy(tangent_norms, sharp_curv)

        local_flat_loss = torch.stack(local_grad_terms).mean()
        global_flat_terms = [
            torch.abs(gl - sl.detach()) for gl, sl in zip(global_losses, sup_losses)
        ]
        global_flat_loss = torch.stack(global_flat_terms).mean()

        total_loss = (
            sup_loss_total
            + self.alpha_curv * curv_loss
            + self.beta_align * align_loss
            + self.gamma_sharp * sharp_loss
            + self.delta_local * local_flat_loss
            + self.delta_global * global_flat_loss
        )

        self.optimizer_main.zero_grad()
        self.optimizer_curv.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(list(self.network_parameters()), self.grad_clip)
        torch.nn.utils.clip_grad_norm_(
            [self.log_curvatures, self.log_global_curvature], self.grad_clip
        )
        self.optimizer_main.step()
        self.optimizer_curv.step()

        with torch.no_grad():
            total_correct = 0
            total_seen = 0
            for x, y in batches:
                preds = self.predict(x)
                total_correct += (preds.argmax(dim=1) == y).sum().item()
                total_seen += y.size(0)
            train_acc = total_correct / max(total_seen, 1)

        metrics = {
            "loss_sup": sup_loss_total.item(),
            "loss_curv": curv_loss.item(),
            "loss_align": align_loss.item(),
            "loss_sharp": sharp_loss.item(),
            "loss_local_flat": local_flat_loss.item(),
            "loss_global_flat": global_flat_loss.item(),
            "loss_total": total_loss.item(),
            "train_acc": train_acc,
            "mean_curvature": curvatures.mean().item(),
            "global_curvature": global_curv.item(),
            "single_curvature_mode": int(self.single_curvature),
        }
        return metrics
