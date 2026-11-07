"""Federated client implementation for HyperDG-style training."""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from domainbed import networks

from sftdg.models import build_featurizer


def _softplus_curvature(params: torch.Tensor, eps: float) -> torch.Tensor:
    return torch.nn.functional.softplus(params) + eps


class HyperDGFedClient(nn.Module):
    """Client model for federated HyperDG training."""

    def __init__(self, input_shape, num_classes, hparams: Dict) -> None:
        super().__init__()
        self.featurizer = build_featurizer(input_shape, hparams)
        feat_dim = self.featurizer.n_outputs
        self.classifier = networks.Classifier(
            feat_dim,
            num_classes,
            hparams.get("nonlinear_classifier", False),
        )

        self.curv_eps = hparams.get("curvature_epsilon", 1e-4)
        self.grad_clip = hparams.get("grad_clip_norm", 5.0)
        self.alpha_curv = hparams.get("alpha_curv", 0.2)
        self.beta_align = hparams.get("beta_align", 0.5)
        self.gamma_sharp = hparams.get("gamma_sharp", 0.05)
        self.delta_local = hparams.get("delta_local_flat", 0.1)
        self.delta_global = hparams.get("delta_global_flat", 0.1)

        self.log_curvature = nn.Parameter(torch.zeros(1))
        self.log_global_curvature = nn.Parameter(torch.tensor(0.0))

        curvature_lr = hparams.get("curvature_lr", hparams["lr"])

        self.optimizer_main = torch.optim.Adam(
            list(self.network_parameters()),
            lr=hparams["lr"],
            weight_decay=hparams["weight_decay"],
        )
        self.optimizer_curv = torch.optim.Adam(
            [self.log_curvature, self.log_global_curvature],
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

    def _sharpness_proxy(self, tangent_norms: torch.Tensor, curvature: torch.Tensor) -> torch.Tensor:
        if tangent_norms.numel() < 2:
            return torch.tensor(0.0, device=tangent_norms.device)
        return tangent_norms.var(unbiased=False) * curvature

    def update(
        self,
        minibatch: Tuple[torch.Tensor, torch.Tensor],
        mu_global: torch.Tensor | None = None,
        curv_mean: torch.Tensor | None = None,
    ) -> Tuple[dict, torch.Tensor, int]:
        device = next(self.parameters()).device
        x, y = minibatch
        x = x.to(device)
        y = y.to(device)

        if mu_global is not None:
            mu_global = mu_global.to(device)
        if curv_mean is not None:
            curv_mean = curv_mean.to(device)

        curvatures = _softplus_curvature(self.log_curvature, self.curv_eps)
        global_curv = _softplus_curvature(self.log_global_curvature, self.curv_eps)

        feats = self.featurizer(x)
        tangent_local = self._tangent_embedding(feats, curvatures)
        logits_local = self.classifier(tangent_local)
        sup_loss = F.cross_entropy(logits_local, y)

        mu_batch = tangent_local.mean(dim=0)
        if mu_global is None:
            align_loss = torch.tensor(0.0, device=device)
        else:
            align_loss = ((mu_batch - mu_global) ** 2).sum()

        if curv_mean is None:
            curv_loss = torch.tensor(0.0, device=device)
        else:
            curv_loss = (curvatures - curv_mean).pow(2).mean()

        tangent_norms = tangent_local.norm(dim=1)
        sharp_loss = self._sharpness_proxy(tangent_norms, curvatures)

        grad_log = torch.autograd.grad(
            sup_loss,
            self.log_curvature,
            retain_graph=True,
            create_graph=True,
            allow_unused=True,
        )[0]
        local_flat_loss = grad_log.abs().mean() if grad_log is not None else torch.tensor(0.0, device=device)

        tangent_global = self._tangent_embedding(feats, global_curv)
        logits_global = self.classifier(tangent_global)
        global_loss = F.cross_entropy(logits_global, y)
        global_flat_loss = torch.abs(global_loss - sup_loss.detach())

        total_loss = (
            sup_loss
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
            [self.log_curvature, self.log_global_curvature], self.grad_clip
        )
        self.optimizer_main.step()
        self.optimizer_curv.step()

        with torch.no_grad():
            preds = self.predict(x)
            train_acc = (preds.argmax(dim=1) == y).float().mean().item()

        metrics = {
            "loss_sup": sup_loss.item(),
            "loss_curv": curv_loss.item(),
            "loss_align": align_loss.item(),
            "loss_sharp": sharp_loss.item(),
            "loss_local_flat": local_flat_loss.item(),
            "loss_global_flat": global_flat_loss.item(),
            "loss_total": total_loss.item(),
            "train_acc": train_acc,
            "curvature": curvatures.item(),
            "global_curvature": global_curv.item(),
        }
        return metrics, mu_batch.detach(), int(y.size(0))

    def get_global_state(self) -> Dict[str, torch.Tensor]:
        state = self.state_dict()
        state.pop("log_curvature", None)
        return {k: v.detach().cpu() for k, v in state.items()}

    def set_global_state(self, state: Dict[str, torch.Tensor]) -> None:
        self.load_state_dict(state, strict=False)
