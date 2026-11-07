"""HyperDG-MERU: HyperDG with MERU pretrained backbone + hyperbolic contrastive loss.

Addresses Reviewer hQnG's second experiment request:
  "Redoing all the experiments with a pretrained Meru backbone and a
   hyperbolic-contrastive loss."

The hyperbolic contrastive loss operates in the Lorentz model and uses all
same-class representations as positives after concatenating domains. The
current implementation therefore mixes intra-domain and inter-domain pairs.

No entailment loss is used (as the reviewer specified).
"""

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
    return F.softplus(params) + eps


def _lorentz_inner(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Lorentz inner product: -x0*y0 + sum(xi*yi).  Shape: (B,) or scalar."""
    return -x[..., 0] * y[..., 0] + (x[..., 1:] * y[..., 1:]).sum(dim=-1)


def _lorentz_dist(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Geodesic distance on the Lorentz manifold (curvature K=-1).
    Clamps acosh input to [1+eps, inf) for numerical stability.
    """
    inner = _lorentz_inner(x, y)
    # acosh requires input >= 1; clamp ensures no NaN from floating-point drift
    return torch.acosh((-inner).clamp(min=1.0 + 1e-6))


def _exp_map_origin(v: torch.Tensor, curv: torch.Tensor) -> torch.Tensor:
    """Exponential map from the Lorentz origin.

    v   : tangent vector in R^n, shape (B, n)
    curv: scalar curvature magnitude (|K|)
    Returns: point on L^n, shape (B, n+1)
    """
    # Ensure curv is a scalar for safe broadcasting with (B,1) norm
    sqrt_c = curv.squeeze().sqrt().clamp(min=1e-8)        # scalar
    norm = v.norm(dim=-1, keepdim=True).clamp(min=1e-8)   # (B,1)
    x0 = torch.cosh(sqrt_c * norm)                         # (B,1)
    xi = torch.sinh(sqrt_c * norm) / norm * v              # (B,n)
    return torch.cat([x0, xi], dim=-1)                     # (B, n+1)


class HyperbolicContrastiveLoss(nn.Module):
    """InfoNCE-style contrastive loss in Lorentz space.

    Positive pairs  : same-class samples, within or across domains.
    Negative pairs  : different-class samples (random subset).
    Temperature is in geodesic-distance units.
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        lorentz_feats: List[torch.Tensor],  # list of (Ni, D+1) per domain
        labels: List[torch.Tensor],          # list of (Ni,) per domain
    ) -> torch.Tensor:
        # Concatenate across domains
        all_feats = torch.cat(lorentz_feats, dim=0)   # (N, D+1)
        all_labels = torch.cat(labels, dim=0)          # (N,)
        N = all_feats.shape[0]
        if N < 2:
            return torch.tensor(0.0, device=all_feats.device)

        # Vectorised pairwise Lorentz distances  (N, N)
        # inner(i,j) computed via broadcasting: (N,1,D+1) vs (1,N,D+1)
        x1 = all_feats.unsqueeze(1)   # (N, 1, D+1)
        x2 = all_feats.unsqueeze(0)   # (1, N, D+1)
        inner_mat = _lorentz_inner(x1, x2)                        # (N, N)
        dists = torch.acosh((-inner_mat).clamp(min=1.0 + 1e-6))   # (N, N)
        sim = -dists / self.temperature                            # (N, N)

        # Positive mask: same class, different index
        label_match = all_labels.unsqueeze(0) == all_labels.unsqueeze(1)  # (N,N)
        eye_mask = torch.eye(N, dtype=torch.bool, device=all_feats.device)
        pos_mask = label_match & ~eye_mask

        # InfoNCE loss per anchor — mask self-similarity
        sim = sim.masked_fill(eye_mask, float("-inf"))
        log_probs = sim - torch.logsumexp(sim, dim=1, keepdim=True)  # (N, N)

        # Use torch.where to avoid 0 * (-inf) = NaN when masking non-positives
        zeros = torch.zeros_like(log_probs)
        pos_log_probs = torch.where(pos_mask, log_probs, zeros)       # (N, N)

        n_pos = pos_mask.float().sum(dim=1).clamp(min=1.0)
        loss_per_anchor = -pos_log_probs.sum(dim=1) / n_pos
        return loss_per_anchor.mean()


@register_algorithm("hyperdg_meru")
class HyperDGMERU(Algorithm):
    """HyperDG with MERU backbone and hyperbolic contrastive loss.

    Config keys:
      backbone           : "meru_vit_b16" or "meru_vit_s16"
      single_curvature   : bool  (default False — per-domain)
      meru_variant       : "vit_b16" | "vit_s16"
      meru_pretrained    : bool (default True)
      meru_freeze_backbone: bool (default True)
      contrast_temp      : float (default 0.07)
      lambda_contrast    : float (default 0.1)
      alpha_curv / beta_align / gamma_sharp / delta_local_flat / delta_global_flat
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super().__init__(input_shape, num_classes, num_domains, hparams)

        self.featurizer = build_featurizer(input_shape, hparams)
        feat_dim = self.featurizer.n_outputs
        self.classifier = networks.Classifier(
            feat_dim, num_classes, hparams.get("nonlinear_classifier", False)
        )

        self.num_domains = num_domains
        self.curv_eps = hparams.get("curvature_epsilon", 1e-4)
        self.grad_clip = hparams.get("grad_clip_norm", 5.0)
        self.alpha_curv = hparams.get("alpha_curv", 0.2)
        self.beta_align = hparams.get("beta_align", 0.5)
        self.gamma_sharp = hparams.get("gamma_sharp", 0.05)
        self.delta_local = hparams.get("delta_local_flat", 0.1)
        self.delta_global = hparams.get("delta_global_flat", 0.1)
        self.lambda_contrast = hparams.get("lambda_contrast", 0.1)
        curvature_lr = hparams.get("curvature_lr", hparams["lr"])

        self.single_curvature = bool(hparams.get("single_curvature", False))
        if self.single_curvature:
            self.log_curvatures = nn.Parameter(torch.zeros(1))
        else:
            self.log_curvatures = nn.Parameter(torch.zeros(num_domains))
        self.log_global_curvature = nn.Parameter(torch.tensor(0.0))

        self.contrast_loss_fn = HyperbolicContrastiveLoss(
            temperature=hparams.get("contrast_temp", 0.07)
        )

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
        feats = F.normalize(self.featurizer(x), dim=-1)
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
            raise ValueError("HyperDGMERU requires at least one minibatch.")

        curvatures = _softplus_curvature(self.log_curvatures, self.curv_eps)
        global_curv = _softplus_curvature(self.log_global_curvature, self.curv_eps)

        sup_losses, global_losses = [], []
        tangent_means, tangent_norms = [], []
        lorentz_feats_per_domain, labels_per_domain = [], []
        local_grad_terms, weights = [], []

        for env_idx, (x, y) in enumerate(batches):
            feats = self.featurizer(x)                        # (B, feat_dim)
            # L2-normalise to unit sphere so that sqrt_K * ||feat|| ≈ sqrt_K << 1,
            # preventing cosh/sinh overflow in exp_map for large-norm backbones (MERU).
            feats = F.normalize(feats, dim=-1)

            curv_param = self.log_curvatures[0] if self.single_curvature else self.log_curvatures[env_idx]
            curv_val = curvatures[0] if self.single_curvature else curvatures[env_idx]

            # Tangent embedding + supervised loss
            tangent_local = self._tangent_embedding(feats, curv_val)
            logits_local = self.classifier(tangent_local)
            sup_loss = F.cross_entropy(logits_local, y)
            sup_losses.append(sup_loss)

            tangent_means.append(tangent_local.mean(dim=0))
            tangent_norms.append(tangent_local.norm(dim=1))
            weights.append(float(y.size(0)))

            # Local flatness: gradient of loss w.r.t. curvature param
            grad_log = torch.autograd.grad(
                sup_loss, curv_param,
                retain_graph=True, create_graph=True, allow_unused=True,
            )[0]
            local_grad_terms.append(
                torch.zeros_like(curv_param) if grad_log is None else grad_log.abs()
            )

            # Global tangent embedding
            tangent_global = self._tangent_embedding(feats, global_curv)
            logits_global = self.classifier(tangent_global)
            global_losses.append(F.cross_entropy(logits_global, y))

            # Map tangent embeddings to Lorentz for contrastive loss
            lorentz_pt = _exp_map_origin(tangent_local, curv_val.detach())
            lorentz_feats_per_domain.append(lorentz_pt)
            labels_per_domain.append(y)

        sup_loss_total = torch.stack(sup_losses).mean()

        weight_tensor = torch.tensor(weights, device=device, dtype=torch.float32)
        weight_tensor = weight_tensor / (weight_tensor.sum() + 1e-8)
        align_loss = self._alignment_loss(tangent_means, weight_tensor)

        if self.single_curvature:
            curv_loss = torch.tensor(0.0, device=device)
        else:
            curv_diff = curvatures.unsqueeze(0) - curvatures.unsqueeze(1)
            curv_loss = (curv_diff ** 2).mean()

        sharp_curv = curvatures.expand(len(tangent_norms)) if self.single_curvature else curvatures
        sharp_loss = self._sharpness_proxy(tangent_norms, sharp_curv)

        local_flat_loss = torch.stack(local_grad_terms).mean()
        global_flat_loss = torch.stack([
            torch.abs(gl - sl.detach()) for gl, sl in zip(global_losses, sup_losses)
        ]).mean()

        # Hyperbolic contrastive loss across domains
        contrast_loss = self.contrast_loss_fn(lorentz_feats_per_domain, labels_per_domain)

        total_loss = (
            sup_loss_total
            + self.alpha_curv * curv_loss
            + self.beta_align * align_loss
            + self.gamma_sharp * sharp_loss
            + self.delta_local * local_flat_loss
            + self.delta_global * global_flat_loss
            + self.lambda_contrast * contrast_loss
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
            total_correct = total_seen = 0
            for x, y in batches:
                preds = self.predict(x)
                total_correct += (preds.argmax(dim=1) == y).sum().item()
                total_seen += y.size(0)
            train_acc = total_correct / max(total_seen, 1)

        return {
            "loss_sup": sup_loss_total.item(),
            "loss_curv": curv_loss.item(),
            "loss_align": align_loss.item(),
            "loss_sharp": sharp_loss.item(),
            "loss_local_flat": local_flat_loss.item(),
            "loss_global_flat": global_flat_loss.item(),
            "loss_contrast": contrast_loss.item(),
            "loss_total": total_loss.item(),
            "train_acc": train_acc,
            "mean_curvature": curvatures.mean().item(),
            "global_curvature": global_curv.item(),
            "single_curvature_mode": int(self.single_curvature),
        }
