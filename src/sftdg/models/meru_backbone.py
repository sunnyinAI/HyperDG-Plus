"""MERU backbone: ViT pretrained in hyperbolic space (Desai et al., ICML 2023).

Checkpoint key layout (verified from meru_vit_b.pth):
  visual.*          -> ViT trunk (strip 'visual.' prefix to get timm keys)
  visual_proj.weight -> single linear projection: embed_dim -> proj_dim (512)
  curv              -> learnable curvature scalar (log scale)
  pixel_mean/std    -> normalisation constants

Weight download URLs (official MERU release):
  ViT-S/16 : https://dl.fbaipublicfiles.com/meru/meru_vit_s.pth
  ViT-B/16 : https://dl.fbaipublicfiles.com/meru/meru_vit_b.pth
  ViT-L/16 : https://dl.fbaipublicfiles.com/meru/meru_vit_l.pth
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import torch
import torch.nn as nn
import timm


_MERU_URLS = {
    "vit_s16": "https://dl.fbaipublicfiles.com/meru/meru_vit_s.pth",
    "vit_b16": "https://dl.fbaipublicfiles.com/meru/meru_vit_b.pth",
    "vit_l16": "https://dl.fbaipublicfiles.com/meru/meru_vit_l.pth",
}

_TIMM_NAMES = {
    "vit_s16": "vit_small_patch16_224",
    "vit_b16": "vit_base_patch16_224",
    "vit_l16": "vit_large_patch16_224",
}

_EMBED_DIMS = {
    "vit_s16": 384,
    "vit_b16": 768,
    "vit_l16": 1024,
}

# MERU uses a single linear projection: embed_dim -> proj_dim
# Verified: visual_proj.weight shape (512, 768) for ViT-B
_PROJ_DIMS = {
    "vit_s16": 256,
    "vit_b16": 512,
    "vit_l16": 512,
}


def _download_meru_weights(variant: str, cache_dir: Path) -> Path:
    url = _MERU_URLS[variant]
    url_basename = url.rsplit("/", 1)[-1]   # e.g. "meru_vit_b.pth"
    fname = cache_dir / url_basename
    if fname.exists():
        return fname
    cache_dir.mkdir(parents=True, exist_ok=True)
    print(f"[MERU] Downloading {variant} from {url} …")
    torch.hub.download_url_to_file(url, str(fname), progress=True)
    print(f"[MERU] Saved to {fname}")
    return fname


def _load_meru_state(ckpt_path: Path) -> dict:
    """Load MERU checkpoint (requires omegaconf in env)."""
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    return ckpt.get("model", ckpt)


def _remap_visual_keys(meru_state: dict, timm_model: nn.Module) -> dict:
    """Strip 'visual.' prefix and match against timm state-dict keys."""
    prefix = "visual."
    timm_keys = set(timm_model.state_dict().keys())
    remapped = {}
    for k, v in meru_state.items():
        if k.startswith(prefix):
            new_k = k[len(prefix):]
            if new_k in timm_keys:
                remapped[new_k] = v
    return remapped


class MERUBackbone(nn.Module):
    """
    MERU image encoder: ViT trunk + single linear projection head.

    Args:
        variant          : "vit_s16" | "vit_b16" | "vit_l16"
        meru_output      : "euclidean" (proj output) | "lorentz" (after exp-map)
        freeze_backbone  : Freeze ViT trunk weights (recommended)
        cache_dir        : Where to cache downloaded weights
        pretrained       : Load MERU pretrained weights (True) or random init
    """

    def __init__(
        self,
        variant: str = "vit_b16",
        meru_output: Literal["euclidean", "lorentz"] = "euclidean",
        freeze_backbone: bool = True,
        cache_dir: str | Path = Path.home() / ".cache" / "meru",
        pretrained: bool = True,
    ):
        super().__init__()
        assert variant in _TIMM_NAMES, f"variant must be one of {list(_TIMM_NAMES)}"
        self.variant = variant
        self.meru_output = meru_output
        embed_dim = _EMBED_DIMS[variant]
        proj_dim  = _PROJ_DIMS[variant]

        # ViT trunk
        self.vit = timm.create_model(
            _TIMM_NAMES[variant],
            pretrained=False,
            num_classes=0,
            global_pool="",
        )

        # Single linear projection (matches MERU architecture exactly)
        self.proj = nn.Linear(embed_dim, proj_dim, bias=False)

        if pretrained:
            cache_dir = Path(cache_dir)
            ckpt_path = _download_meru_weights(variant, cache_dir)
            state = _load_meru_state(ckpt_path)

            # Load ViT trunk
            vit_state = _remap_visual_keys(state, self.vit)
            missing, unexpected = self.vit.load_state_dict(vit_state, strict=False)
            n_loaded = len(vit_state) - len(missing)
            print(f"[MERU] ViT trunk: loaded {n_loaded}/{len(vit_state)} keys"
                  + (f", missing {len(missing)}" if missing else ""))

            # Load projection head
            if "visual_proj.weight" in state:
                self.proj.weight.data.copy_(state["visual_proj.weight"])
                print(f"[MERU] Projection head loaded: {state['visual_proj.weight'].shape}")
            else:
                print("[MERU] visual_proj.weight not found – using random init.")

            # Store MERU's learnable curvature as a buffer (for reference)
            if "curv" in state:
                self.register_buffer("meru_curv", state["curv"])
                print(f"[MERU] Curvature: {state['curv'].item():.4f} (log scale)")

        if freeze_backbone:
            for param in self.vit.parameters():
                param.requires_grad = False

        self.n_outputs = proj_dim

    def _exp_map_origin(self, v: torch.Tensor) -> torch.Tensor:
        """Exp map at Lorentz origin with curvature |K|=1."""
        norm = v.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        x0 = torch.cosh(norm)
        xi = torch.sinh(norm) / norm * v
        return torch.cat([x0, xi], dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        patch_tokens = self.vit.patch_embed(x)
        cls_tokens   = self.vit.cls_token.expand(B, -1, -1)
        tokens = torch.cat((cls_tokens, patch_tokens), dim=1)
        tokens = tokens + self.vit.pos_embed
        tokens = self.vit.pos_drop(tokens)
        tokens = self.vit.blocks(tokens)
        tokens = self.vit.norm(tokens)
        cls_feat = tokens[:, 0]              # (B, embed_dim)

        proj_feat = self.proj(cls_feat)      # (B, proj_dim)

        if self.meru_output == "lorentz":
            return self._exp_map_origin(proj_feat)   # (B, proj_dim+1)
        return proj_feat                             # (B, proj_dim)


def build_meru_backbone(input_shape, hparams) -> MERUBackbone:
    variant   = hparams.get("meru_variant", "vit_b16")
    output    = hparams.get("meru_output", "euclidean")
    freeze    = bool(hparams.get("meru_freeze_backbone", True))
    pretrained = bool(hparams.get("meru_pretrained", True))
    cache_dir = hparams.get("meru_cache_dir", str(Path.home() / ".cache" / "meru"))
    return MERUBackbone(
        variant=variant,
        meru_output=output,
        freeze_backbone=freeze,
        cache_dir=cache_dir,
        pretrained=pretrained,
    )
