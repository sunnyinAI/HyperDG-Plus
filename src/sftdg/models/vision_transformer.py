"""Vision Transformer backbones with optional Visual Prompt Tuning."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import timm


@dataclass
class ViTConfig:
    model_name: str = "vit_base_patch16_224"
    prompt_length: int = 0
    prompt_dropout: float = 0.1
    freeze_backbone: bool = False


class VisionTransformerBackbone(nn.Module):
    """Wrapper around timm ViT models with optional prompt tokens."""

    def __init__(self, input_shape, cfg: ViTConfig):
        super().__init__()

        if len(input_shape) != 3 or input_shape[0] != 3:
            raise ValueError(
                f"ViT backbones expect 3-channel images, got shape={input_shape}"
            )

        self.cfg = cfg
        self.vit = timm.create_model(
            cfg.model_name,
            pretrained=True,
            num_classes=0,
            global_pool="",
        )
        self.embed_dim = getattr(self.vit, "embed_dim", None)
        if self.embed_dim is None:
            raise RuntimeError(f"Model {cfg.model_name} missing embed_dim attribute.")

        self.prompt_length = int(cfg.prompt_length)
        if self.prompt_length > 0:
            self.prompt_embeddings = nn.Parameter(
                torch.zeros(1, self.prompt_length, self.embed_dim)
            )
            nn.init.normal_(self.prompt_embeddings, std=0.02)
            self.prompt_pos_embeddings = nn.Parameter(
                torch.zeros(1, self.prompt_length, self.embed_dim)
            )
            nn.init.normal_(self.prompt_pos_embeddings, std=0.02)
            self.prompt_dropout = nn.Dropout(cfg.prompt_dropout)
        else:
            self.register_parameter("prompt_embeddings", None)
            self.register_parameter("prompt_pos_embeddings", None)
            self.prompt_dropout = nn.Identity()

        if cfg.freeze_backbone:
            for param in self.vit.parameters():
                param.requires_grad = False

        self.n_outputs = self.embed_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        vit = self.vit

        patch_tokens = vit.patch_embed(x)
        cls_tokens = vit.cls_token.expand(B, -1, -1)

        dist_token = getattr(vit, "dist_token", None)
        if dist_token is not None and dist_token.numel() != 0:
            dist_token = dist_token.expand(B, -1, -1)
            tokens = torch.cat((cls_tokens, dist_token, patch_tokens), dim=1)
        else:
            dist_token = None
            tokens = torch.cat((cls_tokens, patch_tokens), dim=1)

        pos_embed = vit.pos_embed

        if self.prompt_embeddings is not None:
            prompt_tokens = self.prompt_embeddings.expand(B, -1, -1)
            prompt_tokens = self.prompt_dropout(prompt_tokens)
            tokens = torch.cat(
                (tokens[:, :1, :], prompt_tokens, tokens[:, 1:, :]),
                dim=1,
            )
            if pos_embed.shape[1] != tokens.shape[1]:
                pos_embed = torch.cat(
                    (
                        vit.pos_embed[:, :1, :],
                        self.prompt_pos_embeddings,
                        vit.pos_embed[:, 1:, :],
                    ),
                    dim=1,
                )

        if pos_embed.shape[1] != tokens.shape[1]:
            raise RuntimeError(
                f"Position embedding length {pos_embed.shape[1]} does not match tokens {tokens.shape[1]}"
            )

        x = tokens + pos_embed
        x = vit.pos_drop(x)
        x = vit.blocks(x)
        x = vit.norm(x)

        if dist_token is None:
            feats = x[:, 0]
        else:
            feats = x[:, 0]
        return feats


def build_vit_backbone(input_shape, hparams) -> VisionTransformerBackbone:
    cfg = ViTConfig(
        model_name=hparams.get("vit_model_name", "vit_base_patch16_224"),
        prompt_length=int(hparams.get("vit_prompt_length", 0)),
        prompt_dropout=float(hparams.get("vit_prompt_dropout", 0.1)),
        freeze_backbone=bool(hparams.get("vit_freeze_backbone", False)),
    )
    return VisionTransformerBackbone(input_shape, cfg)
