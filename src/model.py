from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ViT_B_16_Weights, vit_b_16


class ImageEncoder(nn.Module):
    def __init__(
        self,
        embed_dim: int = 256,
        pretrained: bool = True,
        freeze_backbone: bool = False,
    ):
        super().__init__()
        weights = ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = vit_b_16(weights=weights)
        self.hidden_dim = backbone.hidden_dim
        backbone.heads = nn.Identity()
        self.backbone = backbone
        self.proj = nn.Linear(self.hidden_dim, embed_dim)

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def forward(self, x):
        feats = self.backbone(x)  # (B, hidden_dim)
        out = self.proj(feats)
        return F.normalize(out, dim=-1)


class TransformerTextEncoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        embed_dim: int = 256,
        hidden_dim: int = 512,
        num_layers: int = 4,
        num_heads: int = 8,
        mlp_dim: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.context_length = context_length
        self.token_embedding = nn.Embedding(vocab_size, hidden_dim)
        self.positional_embedding = nn.Parameter(
            torch.zeros(1, context_length, hidden_dim)
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=mlp_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.proj = nn.Linear(hidden_dim, embed_dim)
        self.ln_post = nn.LayerNorm(hidden_dim)

    def forward(self, tokens, lengths):
        # tokens: (B, L)
        attn_mask = (
            torch.arange(tokens.size(1), device=tokens.device)
            .unsqueeze(0)
            .expand(tokens.size(0), -1)
        ) >= lengths.unsqueeze(1)

        x = self.token_embedding(tokens)
        pos = self.positional_embedding[:, : tokens.size(1), :]
        x = x + pos
        x = self.transformer(x, src_key_padding_mask=attn_mask)
        x = self.ln_post(x)
        cls_state = x[:, 0, :]
        out = self.proj(cls_state)
        return F.normalize(out, dim=-1)


class CLIPModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        embed_dim: int = 256,
        temperature: float = 0.07,
        vision_cfg: Optional[Dict] = None,
        text_cfg: Optional[Dict] = None,
        logit_scale_max: float = 100.0,
    ):
        super().__init__()
        vision_cfg = vision_cfg or {}
        text_cfg = text_cfg or {}

        self.image_encoder = ImageEncoder(embed_dim=embed_dim, **vision_cfg)
        self.text_encoder = TransformerTextEncoder(
            vocab_size=vocab_size,
            context_length=context_length,
            embed_dim=embed_dim,
            **text_cfg,
        )
        self.logit_scale = nn.Parameter(torch.log(torch.tensor(1 / temperature)))
        self._max_logit_scale = logit_scale_max

    def forward(self, images, tokens, lengths):
        img_emb = self.image_encoder(images)   # (B, D)
        txt_emb = self.text_encoder(tokens, lengths)  # (B, D)

        logit_scale = self.logit_scale.exp()
        if self._max_logit_scale is not None:
            max_scale = torch.tensor(self._max_logit_scale, device=logit_scale.device)
            logit_scale = torch.clamp(logit_scale, max=max_scale)
        logits_per_image = logit_scale * img_emb @ txt_emb.t()
        logits_per_text = logits_per_image.t()
        return logits_per_image, logits_per_text


def clip_loss(logits_per_image, logits_per_text, hard_neg_cfg: dict | None = None):
    batch_size = logits_per_image.size(0)
    labels = torch.arange(batch_size, device=logits_per_image.device)
    loss_i = F.cross_entropy(logits_per_image, labels)
    loss_t = F.cross_entropy(logits_per_text, labels)
    loss = (loss_i + loss_t) / 2

    if hard_neg_cfg and hard_neg_cfg.get("enabled", False) and batch_size > 1:
        top_k = min(hard_neg_cfg.get("top_k", 5), batch_size - 1)
        margin = hard_neg_cfg.get("margin", 0.1)
        weight = hard_neg_cfg.get("weight", 0.1)
        if top_k > 0 and weight > 0:
            neg_loss_img = hard_negative_rank_loss(logits_per_image, top_k, margin)
            neg_loss_txt = hard_negative_rank_loss(logits_per_text, top_k, margin)
            loss = loss + weight * (neg_loss_img + neg_loss_txt) / 2

    return loss


def hard_negative_rank_loss(similarity: torch.Tensor, top_k: int, margin: float):
    batch_size = similarity.size(0)
    diag_mask = torch.eye(batch_size, device=similarity.device, dtype=torch.bool)
    neg_scores = similarity.masked_fill(diag_mask, float("-inf"))
    hard_vals, _ = torch.topk(neg_scores, k=top_k, dim=1)
    pos_scores = similarity.diag().unsqueeze(1)
    loss = F.relu(margin + hard_vals - pos_scores)
    return loss.mean()
