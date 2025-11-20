import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18


class ImageEncoder(nn.Module):
    def __init__(self, embed_dim: int = 256):
        super().__init__()
        backbone = resnet18(weights="IMAGENET1K_V1")
        modules = list(backbone.children())[:-1]  # remove fc
        self.backbone = nn.Sequential(*modules)
        self.fc = nn.Linear(backbone.fc.in_features, embed_dim)

    def forward(self, x):
        feats = self.backbone(x)        # (B, C, 1, 1)
        feats = feats.view(feats.size(0), -1)
        out = self.fc(feats)
        return F.normalize(out, dim=-1)


class TextEncoder(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int = 256, hidden_dim: int = 512):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.rnn = nn.GRU(hidden_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(2 * hidden_dim, embed_dim)

    def forward(self, tokens, lengths):
        # tokens: (B, L)
        x = self.embedding(tokens)
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, h_n = self.rnn(packed)  # (2, B, H)
        h = torch.cat([h_n[0], h_n[1]], dim=-1)  # (B, 2H)
        out = self.fc(h)
        return F.normalize(out, dim=-1)


class CLIPModel(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int = 256, temperature: float = 0.07):
        super().__init__()
        self.image_encoder = ImageEncoder(embed_dim)
        self.text_encoder = TextEncoder(vocab_size, embed_dim)
        self.logit_scale = nn.Parameter(torch.log(torch.tensor(1 / temperature)))

    def forward(self, images, tokens, lengths):
        img_emb = self.image_encoder(images)   # (B, D)
        txt_emb = self.text_encoder(tokens, lengths)  # (B, D)

        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * img_emb @ txt_emb.t()
        logits_per_text = logits_per_image.t()
        return logits_per_image, logits_per_text


def clip_loss(logits_per_image, logits_per_text):
    batch_size = logits_per_image.size(0)
    labels = torch.arange(batch_size, device=logits_per_image.device)
    loss_i = F.cross_entropy(logits_per_image, labels)
    loss_t = F.cross_entropy(logits_per_text, labels)
    return (loss_i + loss_t) / 2
