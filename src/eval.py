# src/eval.py

import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    import faiss
except ImportError as exc:  # pragma: no cover - import guard
    raise ImportError(
        "FAISS is required for evaluation. Install faiss-cpu via pip."  # noqa: EM101
    ) from exc

from .dataset import ImageTextDataset
from .model import CLIPModel
from .utils import load_config


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Mini-CLIP Retrieval")
    parser.add_argument(
        "--config", type=str, default="configs/default.yaml",
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to model checkpoint (.pt)",
    )
    return parser.parse_args()


def recall_from_neighbors(neighbors: np.ndarray, k: int) -> float:
    hits = sum(1 for idx, row in enumerate(neighbors) if idx in row[:k])
    return hits / neighbors.shape[0]


def evaluate(cfg, ckpt_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = ImageTextDataset(
        annotations_path=cfg["data"]["annotations"],
        img_root=cfg["data"]["img_root"],
        max_length=cfg["data"]["max_length"],
    )
    loader = DataLoader(
        dataset,
        batch_size=64,
        shuffle=False,
        num_workers=cfg["train"]["num_workers"],
        collate_fn=ImageTextDataset.collate_fn,
    )

    vision_cfg = cfg["model"].get("vision_encoder", {})
    text_cfg = cfg["model"].get("text_encoder", {})
    model = CLIPModel(
        vocab_size=dataset.vocab_size,
        context_length=dataset.max_length,
        embed_dim=cfg["model"]["embed_dim"],
        temperature=cfg["model"]["temperature"],
        vision_cfg=vision_cfg,
        text_cfg=text_cfg,
        logit_scale_max=cfg["model"].get("logit_scale_max", 100.0),
    )
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()

    all_img_embs = []
    all_txt_embs = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Encoding"):
            images = batch["images"].to(device)
            tokens = batch["tokens"].to(device)
            lengths = batch["lengths"].to(device)

            img_emb = model.image_encoder(images)
            txt_emb = model.text_encoder(tokens, lengths)

            all_img_embs.append(img_emb.cpu())
            all_txt_embs.append(txt_emb.cpu())

    img_embs = torch.cat(all_img_embs, dim=0).cpu()
    txt_embs = torch.cat(all_txt_embs, dim=0).cpu()

    img_np = img_embs.numpy().astype("float32")
    txt_np = txt_embs.numpy().astype("float32")

    recall_ks = cfg.get("eval", {}).get("recall_k", [1, 5, 10])
    max_k = max(recall_ks)

    text_index = faiss.IndexFlatIP(txt_np.shape[1])
    text_index.add(txt_np)
    _, img_to_text = text_index.search(img_np, max_k)

    image_index = faiss.IndexFlatIP(img_np.shape[1])
    image_index.add(img_np)
    _, text_to_image = image_index.search(txt_np, max_k)

    for k in recall_ks:
        r_img_text = recall_from_neighbors(img_to_text, k)
        r_text_img = recall_from_neighbors(text_to_image, k)
        print(f"Image->Text Recall@{k}: {r_img_text * 100:.2f}%")
        print(f"Text->Image Recall@{k}: {r_text_img * 100:.2f}%")


if __name__ == "__main__":
    args = parse_args()
    cfg = load_config(args.config)
    evaluate(cfg, args.checkpoint)
