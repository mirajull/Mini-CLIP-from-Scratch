# src/eval.py

import argparse
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .model import CLIPModel
from .dataset import ImageTextDataset
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


def compute_recall_at_k(sim_matrix, k):
    """
    sim_matrix: (N, N) image-to-text similarity (higher is better)
    assumes diagonal is the correct pair.
    """
    n = sim_matrix.size(0)
    ranks = []
    for i in range(n):
        scores = sim_matrix[i]
        sorted_indices = torch.argsort(scores, descending=True)
        rank = (sorted_indices == i).nonzero(as_tuple=False).item()
        ranks.append(rank)

    ranks = torch.tensor(ranks)
    recall = (ranks < k).float().mean().item()
    return recall


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

    model = CLIPModel(
        vocab_size=dataset.vocab_size,
        embed_dim=cfg["model"]["embed_dim"],
        temperature=cfg["model"]["temperature"],
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

    img_embs = torch.cat(all_img_embs, dim=0)
    txt_embs = torch.cat(all_txt_embs, dim=0)

    sim_matrix = img_embs @ txt_embs.t()  # cosine, since normalized

    for k in [1, 5, 10]:
        r_at_k = compute_recall_at_k(sim_matrix, k)
        print(f"Image->Text Recall@{k}: {r_at_k * 100:.2f}%")

    # You can also evaluate text->image by transposing sim_matrix if you want.


if __name__ == "__main__":
    args = parse_args()
    cfg = load_config(args.config)
    evaluate(cfg, args.checkpoint)
