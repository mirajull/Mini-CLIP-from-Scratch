import argparse
import os

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm

from .model import CLIPModel, clip_loss
from .dataset import ImageTextDataset
from .utils import load_config, set_seed, ensure_dir, save_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description="Train Mini-CLIP")
    parser.add_argument(
        "--config", type=str, default="configs/default.yaml",
        help="Path to YAML config file",
    )
    return parser.parse_args()


def train(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # Seed
    seed = cfg["logging"].get("seed", 42)
    set_seed(seed)

    # Dataset & DataLoader
    dataset = ImageTextDataset(
        annotations_path=cfg["data"]["annotations"],
        img_root=cfg["data"]["img_root"],
        max_length=cfg["data"]["max_length"],
    )

    loader = DataLoader(
        dataset,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        num_workers=cfg["train"]["num_workers"],
        collate_fn=ImageTextDataset.collate_fn,
        pin_memory=True,
    )

    # Model
    model = CLIPModel(
        vocab_size=dataset.vocab_size,
        embed_dim=cfg["model"]["embed_dim"],
        temperature=cfg["model"]["temperature"],
    )
    model.to(device)

    # Optimizer
    optim = AdamW(model.parameters(), lr=cfg["train"]["lr"])

    # AMP scaler
    use_amp = cfg["train"].get("amp", False)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    ckpt_dir = cfg["logging"]["ckpt_dir"]
    ensure_dir(ckpt_dir)

    global_step = 0
    for epoch in range(cfg["train"]["epochs"]):
        model.train()
        running_loss = 0.0

        pbar = tqdm(loader, desc=f"Epoch {epoch}", dynamic_ncols=True)
        for i, batch in enumerate(pbar):
            images = batch["images"].to(device, non_blocking=True)
            tokens = batch["tokens"].to(device, non_blocking=True)
            lengths = batch["lengths"].to(device, non_blocking=True)

            optim.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=use_amp):
                logits_i, logits_t = model(images, tokens, lengths)
                loss = clip_loss(logits_i, logits_t)

            scaler.scale(loss).backward()
            if cfg["train"]["grad_clip"] is not None:
                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), cfg["train"]["grad_clip"]
                )
            scaler.step(optim)
            scaler.update()

            batch_loss = loss.item()
            running_loss += batch_loss * images.size(0)
            global_step += 1

            if (i + 1) % cfg["logging"]["print_every"] == 0:
                avg_loss = running_loss / ((i + 1) * cfg["train"]["batch_size"])
                pbar.set_postfix(loss=f"{avg_loss:.4f}")

        epoch_loss = running_loss / len(dataset)
        print(f"[INFO] Epoch {epoch} finished | Loss: {epoch_loss:.4f}")

        # Save checkpoint
        if (epoch + 1) % cfg["logging"]["ckpt_interval"] == 0:
            save_checkpoint(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optim.state_dict(),
                    "config": cfg,
                },
                ckpt_dir=ckpt_dir,
                filename=f"clip_epoch{epoch+1}.pt",
            )


if __name__ == "__main__":
    args = parse_args()
    cfg = load_config(args.config)
    train(cfg)
