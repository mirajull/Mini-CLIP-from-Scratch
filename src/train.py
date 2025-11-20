import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm
from model import CLIPModel, clip_loss
from dataset import ImageTextDataset  # you'll define this

def train(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = ImageTextDataset(config["data"]["annotations"], config["data"]["img_root"])
    loader = DataLoader(dataset, batch_size=config["train"]["batch_size"], shuffle=True,
                        num_workers=4, collate_fn=dataset.collate_fn)

    model = CLIPModel(vocab_size=dataset.vocab_size)
    model.to(device)
    optim = AdamW(model.parameters(), lr=config["train"]["lr"])

    for epoch in range(config["train"]["epochs"]):
        model.train()
        epoch_loss = 0.0
        for batch in tqdm(loader, desc=f"Epoch {epoch}"):
            images = batch["images"].to(device)
            tokens = batch["tokens"].to(device)
            lengths = batch["lengths"].to(device)

            logits_i, logits_t = model(images, tokens, lengths)
            loss = clip_loss(logits_i, logits_t)

            optim.zero_grad()
            loss.backward()
            optim.step()
            epoch_loss += loss.item() * images.size(0)

        print(f"Epoch {epoch} | Loss: {epoch_loss / len(dataset):.4f}")

if __name__ == "__main__":
    # minimal config inline; in practice load from YAML
    cfg = {
        "data": {"annotations": "data/train.json", "img_root": "data/images"},
        "train": {"batch_size": 64, "lr": 1e-4, "epochs": 10},
    }
    train(cfg)
