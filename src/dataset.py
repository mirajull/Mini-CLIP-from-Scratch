import json
import os
from typing import List, Dict, Any

import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms


SPECIAL_TOKENS = {
    "<pad>": 0,
    "<bos>": 1,
    "<eos>": 2,
    "<unk>": 3,
}


class Vocab:
    def __init__(self, min_freq: int = 1):
        self.stoi = dict(SPECIAL_TOKENS)        # string -> index
        self.itos = {i: s for s, i in self.stoi.items()}  # index -> string
        self.freqs = {}
        self.min_freq = min_freq

    def add_sentence(self, sentence: str):
        for token in self.tokenize(sentence):
            self.freqs[token] = self.freqs.get(token, 0) + 1

    def build(self):
        idx = len(self.stoi)
        for token, freq in sorted(self.freqs.items()):
            if freq >= self.min_freq and token not in self.stoi:
                self.stoi[token] = idx
                self.itos[idx] = token
                idx += 1

    @staticmethod
    def tokenize(text: str) -> List[str]:
        # super simple whitespace tokenizer; you can swap with anything else
        return text.lower().strip().split()

    def encode(self, text: str, max_length: int) -> List[int]:
        tokens = self.tokenize(text)
        ids = [self.stoi["<bos>"]]
        for tok in tokens:
            ids.append(self.stoi.get(tok, self.stoi["<unk>"]))
            if len(ids) >= max_length - 1:
                break
        ids.append(self.stoi["<eos>"])
        return ids

    @property
    def size(self) -> int:
        return len(self.stoi)


class ImageTextDataset(Dataset):
    def __init__(
        self,
        annotations_path: str,
        img_root: str,
        max_length: int = 32,
        min_freq: int = 1,
    ):
        """
        annotations_path: JSON file: list of {"image": "xxx.jpg", "caption": "some text"}
        img_root: directory with images
        """
        super().__init__()
        self.img_root = img_root
        self.max_length = max_length

        with open(annotations_path, "r", encoding="utf-8") as f:
            self.samples: List[Dict[str, Any]] = json.load(f)

        # build vocab from captions
        self.vocab = Vocab(min_freq=min_freq)
        for s in self.samples:
            self.vocab.add_sentence(s["caption"])
        self.vocab.build()

        # basic image transform (for ResNet)
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        img_path = os.path.join(self.img_root, sample["image"])
        caption = sample["caption"]

        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        token_ids = self.vocab.encode(caption, self.max_length)
        length = len(token_ids)

        return {
            "image": image,
            "tokens": torch.tensor(token_ids, dtype=torch.long),
            "length": length,
        }

    @property
    def vocab_size(self) -> int:
        return self.vocab.size

    @staticmethod
    def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        # sort by length (descending) – good for RNN efficiency (optional)
        batch = sorted(batch, key=lambda x: x["length"], reverse=True)
        images = torch.stack([b["image"] for b in batch], dim=0)

        lengths = torch.tensor([b["length"] for b in batch], dtype=torch.long)
        max_len = lengths.max().item()

        padded_tokens = torch.full(
            (len(batch), max_len),
            fill_value=SPECIAL_TOKENS["<pad>"],
            dtype=torch.long,
        )

        for i, b in enumerate(batch):
            t = b["tokens"]
            padded_tokens[i, : t.size(0)] = t

        return {
            "images": images,
            "tokens": padded_tokens,
            "lengths": lengths,
        }
