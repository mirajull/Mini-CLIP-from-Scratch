import os
import random
from typing import Any, Dict

import numpy as np
import torch
import yaml


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def save_checkpoint(state: Dict[str, Any], ckpt_dir: str, filename: str):
    ensure_dir(ckpt_dir)
    path = os.path.join(ckpt_dir, filename)
    torch.save(state, path)
    print(f"[INFO] Saved checkpoint to {path}")
