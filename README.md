# 📘 Mini-CLIP: Contrastive Image–Text Retrieval from Scratch (PyTorch)

A minimal, fully open-source implementation of a **CLIP-style Vision–Language Model** trained from scratch on the **MS-COCO image–caption dataset**.  
This project demonstrates how contrastive multimodal learning can align images and text into a shared embedding space, enabling **cross-modal retrieval**.

---

## 🚀 Highlights

- **ResNet-18** visual encoder + **BiGRU** text encoder  
- **Symmetric InfoNCE contrastive loss**  
- **Mixed-precision training** (optional)  
- **Full COCO support** (train/val splits)  
- **Efficient batching + padded text sequences**  
- **Recall@K metrics for retrieval evaluation**  
- Tiny toy dataset for instant debugging  
- 100% PyTorch — no external CLIP dependencies

---

# 🧠 Model Overview

### 🖼 Image Encoder
- ResNet-18 backbone (pretrained on ImageNet)  
- 512 → 256 projection head  
- L2-normalized embeddings  

### ✍️ Text Encoder
- Simple whitespace tokenizer + custom vocabulary  
- Embedding layer  
- Bi-Directional GRU  
- 1024 → 256 projection  
- L2-normalized embeddings  

### 🔗 Contrastive Loss (InfoNCE)
For a batch size *N*:

- Compute similarity matrix **S = image_emb @ text_embᵀ**
- Apply cross-entropy loss in both directions:
  - image → text  
  - text → image  
- Total loss = average of both

---

# 📦 Dataset

This project uses:

### **MS-COCO 2014**
- ~82k training images (`train2014/`)  
- ~40k validation images (`val2014/`)  
- Each image has 5 human-written captions  

COCO is converted into a simple JSON format:

```json
[
  {
    "image": "COCO_train2014_000000000009.jpg",
    "caption": "A man riding a bike on a city street."
  }
]
```

---

# 🏋️ Training & Evaluation

Train on COCO subset:

```bash
python -m src.train --config configs/default.yaml
```

Evaluate on COCO val:

```bash
python -m src.eval \
  --config configs/eval_coco.yaml \
  --checkpoint checkpoints/clip_epoch10.pt
```

---

# 🛠 Project Structure

```
Mini-CLIP-from-Scratch/
|-- src/
|   |-- model.py         # Image/Text encoders + CLIP head
|   |-- dataset.py       # COCO + toy dataset loaders
|   |-- train.py         # Training loop entrypoint
|   |-- eval.py          # Retrieval evaluation script
|   `-- utils.py         # Helper utilities
|-- configs/
|   `-- default.yaml     # Training configuration
|-- data/
|   |-- coco/            # Raw COCO images (gitignored)
|   |-- images/          # Toy/demo images
|   `-- train_captions.json
|-- checkpoints/         # Saved CLIP weights
|-- scripts/
|   |-- make_toy_dataset.py
|   `-- run_train.sh
|-- requirement.txt      # Python dependencies
`-- README.md
```
