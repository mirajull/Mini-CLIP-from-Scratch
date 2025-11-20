# 📘 Mini-CLIP: Contrastive Image–Text Retrieval from Scratch (PyTorch)

A minimal, fully open-source implementation of a **CLIP-style Vision–Language Model** trained from scratch on the **MS-COCO image–caption dataset**.  
This project demonstrates how contrastive multimodal learning can align images and text into a shared embedding space, enabling **cross-modal retrieval**.

---

## 🚀 Highlights

- **ViT-B/16** visual encoder + **Transformer** text encoder  
- **Symmetric InfoNCE** contrastive loss with **hard negative mining**  
- **Mixed-precision training** (optional)  
- **Full COCO support** (train/val splits)  
- **FAISS-powered retrieval evaluation** for instant Recall@K  
- **Efficient batching + padded text sequences**  
- Tiny toy dataset for instant debugging  
- 100% PyTorch — no external CLIP dependencies

---

# 🧠 Model Overview

### 🖼 Image Encoder
- ViT-B/16 backbone (ImageNet-pretrained) with frozen classification head  
- Linear projection into the shared embedding space  
- L2-normalized embeddings for cosine similarity  

### ✍️ Text Encoder
- Whitespace tokenizer + learnable vocabulary  
- Learned token + positional embeddings  
- Multi-layer Transformer encoder (GELU + dropout)  
- `[BOS]` state projected into the shared embedding space and normalized  

### 🔗 Contrastive Loss (InfoNCE)
For a batch size *N*:

- Compute similarity matrix **S = image_emb @ text_embᵀ**
- Apply cross-entropy loss in both directions (image→text & text→image)  
- Add a **margin-based hard negative term** that selects the top-*k* most confusing mismatched pairs every batch  
- Total loss = symmetric InfoNCE + weighted hard-negative penalty

### ⚡ Retrieval with FAISS
- Encoded image/text features are indexed with `faiss.IndexFlatIP`
- Instant **Image→Text** and **Text→Image** Recall@K using cosine similarity search  
- Requires `faiss-cpu` (installed via `pip install -r requirement.txt`)

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
