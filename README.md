# Mini-CLIP: Contrastive Image–Text Retrieval in PyTorch

## Overview
This project implements a lightweight CLIP-style Vision-Language Model
trained from scratch for image–text retrieval. The model learns a joint
embedding space for images and captions using a contrastive InfoNCE loss.

## Dataset
- Default: MS-COCO / Flickr30k (or any image-caption JSON)
- Each sample: `{ "image_path": ..., "caption": ... }`

## Model
- Vision encoder: ResNet-50 (pretrained or from scratch)
- Text encoder: Transformer / BiLSTM
- Projection into a shared embedding space with L2-normalization
- Loss: symmetric cross-entropy over image-to-text and text-to-image scores

## Training
- Framework: PyTorch
- Batch size: 128 (adjustable)
- Optimizer: AdamW
- Mixed precision: supported (optional)

```bash
pip install -r requirements.txt
python -m src.train --config configs/default.yaml
