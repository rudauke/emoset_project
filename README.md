# EmoSet Emotion Classification

[![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/gist.github.com/rudauke/61d4c8126b698c53efe7dbdce92161db)

Fine-tunes vision transformers on EmoSet-118K for 8-class emotion classification using two-phase transfer learning. Built to run on free-tier GPUs (Molab / Colab).

## The Problem

Standard vision models need adaptation for emotion recognition - identifying "Awe" or "Contentment" from pixels requires interpreting lighting, color, and context, not just objects. This pipeline provides reproducible benchmarking of ViT, ConvNeXt, and EfficientNet architectures with proper validation splits, all within a reactive Marimo notebook.

## Quick Start

```bash
cd emotion_classification
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
marimo run emoset_project.py
```

## Usage

Open the notebook and use the UI to select:
- **Model**: vit_small, vit_base, convnext_tiny, convnext_base, efficientnetv2_rw_s, efficientnetv2_rw_m, tf_efficientnetv2_l
- **Batch size**, learning rate, weight decay, input resolution
- **Warmup epochs** (frozen backbone) vs fine-tune epochs

Click **Start Training** to run two-phase transfer learning.

## Results

| Model | Test Acc | Macro F1 | Params | Epoch Time (RTX 6000) |
|-------|----------|----------|--------|----------------------|
| **convnext_tiny** | **75.0%** | **0.76** | 28M | ~60s |
| vit_small_patch16_224 | 73.2% | 0.74 | 22M | ~55s |
| efficientnetv2_rw_s | 71.8% | 0.72 | 21M | ~50s |
| vit_base_patch16_224 | 74.1% | 0.75 | 86M | ~180s |

*All models trained on 10% stratified subset (~11K images), 80/10/10 split, 10 epochs (2 warmup + 8 fine-tune), batch size 128, LR 1e-4, weight decay 0.1.*

### Per-Class Performance (convnext_tiny)

| Emotion | Precision | Recall | F1 | Support |
|---------|-----------|--------|-----|---------|
| Disgust | 0.85 | 0.83 | 0.84 | 84 |
| Excitement | 0.80 | 0.84 | 0.82 | 158 |
| Anger | 0.80 | 0.82 | 0.81 | 85 |
| Awe | 0.79 | 0.75 | 0.77 | 120 |
| Fear | 0.77 | 0.76 | 0.77 | 108 |
| Sadness | 0.74 | 0.71 | 0.72 | 102 |
| Amusement | 0.71 | 0.69 | 0.70 | 157 |
| Contentment | 0.59 | 0.64 | 0.62 | 131 |
| **Macro Avg** | **0.76** | **0.75** | **0.76** | **945** |

Contentment and Amusement are the hardest — visually subtle, often confused with each other.

## Two-Phase Transfer Learning

```
Phase 1 — Warmup (frozen backbone)          Phase 2 — Fine-tune (differential LR)
├── Epochs: 2                                 ├── Epochs: 8
├── Backbone: frozen                          ├── Backbone: LR × 0.1 (1e-5)
├── Head: LR 1e-3                             └── Head: LR 1e-3
└── Goal: initialize head                     Goal: adapt features to emotion
```

Without pretrained weights + warmup: ~45% accuracy.  
With ImageNet pretrained + two-phase: **~75%** — a 30 point jump.

## Optimizations for Free GPUs

- **AdamW** with weight decay (prevents memorization on small subset)
- **Cosine Annealing LR** — smooth decay for finer convergence
- **AMP (Automatic Mixed Precision)** — 2× speedup, half memory
- **torch.compile** — JIT optimizes the graph after first epoch
- **10% stratified subset** — keeps class balance, fits in memory

## Dataset

- **EmoSet-118K** — 118,000 images, 8 balanced emotion classes
- **Source**: `Woleek/EmoSet-118K` on Hugging Face Datasets
- **Split used**: 10% stratified sample → 80/10/10 train/val/test
- **Classes**: amusement, awe, contentment, excitement, anger, disgust, fear, sadness

## Requirements

```bash
pip install -e ".[dev]"
```

Dependencies (from `pyproject.toml`):
- `torch>=2.12`
- `timm>=1.0`
- `datasets>=2.19`
- `marimo>=0.23`
- `scikit-learn>=1.3`
- `numpy>=1.26`

## Hardware Notes

Developed on **Molab (NVIDIA RTX 6000, 24GB VRAM)**.  
Also runs on **Google Colab (T4, 16GB)** with `vit_small` / `convnext_tiny` at batch size 64.

Training time: ~10 minutes for 10 epochs on convnext_tiny.

## Project Structure

```
emotion_classification/
├── emoset_project.py      # Main Marimo notebook (reactive UI + training loop)
├── pyproject.toml         # Dependencies & metadata
├── emoset_project.pdf     # Full project report
└── README.md
```

## License

MIT