# HierarchicalVLM: Efficient Long-Context Video Understanding

Adele Chinda, Desire Emeka, Maryam Koya, Nita Ngozi Ezekwem

[![Code](https://img.shields.io/badge/💻-Code-black)](https://github.com/arrdel/hierarchical-vlm)
[![W&B Logs](https://img.shields.io/badge/📊-Logs-orange)](https://wandb.ai/el_chindah/hierarchical-vlm)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Architecture](https://img.shields.io/badge/📐-Architecture-purple)](docs/static/images/teaser.png)

## 🚀 Quick Stats

| Metric | Value |
|--------|-------|
| **Temporal Consistency** | +28.4%  |
| **Activity Classification** | +34.2%  |
| **Temporal Localization** | +41.6%  |
| **Memory Reduction** | 130x  |
| **Model Size** | 15.2M params (298 MB)  |
| **Training Speed** | 1,575 samples/sec  |

## Overview

HierarchicalVLM addresses the fundamental challenge of understanding long video sequences efficiently. We combine **temporal contrastive learning** with **hierarchical feature aggregation** to reduce complexity from O(T²) to O(T) while maintaining strong performance on ActivityNet-1.3.

**Key Innovation:** Multi-scale attention-weighted pooling that progressively compresses temporal sequences while preserving semantic content, enabling processing of 1000+ frame videos without quadratic memory requirements.

## Method at a Glance

**📐 [View Full Architecture Diagram](docs/static/images/teaser.png)**

### Phase 1️⃣: Temporal Contrastive Learning
- **Temporal Loss:** Enforce similarity between consecutive frames
  - L_temporal = Σ(1 - cos(h_i, h_{i+1})) for i=1 to T-1
  - Loss function encourages adjacent frames to have similar embeddings
- **Collapse Prevention:** Batch-level variance regularization
  - L_reg = max(0, τ - std_batch(H))
  - Prevents all frames from collapsing to identical representations
  - Critical for stable training (without it, collapse occurs within 5 epochs)

### Phase 2️⃣: Hierarchical Aggregation
- **Attention-Weighted Pooling:** Multi-scale temporal reduction
  - h^(l)_j = Σ(α_i^(l) * h^(l-1)_i) / Σ(α_i^(l))
  - Learns what to preserve at each hierarchical level
- **Temporal Pyramid:** 250 → 125 → 62 → 31 → 15 → 7 → 3 → 1 frames
  - **130× memory reduction** (from 62,500 to 484 operations)
  - Enables processing of 1000+ frame videos

### Phase 3️⃣: Vision-Language Alignment (Optional)
- Align visual and textual embeddings for multimodal learning

## 📊 Results

| Baseline | Temporal Consistency | Activity Classification | Localization (IoU@0.5) |
|----------|---------------------|-------------------------|----------------------|
| Direct Transformer | 0.582 | 0.672 | 0.514 |
| TSN | 0.598 | 0.681 | 0.527 |
| MoCo v2 | 0.642 | 0.715 | 0.589 |
| VideoClip | 0.691 | 0.758 | 0.627 |
| **HierarchicalVLM** | **0.747** ✓ | **0.841** ✓ | **0.728** ✓ |

### Ablation Study
| Component | Temporal Consistency | Classification |
|-----------|---------------------|-----------------|
| Baseline | 0.582 | 0.672 |
| + Temporal Contrastive | 0.724 (+14.2%) | 0.801 (+19.2%) |
| + Collapse Prevention | 0.731 (+0.7%) | 0.812 (+1.4%) |
| + Hierarchical Aggregation | 0.745 (+1.4%) | 0.823 (+1.4%) |
| + Vision-Language | **0.747** (+0.2%) | **0.841** (+2.2%) |

## 📁 Project Structure

```
hierarchicalvlm/              # Core implementation
├── model/                    # Model architecture
├── train/                    # Training scripts
├── eval/                     # Evaluation code
└── utils/                    # Utilities

quantitative_evaluation/      # Benchmarking
configs/                      # Configuration files
datasets/anet/                # ActivityNet-1.3
docs/                         # Documentation & project page
```

## 📈 Dataset: ActivityNet-1.3

- **13,459 videos** (9,032 train, 4,427 val) with 13,929 action annotations
- **200 activity classes** across diverse domains
- **2048-D C3D features** at 1 FPS (avg. 250 frames/video)



## ⚡ Installation & Quick Start

### Setup (5 minutes)
```bash
# Clone and setup
git clone https://github.com/arrdel/hierarchical-vlm.git
cd HierarchicalVLM

# Create environment
python3 -m venv venv
source venv/bin/activate

# Install
pip install -r requirements.txt
```

### Training
```bash
# Single GPU
python hierarchicalvlm/train/train.py \
  --config configs/train_config.yaml \
  --data_path datasets/anet/

# Multi-GPU (2x RTX 4090)
torchrun --nproc_per_node=2 hierarchicalvlm/train/train.py \
  --config configs/train_config.yaml \
  --data_path datasets/anet/
```

### Evaluation
```bash
# Benchmark evaluation
bash quantitative_evaluation/evaluate_benchmark.sh \
  --model_path checkpoint.pt \
  --data_path datasets/anet/
```


## 📚 References

This work builds on:
- **Video Understanding:** C3D, TSN, VideoClip
- **Temporal Learning:** MoCo, SimCLR, Temporal Contrastive Learning
- **Efficient Transformers:** Performer, Mamba, Linear Attention

<!-- ## 📖 Citation 

```bibtex
@article{chinda2024hierarchicalvlm,
  title={HierarchicalVLM: Efficient Long-Context Video Understanding via Hierarchical Temporal Aggregation},
  author={Chinda, Adele and Azumah, Richmond and Venkateswara, Hemanth Demakethepalli},
  journal={arXiv preprint arXiv:2412.xxxxx},
  year={2024}
}
``` -->


## 📄 License

MIT License - See [LICENSE](LICENSE) for details.

---



