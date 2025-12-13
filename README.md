# HierarchicalVLM

**Adaptive Long Video Understanding with Efficient Attention and Domain-Specific Adaptation**

*See more, understand better, compute less.*

## Overview

HierarchicalVLM builds upon LongVLM to create a more efficient, adaptable, and domain-aware system for understanding long videos. The project focuses on three key innovations:

### 1. 🚀 Efficient Attention Mechanisms
- **Sparse Attention Patterns**: Strided and local+global attention for linear complexity
- **Cross-Attention Between Memory and Features**: Replaces naive concatenation with learned fusion
- **Linear Attention Variants**: Implements Mamba and Performer for sub-quadratic complexity

### 2. 🎯 Domain-Specific Fine-Tuning
- **Modular Domain Adapters**: LoRA-based parameter-efficient fine-tuning
- **Task-Specific Heads**: 
  - Action Detection
  - Visual Question Answering
  - Video Captioning
- **Multi-Domain Support**: Sports, tutorials, news, and more

### 3. 📊 Adaptive Token Merging
- **Motion-Aware Merging**: Uses optical flow to dynamically adjust compression
- **Saliency-Based Merging**: Preserves important regions based on visual saliency
- **Content-Aware Ratios**: High-motion scenes keep more tokens, static scenes compress

## Project Structure

```
HierarchicalVLM/
├── hierarchicalvlm/
│   ├── attention/
│   │   ├── sparse/           # Strided, local+global attention
│   │   └── linear/           # Performer, Mamba implementations
│   ├── domain_modules/
│   │   ├── adapters/         # LoRA, prefix tuning, adapter modules
│   │   └── heads/            # Task-specific prediction heads
│   ├── token_merging/
│   │   ├── motion/           # Optical flow and motion analysis
│   │   └── saliency/         # Saliency detection and weighting
│   ├── model/                # Core hierarchical model
│   ├── train/                # Training utilities
│   ├── eval/                 # Evaluation scripts
│   └── utils/                # Helper functions
├── configs/                  # Configuration files
│   ├── attention/
│   ├── domain_modules/
│   └── token_merging/
├── scripts/                  # Data processing and utilities
├── experiments/              # Results and logs
└── tests/                    # Unit tests
```

## Key Features

- **Modular Design**: Each component (attention, domain, merging) is independent
- **Parameter Efficiency**: LoRA adapters for domain-specific tuning
- **Adaptive Compression**: Dynamic token merging based on content
- **Multiple Attention Mechanisms**: Choose from sparse, linear, or hybrid approaches
- **Task-Agnostic**: Supports multiple downstream tasks with shared backbone

## Installation

```bash
cd HierarchicalVLM
pip install -r requirements.txt
```

## Quick Start

```python
from hierarchicalvlm.attention.sparse import StridedAttention
from hierarchicalvlm.domain_modules.adapters import LoRAAdapter
from hierarchicalvlm.token_merging.motion import MotionAwareTokenMerge

# Initialize components
sparse_attn = StridedAttention(stride=4)
domain_adapter = LoRAAdapter(dim=768, rank=8)
token_merger = MotionAwareTokenMerge()
```

## Implementation Plan

### Phase 1: Efficient Attention ✅ (In Progress)
- [ ] Sparse attention patterns implementation
- [ ] Cross-attention module
- [ ] Linear attention (Performer, Mamba)

### Phase 2: Domain Modules (Next)
- [ ] LoRA adaptation framework
- [ ] Task-specific heads
- [ ] Multi-domain support

### Phase 3: Adaptive Token Merging
- [ ] Motion estimation pipeline
- [ ] Saliency detection
- [ ] Adaptive merging strategy

## References

- **LongVLM**: Base architecture for long video understanding
- **LoRA**: Efficient parameter-tuning for fine-grained domain adaptation
- **Performer**: Fast transformers with linear attention
- **Mamba**: State space models for sequence modeling
- **Optical Flow**: Motion-based content understanding

## Citation

If you use HierarchicalVLM, please cite:

```bibtex
@misc{hierarchicalvlm2025,
  title={HierarchicalVLM: Adaptive Long Video Understanding with Efficient Attention and Domain-Specific Adaptation},
  author={Adele Chinda},
  year={2025}
}
```

<!-- ## License

[Add your chosen license]

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](docs/CONTRIBUTING.md) for guidelines.

## Contact

[Add your contact information] -->
