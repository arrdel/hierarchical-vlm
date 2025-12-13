# HierarchicalVLM - Project Structure

## ✅ Clean Project Organization

### Root Files (Active)
- `setup.py` - Package setup for installation
- `SETUP_DATASET_SCRATCH.sh` - Dataset extraction and organization script
- `RUN_TESTS.sh` - Test runner script
- `README.md` - Project documentation
- `requirements.txt` - Python dependencies

### Directories

```
├── .trash/                       # Archived/unused scripts
│   ├── SETUP_DATASET.sh
│   ├── MONITOR_DOWNLOAD.sh
│   ├── START_TRAINING.sh
│   ├── train_example.py
│   └── training.log
│
├── hierarchicalvlm/              # Main package
│   ├── __init__.py
│   ├── constants.py
│   ├── utils.py
│   ├── video_conversation.py
│   ├── data/                     # Data loading
│   │   ├── __init__.py
│   │   └── activitynet_features_loader.py
│   ├── model/                    # Model architecture
│   │   ├── __init__.py
│   │   ├── longvlm.py
│   │   ├── consolidate.py
│   │   ├── make_delta.py
│   │   ├── merge.py
│   │   └── utils.py
│   ├── train/                    # Training scripts
│   │   ├── __init__.py
│   │   ├── train_features.py     # ⭐ Main training script (2-GPU DDP + W&B)
│   │   ├── llava_trainer.py
│   │   ├── train.py
│   │   ├── train_mem.py
│   │   └── llama_flash_attn_monkey_patch.py
│   ├── eval/                     # Evaluation
│   │   ├── __init__.py
│   │   ├── model_utils.py
│   │   ├── run_inference_benchmark.py
│   │   └── run_inference_qa.py
│   └── quantitative_evaluation/
│
├── configs/                      # Training configurations
│   └── training_config.yaml
│
├── data/                         # Dataset symlink
│   └── raw → /media/scratch/adele/activitynet/
│
├── runs/                         # Training outputs
│   └── feature_training_v1_2gpu/  # Current training run
│
├── tests/                        # Test suite
│   ├── __init__.py
│   ├── test_attention.py
│   ├── test_domain_modules.py
│   ├── test_hierarchical_model.py
│   └── test_token_merging.py
│
├── datasets/                     # Dataset lists
│   └── anet/
│
├── scripts/                      # Utility scripts
│   ├── apply_delta.py
│   ├── convert_instruction_json_to_training_format.py
│   ├── filtering_pkl.py
│   └── save_features.py
│
└── docs/                         # Documentation
```

## 📊 Dataset Structure

```
/media/scratch/adele/activitynet/ActivityNet-13/
├── gt.json                       (9.3 MB - annotations)
├── train/
│   └── train/                    (9,032 .npy feature files)
└── test/
    └── test/                     (4,427 .npy feature files)
```

## 🚀 Quick Start - Training

### Single GPU
```bash
cd /home/adelechinda/home/projects/HierarchicalVLM
conda activate hierarchical_vlm
python hierarchicalvlm/train/train_features.py \
    --batch-size 32 \
    --num-epochs 50
```

### Multi-GPU (2 GPUs) with W&B Logging
```bash
conda run -n hierarchical_vlm torchrun --nproc_per_node=2 \
    hierarchicalvlm/train/train_features.py \
    --batch-size 32 \
    --num-epochs 50 \
    --wandb-project "hierarchical-vlm-features" \
    --wandb-run-name "activitynet_2gpu_training"
```

## 🧹 Cleanup Summary

**Archived to `.trash/`:**
- ✅ SETUP_DATASET.sh (original dataset setup)
- ✅ MONITOR_DOWNLOAD.sh (download monitoring)
- ✅ START_TRAINING.sh (old training launcher)
- ✅ train_example.py (example training code)
- ✅ training.log (old logs)

**Kept Active:**
- ✅ `SETUP_DATASET_SCRATCH.sh` - For dataset organization
- ✅ `RUN_TESTS.sh` - For testing
- ✅ `hierarchicalvlm/train/train_features.py` - Main training script

## 📋 Training Features

✅ Multi-GPU support (2 GPUs)
✅ Distributed Data Parallel (DDP)
✅ Weights & Biases integration
✅ Verbose logging at batch and epoch level
✅ Gradient norm tracking
✅ Learning rate scheduling
✅ Checkpoint saving (best + periodic)
✅ Feature-based training (pre-extracted)
✅ Attention masking support

## 🎯 Ready to Train!

All setup complete. Project is organized and ready for training.

Generated: December 13, 2025
