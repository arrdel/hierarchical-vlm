#!/bin/bash

###############################################################################
# HierarchicalVLM - Multi-GPU Training Script (Observable, Foreground)
# This script starts training on 2 GPUs with real-time console output
# Weights & Biases logging is enabled with online mode
###############################################################################

set -e  # Exit on error

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🚀 HierarchicalVLM - Multi-GPU Training (Observable)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Navigate to project root
PROJECT_ROOT="/home/adelechinda/home/projects/HierarchicalVLM"
cd "$PROJECT_ROOT"

echo "📁 Project Root: $PROJECT_ROOT"
echo ""

# Check if conda environment exists
echo "🔍 Checking conda environment..."
if conda env list | grep -q "hierarchical_vlm"; then
    echo "✅ Found hierarchical_vlm environment"
else
    echo "❌ ERROR: hierarchical_vlm environment not found"
    exit 1
fi
echo ""

# Check if GPUs are available
echo "🎮 Checking GPUs..."
if ! command -v nvidia-smi &> /dev/null; then
    echo "❌ ERROR: nvidia-smi not found. CUDA/GPUs not available"
    exit 1
fi

GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
echo "✅ Found $GPU_COUNT GPUs"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo ""

# Check training script exists
echo "📜 Checking training script..."
if [ ! -f "hierarchicalvlm/train/train_features.py" ]; then
    echo "❌ ERROR: train_features.py not found"
    exit 1
fi
echo "✅ Training script found"
echo ""

# Check dataset exists
echo "📊 Checking dataset..."
if [ ! -d "/media/scratch/adele/activitynet/ActivityNet-13" ]; then
    echo "❌ ERROR: Dataset not found at /media/scratch/adele/activitynet/ActivityNet-13"
    exit 1
fi
TRAIN_COUNT=$(find /media/scratch/adele/activitynet/ActivityNet-13/train/train -name "*.npy" | wc -l)
VAL_COUNT=$(find /media/scratch/adele/activitynet/ActivityNet-13/test/test -name "*.npy" | wc -l)
echo "✅ Dataset found"
echo "   Training samples: $TRAIN_COUNT"
echo "   Validation samples: $VAL_COUNT"
echo ""

# Training configuration
BATCH_SIZE=32
NUM_EPOCHS=50
OUTPUT_DIR="./runs/feature_training_$(date +%Y%m%d_%H%M%S)"
WANDB_PROJECT="hierarchical-vlm"
WANDB_RUN_NAME="feature_training_$(date +%Y%m%d_%H%M%S)"

echo "⚙️  Training Configuration:"
echo "   Batch Size: $BATCH_SIZE"
echo "   Epochs: $NUM_EPOCHS"
echo "   Output Dir: $OUTPUT_DIR"
echo "   W&B Project: $WANDB_PROJECT"
echo "   W&B Run Name: $WANDB_RUN_NAME"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"
echo "✅ Output directory created: $OUTPUT_DIR"
echo ""

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🔴 STARTING TRAINING - OBSERVABLE IN FOREGROUND"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "📌 TIPS:"
echo "   • All output will appear below in real-time"
echo "   • Check W&B dashboard for detailed metrics"
echo "   • Press Ctrl+C to stop training"
echo "   • Use separate terminal for: watch -n 1 nvidia-smi"
echo ""

# Start training with torchrun
conda run -n hierarchical_vlm \
    torchrun --nproc_per_node=2 \
    hierarchicalvlm/train/train_features.py \
    --batch-size "$BATCH_SIZE" \
    --num-epochs "$NUM_EPOCHS" \
    --output-dir "$OUTPUT_DIR" \
    --wandb-project "$WANDB_PROJECT" \
    --wandb-run-name "$WANDB_RUN_NAME"

# If we reach here, training completed successfully
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ TRAINING COMPLETED SUCCESSFULLY"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "📁 Output files saved to: $OUTPUT_DIR"
echo ""
echo "Checkpoints:"
ls -lh "$OUTPUT_DIR"/*.pt 2>/dev/null || echo "   No .pt files found"
echo ""
echo "📊 View results on W&B: https://wandb.ai/your-workspace/$WANDB_PROJECT"
echo ""
