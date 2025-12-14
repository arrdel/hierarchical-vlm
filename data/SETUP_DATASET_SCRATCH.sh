#!/bin/bash

################################################################################
# 🎬 ActivityNet Dataset Setup - /media/scratch/adele
# Comprehensive workflow for extracting and organizing the dataset
################################################################################

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration
SCRATCH_DIR="/media/scratch/adele"
PROJECT_DIR="/home/adelechinda/home/projects/HierarchicalVLM"
DATA_DIR="${PROJECT_DIR}/data"

echo -e "${BLUE}════════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}   🎬 ActivityNet Dataset Setup${NC}"
echo -e "${CYAN}   Location: ${SCRATCH_DIR}${NC}"
echo -e "${BLUE}════════════════════════════════════════════════════════════════${NC}"
echo ""

# Step 1: Check download status
echo -e "${YELLOW}[Step 1/5] Checking download...${NC}"
ZIP_FILE="${SCRATCH_DIR}/activitynet1-3.zip"

if [ ! -f "${ZIP_FILE}" ]; then
    echo -e "${RED}✗ ZIP file not found at ${ZIP_FILE}${NC}"
    echo -e "${YELLOW}  Download in progress. Waiting...${NC}"
    while [ ! -f "${ZIP_FILE}" ]; do
        sleep 5
        echo -e "${YELLOW}  Checking...${NC}"
    done
fi

echo -e "${GREEN}✓ ZIP file found${NC}"

# Wait for download to complete
echo -e "${YELLOW}  Checking download completion...${NC}"
PREV_SIZE=0
STABLE_COUNT=0

while [ $STABLE_COUNT -lt 3 ]; do
    CURRENT_SIZE=$(stat -f%z "${ZIP_FILE}" 2>/dev/null || stat -c%s "${ZIP_FILE}" 2>/dev/null)
    SIZE_GB=$((CURRENT_SIZE / 1073741824))
    SIZE_MB=$(((CURRENT_SIZE % 1073741824) / 1048576))
    
    if [ "$CURRENT_SIZE" -eq "$PREV_SIZE" ]; then
        STABLE_COUNT=$((STABLE_COUNT + 1))
        echo -e "${GREEN}✓ Download complete: ${SIZE_GB}GB ${SIZE_MB}MB${NC}"
    else
        STABLE_COUNT=0
        echo -e "${YELLOW}  Downloading: ${SIZE_GB}GB ${SIZE_MB}MB...${NC}"
    fi
    
    PREV_SIZE="$CURRENT_SIZE"
    sleep 3
done

echo -e "${GREEN}✓ Download verified complete${NC}"
echo ""

# Step 2: Prepare extraction directory
echo -e "${YELLOW}[Step 2/5] Preparing extraction...${NC}"
EXTRACT_DIR="${SCRATCH_DIR}/activitynet"
mkdir -p "${EXTRACT_DIR}"
echo -e "${GREEN}✓ Extraction directory: ${EXTRACT_DIR}${NC}"
echo ""

# Step 3: Extract ZIP file
echo -e "${YELLOW}[Step 3/5] Extracting dataset (this may take 10-30 minutes)...${NC}"
echo -e "${YELLOW}  Source: ${ZIP_FILE}${NC}"
echo -e "${YELLOW}  Target: ${EXTRACT_DIR}${NC}"
echo ""

unzip -q "${ZIP_FILE}" -d "${EXTRACT_DIR}"

echo -e "${GREEN}✓ Extraction complete${NC}"

# Count extracted files
VIDEO_COUNT=$(find "${EXTRACT_DIR}" -type f \( -name "*.mp4" -o -name "*.mkv" -o -name "*.avi" \) 2>/dev/null | wc -l)
JSON_COUNT=$(find "${EXTRACT_DIR}" -type f -name "*.json" 2>/dev/null | wc -l)

echo -e "${GREEN}✓ Found ${VIDEO_COUNT} video files${NC}"
echo -e "${GREEN}✓ Found ${JSON_COUNT} annotation files${NC}"
echo ""

# Step 4: Create project symlinks
echo -e "${YELLOW}[Step 4/5] Setting up project structure...${NC}"

# Ensure data directory exists
mkdir -p "${DATA_DIR}"

# Create symlink for raw videos
RAW_LINK="${DATA_DIR}/raw"
if [ -L "${RAW_LINK}" ]; then
    rm "${RAW_LINK}"
fi
if [ -d "${RAW_LINK}" ] && [ ! -L "${RAW_LINK}" ]; then
    mv "${RAW_LINK}" "${RAW_LINK}.backup"
fi

ln -s "${EXTRACT_DIR}" "${RAW_LINK}"
echo -e "${GREEN}✓ Symlink: ${RAW_LINK} → ${EXTRACT_DIR}${NC}"
echo ""

# Step 5: Run validation and get stats
echo -e "${YELLOW}[Step 5/5] Validating dataset...${NC}"

cd "${PROJECT_DIR}"

if [ -f "scripts/organize_and_validate.py" ]; then
    echo -e "${GREEN}✓ Running validation...${NC}"
    python scripts/organize_and_validate.py --stats --data-root "${DATA_DIR}"
else
    echo -e "${YELLOW}⚠ organize_and_validate.py not found, skipping validation${NC}"
fi

echo ""
echo -e "${BLUE}════════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}✅ Dataset Setup Complete!${NC}"
echo -e "${BLUE}════════════════════════════════════════════════════════════════${NC}"

echo ""
echo -e "${CYAN}📊 Summary:${NC}"
echo -e "  ${GREEN}Scratch Location:${NC} ${SCRATCH_DIR}"
echo -e "  ${GREEN}Extract Location:${NC} ${EXTRACT_DIR}"
echo -e "  ${GREEN}Project Link:${NC} ${RAW_LINK}"
echo -e "  ${GREEN}Videos Found:${NC} ${VIDEO_COUNT}"
echo -e "  ${GREEN}Annotations:${NC} ${JSON_COUNT}"
echo ""

echo -e "${YELLOW}🚀 Next Steps:${NC}"
echo ""
echo -e "${GREEN}1. Organize videos into train/val/test splits:${NC}"
echo -e "   ${BLUE}python scripts/organize_and_validate.py --organize --data-root ${DATA_DIR}${NC}"
echo ""
echo -e "${GREEN}2. View detailed dataset statistics:${NC}"
echo -e "   ${BLUE}python scripts/organize_and_validate.py --stats --data-root ${DATA_DIR}${NC}"
echo ""
echo -e "${GREEN}3. Start training (single GPU):${NC}"
echo -e "   ${BLUE}python hierarchicalvlm/train/train_hierarchical.py \\${NC}"
echo -e "   ${BLUE}    --config configs/training_config.yaml \\${NC}"
echo -e "   ${BLUE}    --train-data ${DATA_DIR}/processed/training/videos \\${NC}"
echo -e "   ${BLUE}    --val-data ${DATA_DIR}/processed/validation/videos${NC}"
echo ""
echo -e "${GREEN}4. Start training (multi-GPU - 4 GPUs):${NC}"
echo -e "   ${BLUE}python -m torch.distributed.launch --nproc_per_node=4 \\${NC}"
echo -e "   ${BLUE}    hierarchicalvlm/train/train_hierarchical.py \\${NC}"
echo -e "   ${BLUE}    --config configs/training_config.yaml \\${NC}"
echo -e "   ${BLUE}    --train-data ${DATA_DIR}/processed/training/videos \\${NC}"
echo -e "   ${BLUE}    --val-data ${DATA_DIR}/processed/validation/videos${NC}"
echo ""
echo -e "${GREEN}5. Monitor training progress:${NC}"
echo -e "   ${BLUE}tensorboard --logdir ./runs${NC}"
echo ""

echo -e "${BLUE}════════════════════════════════════════════════════════════════${NC}"
