#!/bin/bash

################################################################################
# 🎬 ActivityNet Dataset Setup Script
# Comprehensive workflow for downloading, extracting, and organizing the dataset
################################################################################

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
DOWNLOAD_DIR="${HOME}/Downloads"
SCRATCH_DIR="/scratch"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${PROJECT_DIR}/data"

echo -e "${BLUE}════════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}   🎬 ActivityNet Dataset Setup${NC}"
echo -e "${BLUE}════════════════════════════════════════════════════════════════${NC}"
echo ""

# Step 1: Check download status
echo -e "${YELLOW}[Step 1/5] Checking download status...${NC}"
if [ -f "${DOWNLOAD_DIR}/activitynet1-3.zip" ]; then
    SIZE=$(ls -lh "${DOWNLOAD_DIR}/activitynet1-3.zip" | awk '{print $5}')
    echo -e "${GREEN}✓ Download found: ${SIZE}${NC}"
    
    # Check if download is complete (should be ~16.6GB)
    ACTUAL_SIZE=$(stat -f%z "${DOWNLOAD_DIR}/activitynet1-3.zip" 2>/dev/null || stat -c%s "${DOWNLOAD_DIR}/activitynet1-3.zip" 2>/dev/null)
    if [ "$ACTUAL_SIZE" -lt 16000000000 ]; then
        echo -e "${YELLOW}⚠ Download incomplete. Size: $((ACTUAL_SIZE / 1073741824))GB (need ~16.6GB)${NC}"
        echo -e "${YELLOW}  Waiting for download to complete...${NC}"
        while [ "$ACTUAL_SIZE" -lt 16000000000 ]; do
            sleep 10
            ACTUAL_SIZE=$(stat -f%z "${DOWNLOAD_DIR}/activitynet1-3.zip" 2>/dev/null || stat -c%s "${DOWNLOAD_DIR}/activitynet1-3.zip" 2>/dev/null)
            echo -e "${YELLOW}  Current size: $((ACTUAL_SIZE / 1073741824))GB${NC}"
        done
    fi
else
    echo -e "${RED}✗ Download not found at ${DOWNLOAD_DIR}/activitynet1-3.zip${NC}"
    echo -e "${YELLOW}  Run the download command first:${NC}"
    echo -e "${YELLOW}  curl -L -o ~/Downloads/activitynet1-3.zip \\${NC}"
    echo -e "${YELLOW}    https://www.kaggle.com/api/v1/datasets/download/valuejack/activitynet1-3${NC}"
    exit 1
fi

echo ""

# Step 2: Prepare extraction directory
echo -e "${YELLOW}[Step 2/5] Preparing extraction directory...${NC}"
if [ ! -d "${SCRATCH_DIR}" ]; then
    echo -e "${YELLOW}⚠ /scratch not available, using project data directory instead${NC}"
    EXTRACT_DIR="${DATA_DIR}/raw"
else
    EXTRACT_DIR="${SCRATCH_DIR}/activitynet"
    echo -e "${GREEN}✓ Using /scratch: ${EXTRACT_DIR}${NC}"
fi

mkdir -p "${EXTRACT_DIR}"
echo -e "${GREEN}✓ Extraction directory ready: ${EXTRACT_DIR}${NC}"

echo ""

# Step 3: Extract ZIP file
echo -e "${YELLOW}[Step 3/5] Extracting dataset (this may take 10-30 minutes)...${NC}"
echo -e "${YELLOW}  Source: ${DOWNLOAD_DIR}/activitynet1-3.zip${NC}"
echo -e "${YELLOW}  Target: ${EXTRACT_DIR}${NC}"

if command -v pv &> /dev/null; then
    # Use pv for progress if available
    pv "${DOWNLOAD_DIR}/activitynet1-3.zip" | unzip -q - -d "${EXTRACT_DIR}"
else
    # Standard unzip
    unzip -q "${DOWNLOAD_DIR}/activitynet1-3.zip" -d "${EXTRACT_DIR}"
fi

echo -e "${GREEN}✓ Extraction complete${NC}"

# Count extracted files
VIDEO_COUNT=$(find "${EXTRACT_DIR}" -type f \( -name "*.mp4" -o -name "*.mkv" -o -name "*.avi" \) 2>/dev/null | wc -l)
echo -e "${GREEN}✓ Found ${VIDEO_COUNT} video files${NC}"

echo ""

# Step 4: Create symlinks in project data directory
echo -e "${YELLOW}[Step 4/5] Organizing into project structure...${NC}"

mkdir -p "${DATA_DIR}"
RAW_LINK="${DATA_DIR}/raw"

if [ -L "${RAW_LINK}" ]; then
    echo -e "${YELLOW}⚠ Symlink already exists, removing old one${NC}"
    rm "${RAW_LINK}"
fi

if [ -d "${RAW_LINK}" ] && [ ! -L "${RAW_LINK}" ]; then
    echo -e "${YELLOW}⚠ Raw directory exists (not a symlink), moving to backup${NC}"
    mv "${RAW_LINK}" "${RAW_LINK}.backup"
fi

ln -s "${EXTRACT_DIR}" "${RAW_LINK}"
echo -e "${GREEN}✓ Symlink created: ${RAW_LINK} → ${EXTRACT_DIR}${NC}"

echo ""

# Step 5: Validate and prepare
echo -e "${YELLOW}[Step 5/5] Validating dataset structure...${NC}"

cd "${PROJECT_DIR}"

# Check if organize_and_validate.py exists
if [ -f "scripts/organize_and_validate.py" ]; then
    echo -e "${GREEN}✓ Running validation script...${NC}"
    python scripts/organize_and_validate.py --stats --data-root "${DATA_DIR}"
else
    echo -e "${YELLOW}⚠ organize_and_validate.py not found${NC}"
fi

echo ""
echo -e "${BLUE}════════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}✅ Dataset Setup Complete!${NC}"
echo -e "${BLUE}════════════════════════════════════════════════════════════════${NC}"

echo ""
echo -e "${YELLOW}📊 Summary:${NC}"
echo -e "  Extract Location: ${EXTRACT_DIR}"
echo -e "  Project Link: ${RAW_LINK}"
echo -e "  Videos Found: ${VIDEO_COUNT}"
echo ""

echo -e "${YELLOW}🚀 Next Steps:${NC}"
echo ""
echo -e "${GREEN}1. Organize videos into train/val/test:${NC}"
echo -e "   ${BLUE}python scripts/organize_and_validate.py --organize --data-root ${DATA_DIR}${NC}"
echo ""
echo -e "${GREEN}2. View dataset statistics:${NC}"
echo -e "   ${BLUE}python scripts/organize_and_validate.py --stats --data-root ${DATA_DIR}${NC}"
echo ""
echo -e "${GREEN}3. Start training:${NC}"
echo -e "   ${BLUE}python hierarchicalvlm/train/train_hierarchical.py \\${NC}"
echo -e "   ${BLUE}    --config configs/training_config.yaml \\${NC}"
echo -e "   ${BLUE}    --train-data ${DATA_DIR}/processed/training/videos \\${NC}"
echo -e "   ${BLUE}    --val-data ${DATA_DIR}/processed/validation/videos${NC}"
echo ""
echo -e "${GREEN}4. Monitor training:${NC}"
echo -e "   ${BLUE}tensorboard --logdir ./runs${NC}"
echo ""

echo -e "${BLUE}════════════════════════════════════════════════════════════════${NC}"
