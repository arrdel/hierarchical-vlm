#!/bin/bash

################################################################################
# 📊 Dataset Download Monitor
# Real-time monitoring of the ActivityNet download and setup progress
################################################################################

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
RED='\033[0;31m'
NC='\033[0m'

SCRATCH_DIR="/media/scratch/adele"
ZIP_FILE="${SCRATCH_DIR}/activitynet1-3.zip"
TARGET_SIZE=16600000000  # 16.6GB in bytes

while true; do
    clear
    
    echo -e "${BLUE}════════════════════════════════════════════════════════════════${NC}"
    echo -e "${CYAN}📊 ActivityNet Dataset Download Monitor${NC}"
    echo -e "${BLUE}════════════════════════════════════════════════════════════════${NC}"
    echo ""
    
    if [ ! -f "${ZIP_FILE}" ]; then
        echo -e "${YELLOW}⏳ Waiting for download to start...${NC}"
        sleep 5
        continue
    fi
    
    # Get current file size
    CURRENT_SIZE=$(stat -c%s "${ZIP_FILE}" 2>/dev/null || stat -f%z "${ZIP_FILE}" 2>/dev/null)
    PERCENT=$((CURRENT_SIZE * 100 / TARGET_SIZE))
    
    # Format sizes
    SIZE_GB=$((CURRENT_SIZE / 1073741824))
    SIZE_MB=$(((CURRENT_SIZE % 1073741824) / 1048576))
    TARGET_GB=$((TARGET_SIZE / 1073741824))
    TARGET_MB=$(((TARGET_SIZE % 1073741824) / 1048576))
    
    # Create progress bar
    BAR_LENGTH=50
    FILLED=$((PERCENT * BAR_LENGTH / 100))
    EMPTY=$((BAR_LENGTH - FILLED))
    
    BAR="["
    for ((i=0; i<FILLED; i++)); do BAR+="█"; done
    for ((i=0; i<EMPTY; i++)); do BAR+="░"; done
    BAR+="]"
    
    echo -e "${CYAN}Download Progress:${NC}"
    echo -e "${GREEN}${BAR}${NC} ${PERCENT}%"
    echo ""
    echo -e "${YELLOW}Size:${NC} ${SIZE_GB}GB ${SIZE_MB}MB / ${TARGET_GB}GB ${TARGET_MB}MB"
    echo ""
    
    # Calculate speed and ETA
    if [ -f /tmp/download_last_size ]; then
        LAST_SIZE=$(cat /tmp/download_last_size)
        SPEED=$((CURRENT_SIZE - LAST_SIZE))
        SPEED_MB=$((SPEED / 1048576))
        
        if [ "$SPEED" -gt 0 ]; then
            REMAINING=$((TARGET_SIZE - CURRENT_SIZE))
            ETA_SECONDS=$((REMAINING / SPEED))
            ETA_MINS=$((ETA_SECONDS / 60))
            ETA_SECS=$((ETA_SECONDS % 60))
            
            echo -e "${YELLOW}Speed:${NC} ${SPEED_MB} MB/s"
            echo -e "${YELLOW}ETA:${NC} ${ETA_MINS}m ${ETA_SECS}s"
            echo ""
        fi
    fi
    
    echo $CURRENT_SIZE > /tmp/download_last_size
    
    # Check if download is complete
    if [ "$PERCENT" -ge 100 ]; then
        echo -e "${GREEN}✅ Download Complete!${NC}"
        echo ""
        echo -e "${BLUE}════════════════════════════════════════════════════════════════${NC}"
        echo -e "${CYAN}🚀 Next Steps:${NC}"
        echo ""
        echo -e "${GREEN}1. Extract the dataset:${NC}"
        echo -e "   ${BLUE}bash /home/adelechinda/home/projects/HierarchicalVLM/SETUP_DATASET_SCRATCH.sh${NC}"
        echo ""
        echo -e "${BLUE}════════════════════════════════════════════════════════════════${NC}"
        exit 0
    fi
    
    sleep 5
done
