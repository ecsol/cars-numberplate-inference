#!/bin/bash
# ===========================================
# YOLO26 Training Script
# ===========================================

# 1. Setup
echo "=== Setting up environment ==="
pip install ultralytics --upgrade --quiet

# 2. Check GPU
echo "=== GPU Info ==="
nvidia-smi

# 3. Train với YOLO26
echo "=== Starting YOLO26 Training ==="
yolo segment train \
    model=yolo26x-seg.pt \
    data=data.yaml \
    epochs=300 \
    batch=16 \
    imgsz=640 \
    device=0 \
    project=runs \
    name=plate_seg_yolo26 \
    exist_ok=True \
    patience=30 \
    save=True \
    plots=True

echo "=== Training Complete ==="
echo "Best model: runs/plate_seg_yolo26/weights/best.pt"
