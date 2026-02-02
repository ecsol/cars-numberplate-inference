#!/bin/bash
# ============================================================
# YOLO26x-Pose Training on Lambda Labs
# Model mạnh nhất cho 4-corner detection
# ============================================================

# Config
LAMBDA_IP="$1"
if [ -z "$LAMBDA_IP" ]; then
    echo "Usage: ./lambda_train_pose.sh <LAMBDA_IP>"
    echo "Example: ./lambda_train_pose.sh 192.168.1.100"
    exit 1
fi

LAMBDA_USER="ubuntu"
REMOTE_DIR="/home/ubuntu/plate_pose"

echo "============================================================"
echo "🚀 YOLO26x-Pose Training Setup"
echo "============================================================"
echo "Lambda IP: $LAMBDA_IP"
echo "Remote Dir: $REMOTE_DIR"
echo ""

# 1. Create remote directory
echo "📁 Creating remote directory..."
ssh ${LAMBDA_USER}@${LAMBDA_IP} "mkdir -p ${REMOTE_DIR}"

# 2. Upload pose data
echo "📤 Uploading pose data..."
scp -r data/pose ${LAMBDA_USER}@${LAMBDA_IP}:${REMOTE_DIR}/

# 3. Upload training script
echo "📤 Uploading training script..."
scp scripts/train_pose.py ${LAMBDA_USER}@${LAMBDA_IP}:${REMOTE_DIR}/

# 4. Install dependencies and start training
echo "🔧 Installing dependencies and starting training..."
ssh ${LAMBDA_USER}@${LAMBDA_IP} << 'ENDSSH'
cd /home/ubuntu/plate_pose

# Install ultralytics
pip install -q ultralytics

# Start training with nohup
echo "Starting YOLO26x-Pose training..."
nohup python train_pose.py \
    --model yolo26x-pose.pt \
    --device cuda \
    --batch 16 \
    --epochs 200 \
    --imgsz 640 \
    > train.log 2>&1 &

echo "Training started! PID: $!"
echo "Monitor with: tail -f train.log"
ENDSSH

echo ""
echo "============================================================"
echo "✅ Training started on Lambda Labs!"
echo "============================================================"
echo ""
echo "To monitor:"
echo "  ssh ${LAMBDA_USER}@${LAMBDA_IP} 'tail -f ${REMOTE_DIR}/train.log'"
echo ""
echo "To download model after training:"
echo "  scp ${LAMBDA_USER}@${LAMBDA_IP}:${REMOTE_DIR}/runs/pose/plate_corners/weights/best.pt ./models/pose_best.pt"
