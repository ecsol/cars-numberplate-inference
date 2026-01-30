#!/bin/bash
# 毎分実行: 今日の車両画像を取得してナンバープレートをマスク
# crontab -e: * * * * * /home/ec2-user/plate-detection-service/scripts/fetch_and_process.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SERVICE_DIR="$(dirname "$SCRIPT_DIR")"
VENV="$SERVICE_DIR/venv/bin"
LOG_FILE="/var/log/plate-detection/fetch.log"

# 環境変数
export DB_HOST="cs1adb99-instance-1.cdr2jtnoao7j.ap-northeast-1.rds.amazonaws.com"
export DB_NAME="cartrading"
export DB_USER="cars_hitosuke"
export DB_PASSWORD=""  # 環境変数で設定推奨
export S3_BUCKET="your-bucket-name"
export OUTPUT_DIR="/mnt/s3bucket/processed"

mkdir -p "$(dirname "$LOG_FILE")"

echo "[$(date)] Start" >> "$LOG_FILE"
$VENV/python "$SCRIPT_DIR/fetch_today_images.py" >> "$LOG_FILE" 2>&1
echo "[$(date)] Done" >> "$LOG_FILE"
