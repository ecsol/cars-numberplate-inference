#!/bin/bash
# オフラインインストール用のwheelファイルをダウンロード
# ローカルPCで実行してください

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WHEELS_DIR="$SCRIPT_DIR/wheels"

mkdir -p "$WHEELS_DIR"

echo "=== PyTorch CPU版ダウンロード ==="
pip download -d "$WHEELS_DIR" torch torchvision --index-url https://download.pytorch.org/whl/cpu

echo "=== その他の依存関係ダウンロード ==="
pip download -d "$WHEELS_DIR" -r "$SCRIPT_DIR/requirements.txt"

echo "=== 完了 ==="
echo "wheels/ フォルダをサーバーにコピーしてください"
echo "scp -r $SCRIPT_DIR user@server:~/plate-detection-service/"
