#!/usr/bin/env python3
"""
YOLO-Pose Training for License Plate Corner Detection

Stage 2 của Two-Stage Pipeline:
- Input: Vùng biển số (từ YOLO-Seg)
- Output: 4 góc chính xác (keypoints)

Usage (Lambda Labs GPU):
    python train_pose.py --device cuda --batch 16

Usage (Local MPS):
    python train_pose.py --device mps --batch 4
"""

from pathlib import Path
from ultralytics import YOLO
import argparse


def train(
    model_name: str = "yolo11x-pose.pt",
    device: str = "cuda",
    batch: int = 16,
    epochs: int = 200,
    imgsz: int = 640,
):
    """
    YOLO-Pose学習（4コーナー検出）

    Args:
        model_name: ベースモデル (yolo11n-pose.pt, yolo11x-pose.pt等)
        device: cuda / mps / cpu
        batch: バッチサイズ
        epochs: エポック数
        imgsz: 画像サイズ
    """
    project_root = Path(__file__).parent.parent
    data_yaml = project_root / "data" / "pose" / "data.yaml"

    if not data_yaml.exists():
        print(f"Error: {data_yaml} not found!")
        print("Run convert_to_pose.py first.")
        return

    # モデル読み込み
    model = YOLO(model_name)

    print("=" * 60)
    print("🎯 YOLO-POSE: 4 CORNER DETECTION")
    print("=" * 60)
    print(f"Model: {model_name}")
    print(f"Device: {device}")
    print(f"Batch: {batch}")
    print(f"Image Size: {imgsz}")
    print(f"Epochs: {epochs}")
    print(f"Data: {data_yaml}")
    print("=" * 60)

    # 学習実行
    results = model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        # === Pose特有の設定 ===
        pose=12.0,  # Pose loss weight
        kobj=2.0,  # Keypoint objectness loss
        # === 高精度設定 ===
        box=7.5,  # Box loss
        cls=0.5,  # Classification loss
        dfl=1.5,  # Distribution focal loss
        # Training
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        weight_decay=0.0005,
        warmup_epochs=5,
        patience=50,
        cos_lr=True,
        # Augmentation（控えめ - コーナー精度重視）
        degrees=10.0,
        translate=0.15,
        scale=0.3,
        shear=5.0,
        perspective=0.0005,
        flipud=0.0,
        fliplr=0.5,
        hsv_h=0.02,
        hsv_s=0.7,
        hsv_v=0.5,
        mosaic=0.8,
        mixup=0.0,  # OFF for precise keypoints
        copy_paste=0.0,  # OFF for pose
        erasing=0.2,
        close_mosaic=20,
        # Other
        cache=True,
        plots=True,
        project=str(project_root / "runs" / "pose"),
        name="plate_corners",
        exist_ok=True,
    )

    print("\n" + "=" * 60)
    print("✅ POSE TRAINING COMPLETED!")
    print("=" * 60)

    # Best modelをコピー
    best_model = (
        project_root / "runs" / "pose" / "plate_corners" / "weights" / "best.pt"
    )
    if best_model.exists():
        print(f"Best model: {best_model}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train YOLO-Pose for plate corner detection"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolo26x-pose.pt",
        help="Base model (yolo26x-pose.pt = strongest)",
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="Device: cuda / mps / cpu"
    )
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--epochs", type=int, default=200, help="Number of epochs")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")

    args = parser.parse_args()

    train(
        model_name=args.model,
        device=args.device,
        batch=args.batch,
        epochs=args.epochs,
        imgsz=args.imgsz,
    )
