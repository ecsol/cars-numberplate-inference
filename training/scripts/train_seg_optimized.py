#!/usr/bin/env python3
"""
YOLO26x-Seg Training - Optimized for Precise Plate Segmentation

Stage 1 của Two-Stage Pipeline:
- Detect vùng biển số với polygon mask
- Sử dụng model mạnh nhất: YOLO26x-seg

Usage (Lambda Labs GPU):
    python train_seg_optimized.py --device cuda --batch 16
"""

from pathlib import Path
from ultralytics import YOLO
import argparse


def train(
    model_name: str = "yolo26x-seg.pt",
    device: str = "cuda",
    batch: int = 16,
    epochs: int = 200,
    imgsz: int = 800,  # Tăng resolution cho segmentation chính xác
):
    """
    YOLO26x-Seg Training tối ưu

    Args:
        model_name: Base model (yolo26x-seg.pt = mạnh nhất)
        device: cuda / mps / cpu
        batch: Batch size
        epochs: Số epochs
        imgsz: Image size (800 cho segmentation chính xác)
    """
    project_root = Path(__file__).parent.parent
    data_yaml = project_root / "data" / "processed" / "data.yaml"

    # Tạo data.yaml nếu chưa có
    if not data_yaml.exists():
        data_yaml = project_root / "configs" / "data.yaml"

    model = YOLO(model_name)

    print("=" * 60)
    print("🎯 YOLO26x-SEG: OPTIMIZED SEGMENTATION")
    print("=" * 60)
    print(f"Model: {model_name}")
    print(f"Device: {device}")
    print(f"Batch: {batch}")
    print(f"Image Size: {imgsz}")
    print(f"Epochs: {epochs}")
    print(f"Data: {data_yaml}")
    print("=" * 60)

    results = model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        # ============ SEGMENTATION - HIGH PRECISION ============
        mask_ratio=1,  # QUAN TRỌNG: Full resolution mask!
        overlap_mask=True,
        retina_masks=True,  # High-res mask output
        # ============ LOSS WEIGHTS - BOUNDARY FOCUS ============
        box=10.0,  # Tăng box loss (boundary chính xác)
        cls=0.3,  # Giảm (chỉ 1 class)
        dfl=2.0,  # Distribution focal loss
        # ============ TRAINING ============
        optimizer="AdamW",
        lr0=0.0005,  # LR thấp cho precision
        lrf=0.01,
        weight_decay=0.001,
        warmup_epochs=10,
        patience=50,
        cos_lr=True,
        # ============ AUGMENTATION - BALANCED ============
        # Geometric (vừa phải để giữ boundary)
        degrees=10.0,  # Rotation ±10°
        translate=0.15,
        scale=0.3,  # Scale vừa phải
        shear=5.0,
        perspective=0.0005,  # Perspective nhẹ
        # Flip
        flipud=0.0,  # KHÔNG flip dọc
        fliplr=0.5,
        # Color
        hsv_h=0.02,
        hsv_s=0.7,
        hsv_v=0.5,
        # Advanced
        mosaic=0.8,  # Giảm mosaic để giữ boundary
        mixup=0.0,  # TẮT - làm mờ boundary
        copy_paste=0.2,  # Copy-paste cho segmentation
        erasing=0.2,
        close_mosaic=20,  # Tắt mosaic 20 epochs cuối
        # ============ OTHER ============
        single_cls=True,  # Chỉ 1 class
        cache=True,
        plots=True,
        project=str(project_root / "runs" / "seg_optimized"),
        name="plate_seg_v2",
        exist_ok=True,
    )

    print("\n" + "=" * 60)
    print("✅ SEGMENTATION TRAINING COMPLETED!")
    print("=" * 60)

    # Best model path
    best_model = (
        project_root / "runs" / "seg_optimized" / "plate_seg_v2" / "weights" / "best.pt"
    )
    if best_model.exists():
        print(f"Best model: {best_model}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLO26x-Seg optimized")
    parser.add_argument(
        "--model",
        type=str,
        default="yolo26x-seg.pt",
        help="Base model (yolo26x-seg.pt = strongest)",
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="Device: cuda / mps / cpu"
    )
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--epochs", type=int, default=200, help="Number of epochs")
    parser.add_argument(
        "--imgsz",
        type=int,
        default=800,
        help="Image size (800 for precise segmentation)",
    )

    args = parser.parse_args()

    train(
        model_name=args.model,
        device=args.device,
        batch=args.batch,
        epochs=args.epochs,
        imgsz=args.imgsz,
    )
