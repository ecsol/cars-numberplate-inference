#!/usr/bin/env python3
"""
YOLO Segmentation Training - Precise Corner Detection
Mục tiêu: Detect chính xác 4 góc biển số cho masking

Usage (Lambda Labs GPU):
    python train_precise.py --device cuda --batch 16

Usage (Local MPS):
    python train_precise.py --device mps --batch 4
"""

from pathlib import Path
from ultralytics import YOLO
import argparse
import yaml


def train(
    model_name: str = "yolo26x-seg.pt",
    device: str = "cuda",
    batch: int = 16,
    epochs: int = 200,
    imgsz: int = 800,
):
    """
    高精度セグメンテーション学習

    Args:
        model_name: ベースモデル
        device: cuda / mps / cpu
        batch: バッチサイズ
        epochs: エポック数
        imgsz: 画像サイズ
    """
    # パスの設定
    project_root = Path(__file__).parent.parent
    data_yaml = project_root / "configs" / "data.yaml"

    # data.yaml確認・作成
    if not data_yaml.exists():
        data_config = {
            "path": str(project_root / "data" / "processed"),
            "train": "train/images",
            "val": "val/images",
            "names": {0: "plate"},
        }
        with open(data_yaml, "w") as f:
            yaml.dump(data_config, f)
        print(f"Created: {data_yaml}")

    # モデル読み込み
    model = YOLO(model_name)

    print("=" * 60)
    print("🎯 PRECISE CORNER DETECTION TRAINING")
    print("=" * 60)
    print(f"Model: {model_name}")
    print(f"Device: {device}")
    print(f"Batch: {batch}")
    print(f"Image Size: {imgsz}")
    print(f"Epochs: {epochs}")
    print("=" * 60)

    # 学習実行 - 高精度設定
    results = model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        # === 高精度のための設定 ===
        # Loss weights - boundary重視
        box=10.0,  # Box loss高め（角の精度）
        cls=0.3,  # Classification低め（1クラスのみ）
        dfl=2.0,  # Distribution focal loss
        # Segmentation設定
        mask_ratio=1,  # フル解像度マスク！
        overlap_mask=True,
        retina_masks=True,  # 高解像度出力
        # Training設定
        optimizer="AdamW",
        lr0=0.0005,  # 低いLR = 精密な学習
        lrf=0.01,
        weight_decay=0.001,
        warmup_epochs=10,
        patience=40,
        cos_lr=True,  # Cosine LR
        # Augmentation - 控えめ（境界を保つ）
        degrees=10.0,  # 回転 ±10°
        translate=0.15,
        scale=0.3,  # スケール控えめ
        shear=5.0,
        perspective=0.0005,  # 透視変換は最小限
        flipud=0.0,
        fliplr=0.5,
        hsv_h=0.02,
        hsv_s=0.7,
        hsv_v=0.5,
        mosaic=0.8,  # Mosaic控えめ
        mixup=0.0,  # MixUp OFF（境界がぼける）
        copy_paste=0.2,
        erasing=0.2,
        close_mosaic=20,  # 最後20エポックはmosaic OFF
        # Other
        single_cls=True,  # 単一クラス
        cache=True,
        plots=True,
        project=str(project_root / "runs" / "precise"),
        name="plate_precise",
        exist_ok=True,
    )

    print("\n" + "=" * 60)
    print("✅ TRAINING COMPLETED!")
    print("=" * 60)

    # 結果表示
    if hasattr(results, "results_dict"):
        metrics = results.results_dict
        print(f"Box mAP50: {metrics.get('metrics/mAP50(B)', 'N/A'):.4f}")
        print(f"Mask mAP50: {metrics.get('metrics/mAP50(M)', 'N/A'):.4f}")
        print(f"Mask mAP50-95: {metrics.get('metrics/mAP50-95(M)', 'N/A'):.4f}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train YOLO for precise plate segmentation"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolo11x-seg.pt",
        help="Base model (yolo11x-seg.pt or yolo26x-seg.pt)",
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="Device: cuda / mps / cpu"
    )
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--epochs", type=int, default=200, help="Number of epochs")
    parser.add_argument("--imgsz", type=int, default=800, help="Image size")

    args = parser.parse_args()

    train(
        model_name=args.model,
        device=args.device,
        batch=args.batch,
        epochs=args.epochs,
        imgsz=args.imgsz,
    )
