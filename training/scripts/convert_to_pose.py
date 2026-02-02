#!/usr/bin/env python3
"""
Convert Segmentation labels to YOLO-Pose format

Segmentation format (4 points):
    class x1 y1 x2 y2 x3 y3 x4 y4

YOLO-Pose format:
    class x_center y_center width height px1 py1 v1 px2 py2 v2 px3 py3 v3 px4 py4 v4
    
    where:
    - (x_center, y_center, width, height) = bounding box
    - (px, py) = keypoint coordinates (normalized)
    - v = visibility (2=visible, 1=occluded, 0=not labeled)
"""

import os
import shutil
from pathlib import Path


def convert_seg_to_pose(seg_label: str) -> str:
    """
    セグメンテーションラベルをPose形式に変換
    
    Args:
        seg_label: "class x1 y1 x2 y2 x3 y3 x4 y4"
    
    Returns:
        "class cx cy w h px1 py1 v1 px2 py2 v2 px3 py3 v3 px4 py4 v4"
    """
    parts = seg_label.strip().split()
    if len(parts) < 9:
        return None
    
    cls = parts[0]
    coords = list(map(float, parts[1:9]))
    
    # 4点を取得 (x1,y1), (x2,y2), (x3,y3), (x4,y4)
    points = [(coords[i], coords[i+1]) for i in range(0, 8, 2)]
    
    # Bounding boxを計算
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    
    cx = (x_min + x_max) / 2
    cy = (y_min + y_max) / 2
    w = x_max - x_min
    h = y_max - y_min
    
    # Keypointsを順序付け（tl, tr, br, bl）
    # 元のラベルが既に順序付けされていると仮定
    # visibility = 2 (visible)
    keypoints = []
    for px, py in points:
        keypoints.extend([px, py, 2])
    
    # 出力フォーマット
    result = f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"
    for kp in keypoints:
        if isinstance(kp, float):
            result += f" {kp:.6f}"
        else:
            result += f" {kp}"
    
    return result


def convert_dataset(
    seg_dir: Path,
    pose_dir: Path,
):
    """
    データセット全体を変換
    
    Args:
        seg_dir: セグメンテーションデータのディレクトリ (train or val)
        pose_dir: Poseデータの出力ディレクトリ
    """
    seg_images = seg_dir / "images"
    seg_labels = seg_dir / "labels"
    
    pose_images = pose_dir / "images"
    pose_labels = pose_dir / "labels"
    
    pose_images.mkdir(parents=True, exist_ok=True)
    pose_labels.mkdir(parents=True, exist_ok=True)
    
    # 画像ファイルを処理
    image_extensions = {'.jpg', '.jpeg', '.png'}
    converted = 0
    skipped = 0
    
    for img_file in seg_images.iterdir():
        if img_file.suffix.lower() not in image_extensions:
            continue
        
        # 対応するラベルファイル
        label_file = seg_labels / (img_file.stem + '.txt')
        
        if not label_file.exists():
            skipped += 1
            continue
        
        # ラベルを変換
        with open(label_file, 'r') as f:
            lines = f.readlines()
        
        pose_lines = []
        for line in lines:
            pose_line = convert_seg_to_pose(line)
            if pose_line:
                pose_lines.append(pose_line)
        
        if not pose_lines:
            skipped += 1
            continue
        
        # 画像をコピー
        shutil.copy(img_file, pose_images / img_file.name)
        
        # 変換したラベルを保存
        out_label = pose_labels / (img_file.stem + '.txt')
        with open(out_label, 'w') as f:
            f.write('\n'.join(pose_lines))
        
        converted += 1
    
    print(f"  Converted: {converted}, Skipped: {skipped}")


def main():
    project_root = Path(__file__).parent.parent
    
    seg_data = project_root / "data" / "processed"
    pose_data = project_root / "data" / "pose"
    
    print("Converting Segmentation labels to YOLO-Pose format...")
    print()
    
    # Train
    print("Processing train...")
    convert_dataset(seg_data / "train", pose_data / "train")
    
    # Val
    print("Processing val...")
    convert_dataset(seg_data / "val", pose_data / "val")
    
    # data.yaml作成
    data_yaml = pose_data / "data.yaml"
    yaml_content = f"""# YOLO-Pose Dataset for License Plate Corner Detection
path: {pose_data}
train: train/images
val: val/images

# Keypoints
kpt_shape: [4, 3]  # 4 keypoints, each with (x, y, visibility)

# Classes
names:
  0: plate

# Keypoint names (for visualization)
# Order: top-left, top-right, bottom-right, bottom-left
"""
    
    with open(data_yaml, 'w') as f:
        f.write(yaml_content)
    
    print()
    print(f"Created: {data_yaml}")
    print()
    print("Done! Dataset ready for YOLO-Pose training.")


if __name__ == "__main__":
    main()
