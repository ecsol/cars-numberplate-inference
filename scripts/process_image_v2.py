#!/usr/bin/env python3
"""
ナンバープレート検出・マスキング処理スクリプト (Two-Stage版)

Two-Stage Pipeline:
  - YOLO-Pose: 4コーナーの精密検出

Usage:
    # 単一ファイル処理
    python scripts/process_image_v2.py --input=input.jpg --output=output.jpg

    # フォルダ一括処理
    python scripts/process_image_v2.py --input=/path/to/folder --output=/path/to/output
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from PIL.ExifTags import TAGS
from ultralytics import YOLO

# サポートする画像拡張子
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# デフォルトパス
SEG_MODEL_PATH = (
    Path(__file__).parent.parent / "models" / "best_yolo26x_lambda_20260201.pt"
)
POSE_MODEL_PATH = Path(__file__).parent.parent / "models" / "yolo26x_pose_best.pt"
BANNER_PATH = Path(__file__).parent.parent / "assets" / "banner_sample.png"
PLATE_MASK_PATH = Path(__file__).parent.parent / "assets" / "plate_mask.png"


def get_exif_orientation(image_path: str) -> int:
    """EXIF Orientationを取得"""
    try:
        img = Image.open(image_path)
        exif = img._getexif()
        if exif:
            for tag_id, value in exif.items():
                tag = TAGS.get(tag_id, tag_id)
                if tag == "Orientation":
                    return value
    except Exception:
        pass
    return 1


def apply_exif_orientation(image: np.ndarray, orientation: int) -> np.ndarray:
    """EXIF Orientationに基づいて画像を回転"""
    if orientation == 1:
        return image
    elif orientation == 2:
        return cv2.flip(image, 1)
    elif orientation == 3:
        return cv2.rotate(image, cv2.ROTATE_180)
    elif orientation == 4:
        return cv2.flip(image, 0)
    elif orientation == 5:
        return cv2.flip(cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE), 1)
    elif orientation == 6:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    elif orientation == 7:
        return cv2.flip(cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE), 1)
    elif orientation == 8:
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return image


def draw_antialiased_polygon(shape: tuple, corners: np.ndarray) -> np.ndarray:
    """8xスーパーサンプリングでアンチエイリアスポリゴンを描画"""
    h, w = shape[:2]
    scale = 8
    large = np.zeros((h * scale, w * scale), dtype=np.uint8)
    cv2.fillPoly(large, [(corners * scale).astype(np.int32)], 255)
    result = cv2.resize(large, (w, h), interpolation=cv2.INTER_AREA)
    return result.astype(np.float32) / 255.0


def apply_mask_with_shadow(
    image: np.ndarray,
    corners: np.ndarray,
    mask_img: np.ndarray,
    shadow_offset: tuple = (8, 12),
    shadow_strength: float = 0.35,
) -> np.ndarray:
    """4コーナーにマスクを適用（シャドウ付き）"""
    h, w = image.shape[:2]
    mh, mw = mask_img.shape[:2]
    result = image.copy().astype(np.float32)

    # シャドウ
    shadow_corners = corners + np.array(shadow_offset)
    shadow_alpha = draw_antialiased_polygon((h, w), shadow_corners)
    shadow_alpha = cv2.GaussianBlur(shadow_alpha, (41, 41), 0) * shadow_strength
    for c in range(3):
        result[:, :, c] *= 1 - shadow_alpha * 0.5

    # マスク画像をワープ
    src_pts = np.array(
        [[0, 0], [mw - 1, 0], [mw - 1, mh - 1], [0, mh - 1]], dtype=np.float32
    )
    dst_pts = corners.astype(np.float32)
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv2.warpPerspective(mask_img, M, (w, h), flags=cv2.INTER_LANCZOS4)

    # アンチエイリアスアルファ
    alpha = draw_antialiased_polygon((h, w), corners)
    alpha = cv2.GaussianBlur(alpha, (3, 3), 0)

    # 明度調整
    mask_binary = alpha > 0.5
    kernel = np.ones((40, 40), np.uint8)
    dilated = cv2.dilate(mask_binary.astype(np.uint8), kernel)
    surround = (dilated > 0) & ~mask_binary

    warped_float = warped.astype(np.float32)
    if surround.sum() > 50:
        env_brightness = np.mean(image[surround])
        factor = 0.85 if env_brightness > 140 else 0.92
        warped_float *= factor

    # 合成
    for c in range(3):
        result[:, :, c] = alpha * warped_float[:, :, c] + (1 - alpha) * result[:, :, c]

    return np.clip(result, 0, 255).astype(np.uint8)


def order_corners(corners: np.ndarray) -> np.ndarray:
    """コーナーを順序付け: top-left, top-right, bottom-right, bottom-left"""
    corners = corners.reshape(4, 2)
    s = corners.sum(axis=1)
    tl = corners[np.argmin(s)]
    br = corners[np.argmax(s)]
    d = np.diff(corners, axis=1).flatten()
    tr = corners[np.argmin(d)]
    bl = corners[np.argmax(d)]
    return np.array([tl, tr, br, bl], dtype=np.float32)


def calc_iou(box1, box2):
    """2つのbboxのIoUを計算"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0


def nms_corners(detections: list, iou_thresh: float = 0.3) -> list:
    """コーナーベースのNMS（stricter threshold）"""
    if not detections:
        return []

    detections = sorted(detections, key=lambda x: x["confidence"], reverse=True)
    keep = []

    for det in detections:
        corners = det["corners"]
        box = [
            corners[:, 0].min(),
            corners[:, 1].min(),
            corners[:, 0].max(),
            corners[:, 1].max(),
        ]

        should_keep = True
        for kept in keep:
            kept_corners = kept["corners"]
            kept_box = [
                kept_corners[:, 0].min(),
                kept_corners[:, 1].min(),
                kept_corners[:, 0].max(),
                kept_corners[:, 1].max(),
            ]

            if calc_iou(box, kept_box) > iou_thresh:
                should_keep = False
                break

        if should_keep:
            keep.append(det)

    return keep


def detect_with_tiling(
    image: np.ndarray,
    model: YOLO,
    conf: float,
    tile_size: int = 1200,
    overlap: int = 300,
) -> list[dict]:
    """大きな画像をタイル分割して検出（小さいプレートも検出可能）"""
    h, w = image.shape[:2]
    detections = []

    # 画像が小さい場合はタイリング不要
    if w <= tile_size * 1.5 and h <= tile_size * 1.5:
        return []

    # タイル位置を計算
    step = tile_size - overlap
    x_positions = list(range(0, max(1, w - overlap), step))
    y_positions = list(range(0, max(1, h - overlap), step))

    for y in y_positions:
        for x in x_positions:
            x2 = min(x + tile_size, w)
            y2 = min(y + tile_size, h)
            tile = image[y:y2, x:x2]

            results = model.predict(tile, conf=conf, verbose=False)

            if results and len(results[0].boxes) > 0:
                for box, kpts in zip(results[0].boxes, results[0].keypoints):
                    if kpts.xy is None or len(kpts.xy[0]) < 4:
                        continue

                    corners = kpts.xy[0].cpu().numpy()[:4]

                    # サイズ検証
                    corner_w = corners[:, 0].max() - corners[:, 0].min()
                    corner_h = corners[:, 1].max() - corners[:, 1].min()
                    if corner_w < 30 or corner_h < 20:
                        continue

                    # 元の座標に変換
                    corners_orig = corners + np.array([x, y])

                    box_conf = float(box.conf[0])
                    kpt_conf = (
                        float(kpts.conf[0][:4].mean()) if kpts.conf is not None else 0.0
                    )

                    detections.append(
                        {
                            "corners": corners_orig,
                            "confidence": (box_conf + kpt_conf) / 2,
                            "source": "pose_tiled",
                        }
                    )

    # 重複検出を除去（厳しいNMS）
    if len(detections) > 1:
        detections = nms_corners(detections, iou_thresh=0.3)

    return detections


def detect_plates_two_stage(
    image: np.ndarray,
    seg_model: YOLO,
    pose_model: YOLO,
    seg_conf: float = 0.3,
    pose_conf: float = 0.2,
) -> list[dict]:
    """
    Two-Stage検出:
    1. Seg検出（確実な領域検出）
    2. Pose検出 + Segとマッチング（誤検出防止）
    3. Tiling（大きな画像で検出できない場合）
    """
    h, w = image.shape[:2]
    detections = []

    # Step 1: YOLO-Seg - まず領域を検出
    seg_results = seg_model.predict(image, conf=seg_conf, verbose=False)
    seg_boxes = []
    seg_corners_list = []

    if seg_results and len(seg_results[0].boxes) > 0:
        for i, box in enumerate(seg_results[0].boxes):
            seg_box = box.xyxy[0].cpu().numpy()
            seg_boxes.append(seg_box)
            seg_conf_score = float(box.conf[0])

            # Segのpolygonからcornersを取得
            seg_corners = None
            if seg_results[0].masks is not None and i < len(seg_results[0].masks):
                mask_xy = seg_results[0].masks[i].xy[0]
                if len(mask_xy) >= 4:
                    hull = cv2.convexHull(mask_xy.astype(np.float32))
                    rect = cv2.minAreaRect(hull)
                    seg_corners = cv2.boxPoints(rect)
                    seg_corners = order_corners(seg_corners)

            if seg_corners is None:
                x1, y1, x2, y2 = seg_box
                seg_corners = np.array(
                    [[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32
                )

            seg_corners_list.append(
                {
                    "corners": seg_corners,
                    "confidence": seg_conf_score,
                    "box": seg_box,
                }
            )

    # Step 2: YOLO-Pose - 精密コーナー検出（Segとマッチングして検証）
    pose_results = pose_model.predict(image, conf=pose_conf, verbose=False)
    matched_seg_indices = set()

    if pose_results and len(pose_results[0].boxes) > 0:
        for box, kpts in zip(pose_results[0].boxes, pose_results[0].keypoints):
            if kpts.xy is None or len(kpts.xy[0]) < 4:
                continue

            corners = kpts.xy[0].cpu().numpy()[:4]

            # サイズ検証
            x_min, y_min = corners.min(axis=0)
            x_max, y_max = corners.max(axis=0)
            corner_w, corner_h = x_max - x_min, y_max - y_min
            min_size = min(h, w) * 0.005

            if corner_w < min_size or corner_h < min_size:
                continue

            pose_box = [x_min, y_min, x_max, y_max]
            box_conf = float(box.conf[0])
            kpt_conf = float(kpts.conf[0][:4].mean()) if kpts.conf is not None else 0.0

            # SegとのIoUをチェック
            best_iou = 0
            best_seg_idx = -1
            for seg_idx, seg_box in enumerate(seg_boxes):
                iou = calc_iou(pose_box, seg_box)
                if iou > best_iou:
                    best_iou = iou
                    best_seg_idx = seg_idx

            # Segとマッチした場合のみPoseを使用
            if best_iou > 0.3:
                matched_seg_indices.add(best_seg_idx)
                detections.append(
                    {
                        "corners": corners,
                        "confidence": (box_conf + kpt_conf) / 2,
                        "source": "pose",
                    }
                )
            elif not seg_boxes:
                # Segがない場合はPoseをそのまま使用
                detections.append(
                    {
                        "corners": corners,
                        "confidence": (box_conf + kpt_conf) / 2,
                        "source": "pose",
                    }
                )

    # Poseでマッチしなかった Seg detections を追加
    for seg_idx, seg_det in enumerate(seg_corners_list):
        if seg_idx not in matched_seg_indices:
            detections.append(
                {
                    "corners": seg_det["corners"],
                    "confidence": seg_det["confidence"],
                    "source": "seg_fallback",
                }
            )

    # Step 3: Tiling - 大きな画像で検出できない場合のみ
    if not detections and (w > 2000 or h > 1500):
        tiled_detections = detect_with_tiling(image, pose_model, pose_conf)
        if tiled_detections:
            detections.extend(tiled_detections)

    # 最終NMS（全ソースの重複除去）
    if len(detections) > 1:
        detections = nms_corners(detections, iou_thresh=0.3)

    return detections


def add_banner_overlay(
    image: np.ndarray,
    banner_path: Path,
    alpha_channel: np.ndarray = None,
) -> tuple[np.ndarray, np.ndarray]:
    """バナーを画像下部にオーバーレイ"""
    img_h, img_w = image.shape[:2]

    banner = cv2.imread(str(banner_path), cv2.IMREAD_UNCHANGED)
    if banner is None:
        print(f"Warning: Banner not found: {banner_path}")
        return image, alpha_channel

    banner_h, banner_w = banner.shape[:2]
    scale = img_w / banner_w
    new_banner_h = int(banner_h * scale)

    max_banner_h = img_h // 4
    if new_banner_h > max_banner_h:
        scale = max_banner_h / banner_h
        new_banner_h = max_banner_h

    banner_resized = cv2.resize(
        banner, (img_w, new_banner_h), interpolation=cv2.INTER_AREA
    )

    result = image.copy()
    banner_y = img_h - new_banner_h

    if banner_resized.shape[2] == 4:
        alpha = banner_resized[:, :, 3:4].astype(np.float32) / 255.0
        banner_bgr = banner_resized[:, :, :3]
        bg = result[banner_y:img_h, :, :]
        blended = (alpha * banner_bgr + (1 - alpha) * bg).astype(np.uint8)
        result[banner_y:img_h, :, :] = blended
    else:
        result[banner_y:img_h, :, :] = banner_resized[:, :, :3]

    if alpha_channel is not None:
        alpha_channel = alpha_channel.copy()
        alpha_channel[banner_y:img_h, :] = 255

    return result, alpha_channel


def process_image(
    input_path: str,
    output_path: str,
    seg_model: YOLO,
    pose_model: YOLO,
    mask_image: np.ndarray,
    is_masking: bool = True,
    seg_conf: float = 0.3,
    pose_conf: float = 0.2,
) -> dict:
    """Two-Stageで画像を処理"""
    # EXIF Orientation
    exif_orientation = get_exif_orientation(input_path)

    # 画像読み込み
    image_raw = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
    if image_raw is None:
        raise ValueError(f"画像を読み込めません: {input_path}")

    image_raw = apply_exif_orientation(image_raw, exif_orientation)
    original_h, original_w = image_raw.shape[:2]

    # アルファチャンネル分離
    has_alpha = len(image_raw.shape) == 3 and image_raw.shape[2] == 4
    if has_alpha:
        image = image_raw[:, :, :3]
        alpha_channel = image_raw[:, :, 3]
    else:
        image = image_raw
        alpha_channel = None

    # Two-Stage検出
    detections = detect_plates_two_stage(
        image, seg_model, pose_model, seg_conf, pose_conf
    )

    # マスク適用
    result = image.copy()
    for det in detections:
        corners = det["corners"]
        result = apply_mask_with_shadow(result, corners, mask_image)

    # バナー追加
    if is_masking:
        result, alpha_channel = add_banner_overlay(result, BANNER_PATH, alpha_channel)

    # サイズ確認
    result_h, result_w = result.shape[:2]
    if result_h != original_h or result_w != original_w:
        result = cv2.resize(
            result, (original_w, original_h), interpolation=cv2.INTER_AREA
        )
        if alpha_channel is not None:
            alpha_channel = cv2.resize(
                alpha_channel, (original_w, original_h), interpolation=cv2.INTER_AREA
            )

    # アルファチャンネル復元
    output_ext = Path(output_path).suffix.lower()
    if has_alpha and output_ext == ".png":
        result = cv2.merge(
            [result[:, :, 0], result[:, :, 1], result[:, :, 2], alpha_channel]
        )

    # 保存
    if output_ext in [".jpg", ".jpeg"]:
        if len(result.shape) == 3 and result.shape[2] == 4:
            result = result[:, :, :3]
        cv2.imwrite(output_path, result, [cv2.IMWRITE_JPEG_QUALITY, 98])
    elif output_ext == ".png":
        cv2.imwrite(output_path, result, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    else:
        cv2.imwrite(output_path, result)

    return {
        "input": input_path,
        "output": output_path,
        "original_size": (original_w, original_h),
        "detections": len(detections),
        "is_masking": is_masking,
    }


def get_image_files(folder: Path) -> list[Path]:
    """フォルダ内の画像ファイルを取得"""
    images = []
    for ext in SUPPORTED_EXTENSIONS:
        images.extend(folder.glob(f"*{ext}"))
        images.extend(folder.glob(f"*{ext.upper()}"))
    return sorted(set(images))


def process_batch(
    input_folder: Path,
    output_folder: Path,
    seg_model: YOLO,
    pose_model: YOLO,
    mask_image: np.ndarray,
    is_masking: bool,
    seg_conf: float,
    pose_conf: float,
) -> dict:
    """フォルダ内の画像を一括処理"""
    images = get_image_files(input_folder)

    if not images:
        print(f"画像が見つかりません: {input_folder}")
        return {"total": 0, "success": 0, "failed": 0}

    output_folder.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Two-Stage バッチ処理開始 (Pose優先 + Tiling + Seg fallback)")
    print(f"  入力: {input_folder}")
    print(f"  出力: {output_folder}")
    print(f"  画像数: {len(images)}枚")
    print(f"  バナー: {'あり' if is_masking else 'なし'}")
    print("=" * 60)

    success = 0
    failed = 0
    total_detections = 0

    for i, img_path in enumerate(images, 1):
        output_path = output_folder / img_path.name

        try:
            result = process_image(
                input_path=str(img_path),
                output_path=str(output_path),
                seg_model=seg_model,
                pose_model=pose_model,
                mask_image=mask_image,
                is_masking=is_masking,
                seg_conf=seg_conf,
                pose_conf=pose_conf,
            )

            success += 1
            total_detections += result["detections"]
            status = f"検出: {result['detections']}"

        except Exception as e:
            failed += 1
            status = f"エラー: {e}"

        print(f"[{i}/{len(images)}] {img_path.name} - {status}")

    print("=" * 60)
    print("処理完了")
    print(f"  成功: {success}枚")
    print(f"  失敗: {failed}枚")
    print(f"  総検出数: {total_detections}")
    print("=" * 60)

    return {
        "total": len(images),
        "success": success,
        "failed": failed,
        "detections": total_detections,
    }


def main():
    parser = argparse.ArgumentParser(
        description="ナンバープレート検出・マスキング (Two-Stage: Pose + Tiling + Seg)",
    )
    parser.add_argument("--input", required=True, help="入力画像パスまたはフォルダ")
    parser.add_argument("--output", help="出力先")
    parser.add_argument(
        "--is-masking", type=str, default="true", help="バナー追加 (true/false)"
    )
    parser.add_argument(
        "--seg-model", default=str(SEG_MODEL_PATH), help="YOLO-Segモデルパス"
    )
    parser.add_argument(
        "--pose-model", default=str(POSE_MODEL_PATH), help="YOLO-Poseモデルパス"
    )
    parser.add_argument("--seg-conf", type=float, default=0.3, help="Seg検出信頼度")
    parser.add_argument("--pose-conf", type=float, default=0.2, help="Pose検出信頼度")

    args = parser.parse_args()

    is_masking = args.is_masking.lower() in ("true", "1", "yes")
    input_path = Path(args.input)

    # 出力先設定
    if args.output:
        output_path = Path(args.output)
    elif input_path.is_dir():
        output_path = input_path / "output"
    else:
        output_path = input_path.parent / f"{input_path.stem}_masked{input_path.suffix}"

    # モデルロード
    print(f"Segモデル: {args.seg_model}")
    print(f"Poseモデル: {args.pose_model}")
    seg_model = YOLO(args.seg_model)
    pose_model = YOLO(args.pose_model)

    # マスク画像ロード
    mask_image = cv2.imread(str(PLATE_MASK_PATH))
    if mask_image is None:
        print(f"Error: マスク画像が見つかりません: {PLATE_MASK_PATH}")
        sys.exit(1)

    # 処理
    if input_path.is_dir():
        result = process_batch(
            input_folder=input_path,
            output_folder=output_path,
            seg_model=seg_model,
            pose_model=pose_model,
            mask_image=mask_image,
            is_masking=is_masking,
            seg_conf=args.seg_conf,
            pose_conf=args.pose_conf,
        )
        if result["failed"] > 0:
            sys.exit(1)
    else:
        if output_path.is_dir() or (
            not output_path.suffix and not output_path.exists()
        ):
            output_path.mkdir(parents=True, exist_ok=True)
            output_path = output_path / input_path.name
        else:
            output_path.parent.mkdir(parents=True, exist_ok=True)

        result = process_image(
            input_path=str(input_path),
            output_path=str(output_path),
            seg_model=seg_model,
            pose_model=pose_model,
            mask_image=mask_image,
            is_masking=is_masking,
            seg_conf=args.seg_conf,
            pose_conf=args.pose_conf,
        )

        print("処理完了:")
        print(f"  入力: {result['input']}")
        print(f"  出力: {result['output']}")
        print(f"  サイズ: {result['original_size'][0]}x{result['original_size'][1]}")
        print(f"  検出数: {result['detections']}")


if __name__ == "__main__":
    main()
