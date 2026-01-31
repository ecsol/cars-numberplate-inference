#!/usr/bin/env python3
"""
ナンバープレート検出・マスキング処理スクリプト

Usage:
    # 単一ファイル処理
    python scripts/process_image.py --input=input.jpg --output=output.jpg --is-masking=true
    
    # フォルダ一括処理（outputフォルダは自動作成）
    python scripts/process_image.py --input=/path/to/folder --output=/path/to/output
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from PIL.ExifTags import TAGS

# サポートする画像拡張子
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def get_exif_orientation(image_path: str) -> int:
    """
    画像のEXIF Orientationを取得
    
    Args:
        image_path: 画像パス
    
    Returns:
        Orientation値 (1-8)、取得できない場合は1
    """
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
    return 1  # デフォルト: 回転なし


def apply_exif_orientation(image: np.ndarray, orientation: int) -> np.ndarray:
    """
    EXIF Orientationに基づいて画像を回転
    
    Args:
        image: OpenCV画像 (BGR/BGRA)
        orientation: EXIF Orientation値 (1-8)
    
    Returns:
        回転後の画像
    """
    if orientation == 1:
        return image  # 回転なし
    elif orientation == 2:
        return cv2.flip(image, 1)  # 水平反転
    elif orientation == 3:
        return cv2.rotate(image, cv2.ROTATE_180)  # 180度回転
    elif orientation == 4:
        return cv2.flip(image, 0)  # 垂直反転
    elif orientation == 5:
        return cv2.flip(cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE), 1)
    elif orientation == 6:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)  # 90度時計回り
    elif orientation == 7:
        return cv2.flip(cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE), 1)
    elif orientation == 8:
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)  # 90度反時計回り
    return image

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.plate_detection.modeling.predict import PlateDetector
from src.plate_detection.processing.plate_masker import mask_plate_with_image, load_mask_image


# 定数
ASPECT_RATIO_MIN = 1.0
ASPECT_RATIO_MAX = 2.5
BANNER_PATH = Path(__file__).parent.parent / "assets" / "banner_sample.png"
PLATE_MASK_PATH = Path(__file__).parent.parent / "assets" / "plate_mask.png"


def validate_aspect_ratio(polygon: np.ndarray) -> bool:
    """アスペクト比を検証"""
    x, y, w, h = cv2.boundingRect(polygon)
    if h == 0 or w == 0:
        return False
    ratio = max(w, h) / min(w, h)
    return ASPECT_RATIO_MIN <= ratio <= ASPECT_RATIO_MAX


def polygon_to_quad(polygon: np.ndarray) -> np.ndarray:
    """ポリゴンを四角形に変換"""
    hull = cv2.convexHull(polygon)
    epsilon = 0.02 * cv2.arcLength(hull, True)
    approx = cv2.approxPolyDP(hull, epsilon, True)
    
    if len(approx) != 4:
        pts = hull.reshape(-1, 2)
        sum_pts = pts.sum(axis=1)
        diff_pts = np.diff(pts, axis=1).reshape(-1)
        
        tl = pts[np.argmin(sum_pts)]
        br = pts[np.argmax(sum_pts)]
        tr = pts[np.argmin(diff_pts)]
        bl = pts[np.argmax(diff_pts)]
        
        approx = np.array([tl, tr, br, bl], dtype=np.int32).reshape(-1, 1, 2)
    
    return approx


def add_banner_overlay(
    image: np.ndarray,
    banner_path: Path,
    alpha_channel: np.ndarray = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    バナーを画像下部にオーバーレイ（画像サイズ変更なし）
    
    元画像の上にバナーを直接重ねる。画像サイズは維持。
    
    Args:
        image: BGR画像
        banner_path: バナー画像パス
        alpha_channel: 元画像のアルファチャンネル（あれば）
    
    Returns:
        (結果画像, 更新されたアルファチャンネル)
    """
    img_h, img_w = image.shape[:2]
    
    # バナー読み込み
    banner = cv2.imread(str(banner_path), cv2.IMREAD_UNCHANGED)
    if banner is None:
        print(f"Warning: Banner not found: {banner_path}")
        return image, alpha_channel
    
    # バナーを画像幅にリサイズ
    banner_h, banner_w = banner.shape[:2]
    scale = img_w / banner_w
    new_banner_h = int(banner_h * scale)
    
    # バナーが画像の1/4より大きい場合は縮小
    max_banner_h = img_h // 4
    if new_banner_h > max_banner_h:
        scale = max_banner_h / banner_h
        new_banner_h = max_banner_h
    
    banner_resized = cv2.resize(banner, (img_w, new_banner_h), interpolation=cv2.INTER_AREA)
    
    # 結果画像（元画像をコピー）
    result = image.copy()
    
    # バナーを下部にオーバーレイ
    banner_y = img_h - new_banner_h
    
    # バナーがBGRAの場合、アルファブレンド
    if banner_resized.shape[2] == 4:
        alpha = banner_resized[:, :, 3:4].astype(np.float32) / 255.0
        banner_bgr = banner_resized[:, :, :3]
        bg = result[banner_y:img_h, :, :]
        blended = (alpha * banner_bgr + (1 - alpha) * bg).astype(np.uint8)
        result[banner_y:img_h, :, :] = blended
    else:
        result[banner_y:img_h, :, :] = banner_resized[:, :, :3]
    
    # アルファチャンネルがある場合、バナー領域を不透明に
    if alpha_channel is not None:
        alpha_channel = alpha_channel.copy()
        alpha_channel[banner_y:img_h, :] = 255  # バナー部分は完全不透明
    
    return result, alpha_channel


def process_image(
    input_path: str,
    output_path: str,
    is_masking: bool = True,
    model_path: str = "models/best.pt",
    confidence: float = 0.1,
) -> dict:
    """
    画像を処理
    
    Args:
        input_path: 入力画像パス
        output_path: 出力画像パス
        is_masking: マスキング後にバナーを追加するか
        model_path: モデルファイルパス
        confidence: 検出信頼度閾値
    
    Returns:
        処理結果情報
    """
    # EXIF Orientation を取得（OpenCVで読む前に）
    exif_orientation = get_exif_orientation(input_path)
    
    # 画像読み込み（UNCHANGED でアルファチャンネルを保持）
    image_raw = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
    if image_raw is None:
        raise ValueError(f"画像を読み込めません: {input_path}")
    
    # EXIF Orientation を適用（正しい向きに回転）
    image_raw = apply_exif_orientation(image_raw, exif_orientation)
    
    original_h, original_w = image_raw.shape[:2]
    
    # アルファチャンネルを分離（存在する場合）
    has_alpha = len(image_raw.shape) == 3 and image_raw.shape[2] == 4
    if has_alpha:
        # BGRA → BGR + Alpha
        image = image_raw[:, :, :3]
        alpha_channel = image_raw[:, :, 3]
    else:
        image = image_raw
        alpha_channel = None
    
    # モデルロード
    detector = PlateDetector(
        model_path=Path(model_path),
        confidence=confidence,
        device="cpu",
    )
    
    # 検出
    detections = detector.predict(image)
    
    # 有効な検出をフィルタリング
    valid_detections = []
    for det in detections:
        polygon = det["mask"]
        if validate_aspect_ratio(polygon):
            valid_detections.append(det)
    
    # マスク画像をロード
    mask_image = load_mask_image(PLATE_MASK_PATH)
    
    # マスク処理（plate_mask.png を射影変換して合成）
    result = image.copy()
    for det in valid_detections:
        polygon = det["mask"]
        result = mask_plate_with_image(result, polygon, mask_image)
    
    # バナー追加（is_masking=trueの場合）
    if is_masking:
        result, alpha_channel = add_banner_overlay(result, BANNER_PATH, alpha_channel)
    
    # サイズ確認（元サイズを維持）
    result_h, result_w = result.shape[:2]
    if result_h != original_h or result_w != original_w:
        result = cv2.resize(result, (original_w, original_h), interpolation=cv2.INTER_AREA)
        # アルファチャンネルもリサイズ
        if alpha_channel is not None:
            alpha_channel = cv2.resize(alpha_channel, (original_w, original_h), interpolation=cv2.INTER_AREA)
    
    # アルファチャンネルを復元（PNG出力の場合）
    output_ext = Path(output_path).suffix.lower()
    if has_alpha and output_ext == ".png":
        # BGR + Alpha → BGRA
        result = cv2.merge([result[:, :, 0], result[:, :, 1], result[:, :, 2], alpha_channel])
    
    # 保存（高品質）
    # JPEG: quality=98 でノイズを最小化
    # PNG: 圧縮レベル=3（デフォルト）、アルファチャンネル保持
    if output_ext in [".jpg", ".jpeg"]:
        # JPEGはアルファ非対応なので、アルファがある場合はBGRに変換
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
        "detections": len(valid_detections),
        "is_masking": is_masking,
    }


def get_image_files(folder: Path) -> list[Path]:
    """
    フォルダ内の画像ファイルを取得
    
    Args:
        folder: 検索対象フォルダ
    
    Returns:
        画像ファイルパスのリスト
    """
    images = []
    for ext in SUPPORTED_EXTENSIONS:
        images.extend(folder.glob(f"*{ext}"))
        images.extend(folder.glob(f"*{ext.upper()}"))
    return sorted(set(images))


def process_batch(
    input_folder: Path,
    output_folder: Path,
    is_masking: bool,
    model_path: str,
    confidence: float,
) -> dict:
    """
    フォルダ内の画像を一括処理
    
    Args:
        input_folder: 入力フォルダ
        output_folder: 出力フォルダ
        is_masking: バナー追加フラグ
        model_path: モデルパス
        confidence: 検出信頼度
    
    Returns:
        処理結果サマリー
    """
    # 画像ファイル取得
    images = get_image_files(input_folder)
    
    if not images:
        print(f"画像が見つかりません: {input_folder}")
        return {"total": 0, "success": 0, "failed": 0}
    
    # 出力フォルダ作成
    output_folder.mkdir(parents=True, exist_ok=True)
    
    print(f"=" * 60)
    print(f"バッチ処理開始")
    print(f"  入力: {input_folder}")
    print(f"  出力: {output_folder}")
    print(f"  画像数: {len(images)}枚")
    print(f"  バナー: {'あり' if is_masking else 'なし'}")
    print(f"=" * 60)
    
    success = 0
    failed = 0
    total_detections = 0
    
    for i, img_path in enumerate(images, 1):
        output_path = output_folder / img_path.name
        
        try:
            result = process_image(
                input_path=str(img_path),
                output_path=str(output_path),
                is_masking=is_masking,
                model_path=model_path,
                confidence=confidence,
            )
            
            success += 1
            total_detections += result["detections"]
            status = f"検出: {result['detections']}"
            
        except Exception as e:
            failed += 1
            status = f"エラー: {e}"
        
        # 進捗表示
        print(f"[{i}/{len(images)}] {img_path.name} - {status}")
    
    # サマリー
    print(f"=" * 60)
    print(f"処理完了")
    print(f"  成功: {success}枚")
    print(f"  失敗: {failed}枚")
    print(f"  総検出数: {total_detections}")
    print(f"=" * 60)
    
    return {
        "total": len(images),
        "success": success,
        "failed": failed,
        "detections": total_detections,
    }


def main():
    parser = argparse.ArgumentParser(
        description="ナンバープレート検出・マスキング",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  # 単一ファイル処理
  python scripts/process_image.py --input=input.jpg --output=output.jpg
  
  # フォルダ一括処理（outputは自動で input/output に作成）
  python scripts/process_image.py --input=/path/to/images
  
  # 出力先を指定
  python scripts/process_image.py --input=/path/to/images --output=/path/to/output
  
  # バナーなし（マスキングのみ）
  python scripts/process_image.py --input=folder --is-masking=false
        """
    )
    parser.add_argument("--input", required=True, help="入力画像パスまたはフォルダ")
    parser.add_argument("--output", help="出力先（省略時: フォルダ→input/output, ファイル→input_masked.jpg）")
    parser.add_argument("--is-masking", type=str, default="true", help="バナー追加 (true/false)")
    parser.add_argument("--model", default="models/best.pt", help="モデルパス")
    parser.add_argument("--confidence", type=float, default=0.1, help="検出信頼度")
    
    args = parser.parse_args()
    
    # is-masking を bool に変換
    is_masking = args.is_masking.lower() in ("true", "1", "yes")
    
    input_path = Path(args.input)
    
    # 出力先の自動設定
    if args.output:
        output_path = Path(args.output)
    elif input_path.is_dir():
        # フォルダの場合: input/output に出力
        output_path = input_path / "output"
    else:
        # ファイルの場合: input_masked.ext として出力
        output_path = input_path.parent / f"{input_path.stem}_masked{input_path.suffix}"
    
    # 入力がフォルダかファイルか判定
    if input_path.is_dir():
        # フォルダ処理
        try:
            result = process_batch(
                input_folder=input_path,
                output_folder=output_path,
                is_masking=is_masking,
                model_path=args.model,
                confidence=args.confidence,
            )
            
            if result["failed"] > 0:
                sys.exit(1)
                
        except Exception as e:
            print(f"エラー: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        # 単一ファイル処理
        try:
            # 出力先がフォルダの場合、ファイル名を追加
            if output_path.is_dir() or (not output_path.suffix and not output_path.exists()):
                output_path.mkdir(parents=True, exist_ok=True)
                output_path = output_path / input_path.name
            else:
                # 出力ファイルの親フォルダを作成
                output_path.parent.mkdir(parents=True, exist_ok=True)
            
            result = process_image(
                input_path=str(input_path),
                output_path=str(output_path),
                is_masking=is_masking,
                model_path=args.model,
                confidence=args.confidence,
            )
            
            print(f"処理完了:")
            print(f"  入力: {result['input']}")
            print(f"  出力: {result['output']}")
            print(f"  サイズ: {result['original_size'][0]}x{result['original_size'][1]}")
            print(f"  検出数: {result['detections']}")
            print(f"  バナー: {'あり' if result['is_masking'] else 'なし'}")
            
        except Exception as e:
            print(f"エラー: {e}", file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    main()
