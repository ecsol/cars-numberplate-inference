"""
ナンバープレートマスキング処理モジュール

射影変換（Perspective Transform）を使用して、
マスク画像をナンバープレートの形状に合わせて合成する。
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Tuple


# デフォルトマスク画像パス
DEFAULT_MASK_PATH = Path(__file__).parent.parent.parent.parent / "assets" / "plate_mask.png"


def load_mask_image(mask_path: Optional[Path] = None) -> np.ndarray:
    """
    マスク画像を読み込む
    
    Args:
        mask_path: マスク画像のパス（Noneの場合はデフォルト）
    
    Returns:
        マスク画像（BGRA形式）
    """
    path = mask_path or DEFAULT_MASK_PATH
    
    if not path.exists():
        # デフォルトマスクがない場合は白い画像を生成
        mask = np.ones((100, 200, 4), dtype=np.uint8) * 255
        return mask
    
    # アルファチャンネル付きで読み込み
    mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    
    if mask is None:
        mask = np.ones((100, 200, 4), dtype=np.uint8) * 255
        return mask
    
    # アルファチャンネルがない場合は追加
    if mask.shape[2] == 3:
        alpha = np.ones((mask.shape[0], mask.shape[1], 1), dtype=np.uint8) * 255
        mask = np.concatenate([mask, alpha], axis=2)
    
    return mask


def order_points(pts: np.ndarray) -> np.ndarray:
    """
    4点を左上、右上、右下、左下の順に並べ替える
    
    Args:
        pts: 4点の座標 (4, 2)
    
    Returns:
        並べ替えた座標 (4, 2)
    """
    rect = np.zeros((4, 2), dtype=np.float32)
    
    # 左上: x+y が最小
    # 右下: x+y が最大
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    
    # 右上: x-y が最大
    # 左下: x-y が最小
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    
    return rect


def perspective_transform_mask(
    image: np.ndarray,
    mask: np.ndarray,
    quad_points: np.ndarray,
    opacity: float = 1.0,
    padding: float = 0.02,
) -> np.ndarray:
    """
    マスク画像を射影変換してプレート位置に合成
    
    Args:
        image: 元画像（BGR）
        mask: マスク画像（BGRA）
        quad_points: プレートの4点座標 (4, 2)
        opacity: 透明度（0.0〜1.0）
        padding: マスクの余白（プレートより少し大きく、0.0〜0.1推奨）
    
    Returns:
        合成後の画像
    """
    h, w = image.shape[:2]
    mask_h, mask_w = mask.shape[:2]
    
    # 4点を正しい順序に並べ替え（左上、右上、右下、左下）
    dst_pts = order_points(quad_points.astype(np.float32))
    
    # パディングを適用してマスクを少し大きくする（自然な見た目のため）
    if padding > 0:
        # 各点から中心への方向を計算し、外側に拡張
        center = dst_pts.mean(axis=0)
        dst_pts_padded = np.zeros_like(dst_pts)
        for i in range(4):
            direction = dst_pts[i] - center
            dst_pts_padded[i] = dst_pts[i] + direction * padding
        dst_pts = dst_pts_padded
    
    # マスク画像の4点（左上、右上、右下、左下）
    src_pts = np.array([
        [0, 0],
        [mask_w - 1, 0],
        [mask_w - 1, mask_h - 1],
        [0, mask_h - 1]
    ], dtype=np.float32)
    
    # 射影変換行列を計算
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    
    # マスク画像を射影変換
    # INTER_LINEAR: ノイズが少なく安定（LANCZOS4はringingが発生する可能性）
    # BORDER_CONSTANT: 透明な境界
    warped = cv2.warpPerspective(
        mask, M, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0)
    )
    
    # アルファブレンディング
    result = alpha_blend(image, warped, opacity)
    
    return result


def alpha_blend(
    background: np.ndarray,
    overlay: np.ndarray,
    opacity: float = 1.0,
) -> np.ndarray:
    """
    アルファブレンディングで合成
    
    Args:
        background: 背景画像（BGR）
        overlay: オーバーレイ画像（BGRA）
        opacity: 追加の透明度（0.0〜1.0）
    
    Returns:
        合成後の画像（BGR）
    """
    if overlay.shape[2] != 4:
        # アルファチャンネルがない場合は単純上書き
        return overlay[:, :, :3]
    
    # アルファチャンネルを取得して正規化
    alpha = overlay[:, :, 3:4].astype(np.float32) / 255.0 * opacity
    
    # BGR部分を取得
    overlay_bgr = overlay[:, :, :3].astype(np.float32)
    background_f = background.astype(np.float32)
    
    # アルファブレンディング
    result = alpha * overlay_bgr + (1 - alpha) * background_f
    
    return result.astype(np.uint8)


def mask_plate_with_image(
    image: np.ndarray,
    polygon: np.ndarray,
    mask_image: Optional[np.ndarray] = None,
    opacity: float = 1.0,
) -> np.ndarray:
    """
    ナンバープレートをマスク画像で隠す
    
    Args:
        image: 元画像（BGR）
        polygon: プレートのポリゴン座標
        mask_image: マスク画像（BGRA）、Noneの場合は白で塗りつぶし
        opacity: 透明度（0.0〜1.0）
    
    Returns:
        マスク後の画像
    """
    result = image.copy()
    
    # ポリゴンを4点に変換
    quad = polygon_to_quad(polygon)
    
    if mask_image is None:
        # マスク画像がない場合は白で塗りつぶし（従来動作）
        cv2.fillPoly(result, [quad], color=(255, 255, 255))
    else:
        # マスク画像を射影変換して合成
        result = perspective_transform_mask(result, mask_image, quad, opacity)
    
    return result


def polygon_to_quad(polygon: np.ndarray) -> np.ndarray:
    """
    ポリゴンを4点の四角形に変換
    
    approxPolyDPを使用して、プレートの実際の形状を維持する。
    minAreaRectより精度が高い。
    
    Args:
        polygon: ポリゴン座標
    
    Returns:
        4点の四角形座標 (左上、右上、右下、左下の順)
    """
    polygon = polygon.reshape(-1, 2)
    
    if len(polygon) == 4:
        # 4点の場合は順序を正規化して返す
        return order_points(polygon.astype(np.float32)).astype(np.int32)
    
    # 凸包を計算
    hull = cv2.convexHull(polygon.astype(np.float32))
    
    # 多角形近似で4点に簡略化を試みる
    epsilon = 0.02 * cv2.arcLength(hull, True)
    approx = cv2.approxPolyDP(hull, epsilon, True)
    
    if len(approx) == 4:
        # 4点が得られた場合、順序を正規化
        pts = approx.reshape(-1, 2)
        return order_points(pts.astype(np.float32)).astype(np.int32)
    
    # 4点が得られない場合、四隅を推定
    pts = hull.reshape(-1, 2)
    
    # 左上: x+y が最小
    # 右下: x+y が最大
    # 右上: y-x が最小（xが大きく、yが小さい）
    # 左下: y-x が最大（xが小さく、yが大きい）
    sum_pts = pts.sum(axis=1)
    diff_pts = np.diff(pts, axis=1).reshape(-1)  # y - x
    
    tl = pts[np.argmin(sum_pts)]  # 左上
    br = pts[np.argmax(sum_pts)]  # 右下
    tr = pts[np.argmin(diff_pts)]  # 右上
    bl = pts[np.argmax(diff_pts)]  # 左下
    
    quad = np.array([tl, tr, br, bl], dtype=np.int32)
    
    return quad


def create_default_mask(width: int = 200, height: int = 100, color: Tuple[int, int, int] = (255, 255, 255)) -> np.ndarray:
    """
    デフォルトのマスク画像を生成
    
    Args:
        width: 幅
        height: 高さ
        color: 色（BGR）
    
    Returns:
        マスク画像（BGRA）
    """
    mask = np.zeros((height, width, 4), dtype=np.uint8)
    mask[:, :, 0] = color[0]  # B
    mask[:, :, 1] = color[1]  # G
    mask[:, :, 2] = color[2]  # R
    mask[:, :, 3] = 255       # A (不透明)
    
    return mask
