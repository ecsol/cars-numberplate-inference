"""
バナーオーバーレイ処理モジュール
"""

from pathlib import Path
from typing import Tuple, Optional

import cv2
import numpy as np


# デフォルトバナーパス
DEFAULT_BANNER_PATH = Path(__file__).parent.parent.parent.parent / "assets" / "banner_sample.png"


def hex_to_bgr(hex_color: str) -> Tuple[int, int, int]:
    """HEXカラーをBGRに変換"""
    hex_color = hex_color.lstrip("#")
    r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    return (b, g, r)


def load_banner(banner_path: Optional[Path] = None) -> np.ndarray:
    """バナー画像をロード（BGRA）"""
    path = banner_path or DEFAULT_BANNER_PATH
    banner = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    
    if banner is None:
        raise FileNotFoundError(f"バナーが見つかりません: {path}")
    
    # BGRAに変換（アルファチャンネルがない場合）
    if banner.shape[2] == 3:
        banner = cv2.cvtColor(banner, cv2.COLOR_BGR2BGRA)
    
    return banner


def resize_banner_to_width(banner: np.ndarray, target_width: int) -> np.ndarray:
    """バナーを指定幅にリサイズ（アスペクト比維持）"""
    h, w = banner.shape[:2]
    scale = target_width / w
    new_h = int(h * scale)
    return cv2.resize(banner, (target_width, new_h), interpolation=cv2.INTER_AREA)


def overlay_with_alpha(
    base: np.ndarray,
    overlay: np.ndarray,
    x: int,
    y: int,
    opacity: float = 1.0
) -> np.ndarray:
    """アルファブレンドでオーバーレイ"""
    result = base.copy()
    
    oh, ow = overlay.shape[:2]
    bh, bw = base.shape[:2]
    
    # 範囲チェック
    if x >= bw or y >= bh:
        return result
    
    # クリッピング
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(bw, x + ow), min(bh, y + oh)
    
    ox1 = x1 - x
    oy1 = y1 - y
    ox2 = ox1 + (x2 - x1)
    oy2 = oy1 + (y2 - y1)
    
    # オーバーレイ領域
    overlay_crop = overlay[oy1:oy2, ox1:ox2]
    base_crop = result[y1:y2, x1:x2]
    
    if overlay_crop.shape[2] == 4:
        # アルファチャンネルあり
        alpha = (overlay_crop[:, :, 3] / 255.0) * opacity
        alpha = alpha[:, :, np.newaxis]
        
        blended = (overlay_crop[:, :, :3] * alpha + base_crop * (1 - alpha)).astype(np.uint8)
        result[y1:y2, x1:x2] = blended
    else:
        # アルファなし
        if opacity < 1.0:
            blended = cv2.addWeighted(overlay_crop, opacity, base_crop, 1 - opacity, 0)
            result[y1:y2, x1:x2] = blended
        else:
            result[y1:y2, x1:x2] = overlay_crop
    
    return result


def overlay_banner(
    image: np.ndarray,
    mode: str = "overlay",
    position: str = "bottom",
    opacity: float = 1.0,
    bg_color: str = "#FFFFFF",
    banner_path: Optional[Path] = None,
) -> np.ndarray:
    """
    画像にバナーをオーバーレイする
    
    Args:
        image: 入力画像 (BGR)
        mode: "overlay" | "extend" | "fit"
        position: "bottom" | "top"
        opacity: 透明度 (0.0-1.0)
        bg_color: 背景色 (mode=fitの場合)
        banner_path: バナー画像パス
        
    Returns:
        オーバーレイ済み画像
    """
    img_h, img_w = image.shape[:2]
    
    # バナーロードとリサイズ
    banner = load_banner(banner_path)
    banner = resize_banner_to_width(banner, img_w)
    banner_h = banner.shape[0]
    
    if mode == "overlay":
        # バナーを画像の上に重ねる（サイズ変更なし）
        result = image.copy()
        if position == "bottom":
            y = img_h - banner_h
        else:
            y = 0
        result = overlay_with_alpha(result, banner, 0, y, opacity)
        
    elif mode == "extend":
        # キャンバスを拡張してバナーを追加
        new_h = img_h + banner_h
        result = np.zeros((new_h, img_w, 3), dtype=np.uint8)
        
        if position == "bottom":
            result[:img_h, :] = image
            banner_y = img_h
        else:
            result[banner_h:, :] = image
            banner_y = 0
        
        result = overlay_with_alpha(result, banner, 0, banner_y, opacity)
        
    elif mode == "fit":
        # 画像を縮小してバナーと一緒に元サイズに収める
        bg_bgr = hex_to_bgr(bg_color)
        result = np.full((img_h, img_w, 3), bg_bgr, dtype=np.uint8)
        
        # 画像の縮小サイズを計算
        available_h = img_h - banner_h
        if available_h <= 0:
            available_h = img_h // 2
        
        scale = min(img_w / img_w, available_h / img_h)
        new_img_w = int(img_w * scale)
        new_img_h = int(img_h * scale)
        
        resized_img = cv2.resize(image, (new_img_w, new_img_h), interpolation=cv2.INTER_AREA)
        
        # センタリング
        x_offset = (img_w - new_img_w) // 2
        
        if position == "bottom":
            y_offset = (available_h - new_img_h) // 2
            result[y_offset:y_offset + new_img_h, x_offset:x_offset + new_img_w] = resized_img
            banner_y = img_h - banner_h
        else:
            y_offset = banner_h + (available_h - new_img_h) // 2
            result[y_offset:y_offset + new_img_h, x_offset:x_offset + new_img_w] = resized_img
            banner_y = 0
        
        result = overlay_with_alpha(result, banner, 0, banner_y, opacity)
        
    else:
        raise ValueError(f"不明なモード: {mode}")
    
    return result
