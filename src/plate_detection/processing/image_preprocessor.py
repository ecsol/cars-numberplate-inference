"""
画像前処理モジュール

低照度画像の補正、コントラスト調整などの前処理を行う。
"""

import cv2
import numpy as np
from typing import Tuple


def is_low_light(image: np.ndarray, threshold: float = 80.0) -> bool:
    """
    低照度画像かどうかを判定
    
    Args:
        image: 入力画像（BGR）
        threshold: 平均輝度の閾値
    
    Returns:
        低照度の場合True
    """
    # グレースケールに変換
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 平均輝度を計算
    mean_brightness = np.mean(gray)
    
    return mean_brightness < threshold


def adjust_brightness(image: np.ndarray, factor: float = 1.5) -> np.ndarray:
    """
    明るさを調整
    
    Args:
        image: 入力画像（BGR）
        factor: 明るさ係数（1.0が元の明るさ）
    
    Returns:
        調整後の画像
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv = hsv.astype(np.float32)
    
    # V（明度）チャンネルを調整
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * factor, 0, 255)
    
    hsv = hsv.astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def apply_clahe(image: np.ndarray, clip_limit: float = 2.0, tile_size: int = 8) -> np.ndarray:
    """
    CLAHE（Contrast Limited Adaptive Histogram Equalization）を適用
    
    低照度画像のコントラストを局所的に改善する。
    
    Args:
        image: 入力画像（BGR）
        clip_limit: コントラスト制限
        tile_size: タイルサイズ
    
    Returns:
        CLAHE適用後の画像
    """
    # LAB色空間に変換
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # CLAHEを作成してLチャンネルに適用
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
    l = clahe.apply(l)
    
    # 結合してBGRに戻す
    lab = cv2.merge([l, a, b])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def gamma_correction(image: np.ndarray, gamma: float = 1.5) -> np.ndarray:
    """
    ガンマ補正を適用
    
    Args:
        image: 入力画像（BGR）
        gamma: ガンマ値（1.0未満で明るく、1.0より大きいと暗く）
    
    Returns:
        ガンマ補正後の画像
    """
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype(np.uint8)
    return cv2.LUT(image, table)


def denoise(image: np.ndarray, strength: int = 10) -> np.ndarray:
    """
    ノイズ除去
    
    Args:
        image: 入力画像（BGR）
        strength: ノイズ除去の強度
    
    Returns:
        ノイズ除去後の画像
    """
    return cv2.fastNlMeansDenoisingColored(image, None, strength, strength, 7, 21)


def sharpen(image: np.ndarray, amount: float = 1.0) -> np.ndarray:
    """
    シャープネス強調
    
    Args:
        image: 入力画像（BGR）
        amount: シャープネスの強度
    
    Returns:
        シャープネス強調後の画像
    """
    kernel = np.array([[-1, -1, -1],
                       [-1, 9 + amount, -1],
                       [-1, -1, -1]])
    return cv2.filter2D(image, -1, kernel)


def preprocess_image(
    image: np.ndarray,
    auto_brightness: bool = True,
    apply_clahe_flag: bool = True,
    denoise_flag: bool = False,
    sharpen_flag: bool = False,
) -> Tuple[np.ndarray, dict]:
    """
    画像を前処理
    
    Args:
        image: 入力画像（BGR）
        auto_brightness: 自動明るさ補正
        apply_clahe_flag: CLAHE適用
        denoise_flag: ノイズ除去
        sharpen_flag: シャープネス強調
    
    Returns:
        (処理後の画像, 処理情報)
    """
    result = image.copy()
    info = {
        "is_low_light": False,
        "brightness_adjusted": False,
        "clahe_applied": False,
        "denoised": False,
        "sharpened": False,
    }
    
    # 低照度判定
    if is_low_light(result):
        info["is_low_light"] = True
        
        if auto_brightness:
            # ガンマ補正で明るく
            result = gamma_correction(result, gamma=0.7)
            info["brightness_adjusted"] = True
    
    # CLAHE適用
    if apply_clahe_flag:
        result = apply_clahe(result)
        info["clahe_applied"] = True
    
    # ノイズ除去
    if denoise_flag:
        result = denoise(result)
        info["denoised"] = True
    
    # シャープネス強調
    if sharpen_flag:
        result = sharpen(result)
        info["sharpened"] = True
    
    return result, info


def auto_preprocess_for_detection(image: np.ndarray) -> Tuple[np.ndarray, dict]:
    """
    検出用に自動前処理
    
    低照度画像の場合のみ補正を適用する。
    
    Args:
        image: 入力画像（BGR）
    
    Returns:
        (処理後の画像, 処理情報)
    """
    info = {"preprocessed": False, "is_low_light": False}
    
    if is_low_light(image):
        info["is_low_light"] = True
        info["preprocessed"] = True
        
        # CLAHE + ガンマ補正
        result = apply_clahe(image)
        result = gamma_correction(result, gamma=0.8)
        
        return result, info
    
    return image, info
