"""
EXIF向き補正モジュール

画像のEXIF Orientationタグを読み取り、正しい向きに補正する。
スマートフォンやデジカメで撮影した画像の向きを自動修正。
"""

import cv2  # type: ignore
import numpy as np
from PIL import Image
from PIL.ExifTags import TAGS
import io
from typing import Tuple, Optional


# EXIF Orientation値と回転/反転の対応
ORIENTATION_OPERATIONS = {
    1: lambda img: img,  # 正常（そのまま）
    2: lambda img: cv2.flip(img, 1),  # 水平反転
    3: lambda img: cv2.rotate(img, cv2.ROTATE_180),  # 180度回転
    4: lambda img: cv2.flip(img, 0),  # 垂直反転
    5: lambda img: cv2.flip(
        cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE), 1
    ),  # 90度反時計回り+水平反転
    6: lambda img: cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE),  # 90度時計回り
    7: lambda img: cv2.flip(
        cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE), 1
    ),  # 90度時計回り+水平反転
    8: lambda img: cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE),  # 90度反時計回り
}


def get_exif_orientation(image_bytes: bytes) -> Optional[int]:
    """
    画像バイトデータからEXIF Orientationを取得

    Args:
        image_bytes: 画像のバイトデータ

    Returns:
        Orientation値（1-8）、取得できない場合はNone
    """
    try:
        pil_image = Image.open(io.BytesIO(image_bytes))
        exif_data = pil_image._getexif()

        if exif_data is None:
            return None

        for tag_id, value in exif_data.items():
            tag = TAGS.get(tag_id, tag_id)
            if tag == "Orientation":
                return value

        return None
    except Exception:
        return None


def correct_orientation(image: np.ndarray, orientation: int) -> np.ndarray:
    """
    EXIF Orientation値に基づいて画像を補正

    Args:
        image: OpenCV画像（BGR）
        orientation: EXIF Orientation値（1-8）

    Returns:
        補正後の画像
    """
    if orientation in ORIENTATION_OPERATIONS:
        return ORIENTATION_OPERATIONS[orientation](image)
    return image


def auto_orient_image(image_bytes: bytes) -> Tuple[np.ndarray, dict]:
    """
    画像バイトデータを読み込み、EXIF向きを自動補正

    Args:
        image_bytes: 画像のバイトデータ

    Returns:
        (補正後の画像, 処理情報)
    """
    info = {
        "exif_orientation": None,
        "corrected": False,
        "original_size": None,
        "corrected_size": None,
    }

    # 画像を読み込み
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if image is None:
        return None, info

    info["original_size"] = (image.shape[1], image.shape[0])

    # EXIF Orientationを取得
    orientation = get_exif_orientation(image_bytes)
    info["exif_orientation"] = orientation

    # 補正が必要な場合
    if orientation is not None and orientation != 1:
        image = correct_orientation(image, orientation)
        info["corrected"] = True
        info["corrected_size"] = (image.shape[1], image.shape[0])
    else:
        info["corrected_size"] = info["original_size"]

    return image, info


def get_exif_info(image_bytes: bytes) -> dict:
    """
    画像のEXIF情報を取得（デバッグ用）

    Args:
        image_bytes: 画像のバイトデータ

    Returns:
        EXIF情報の辞書
    """
    exif_info = {}

    try:
        pil_image = Image.open(io.BytesIO(image_bytes))
        exif_data = pil_image._getexif()

        if exif_data:
            for tag_id, value in exif_data.items():
                tag = TAGS.get(tag_id, tag_id)
                # バイトデータは文字列に変換
                if isinstance(value, bytes):
                    try:
                        value = value.decode("utf-8", errors="ignore")
                    except Exception:
                        value = str(value)
                exif_info[tag] = value
    except Exception:
        pass

    return exif_info
