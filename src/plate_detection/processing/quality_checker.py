"""
品質チェックモジュール

マスク処理後の画像品質をチェックし、未マスク流出を防止する。
"""

import cv2  # type: ignore
import numpy as np
from typing import Tuple, List


def calculate_edge_density(image: np.ndarray, region: np.ndarray) -> float:
    """
    指定領域のエッジ密度を計算

    高いエッジ密度 = 文字や数字が残っている可能性

    Args:
        image: 入力画像（BGR）
        region: 領域のポリゴン座標

    Returns:
        エッジ密度（0.0〜1.0）
    """
    # グレースケールに変換
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # マスクを作成
    mask = np.zeros(gray.shape, dtype=np.uint8)
    cv2.fillPoly(mask, [region.reshape(-1, 1, 2).astype(np.int32)], 255)

    # Cannyエッジ検出
    edges = cv2.Canny(gray, 50, 150)

    # マスク領域内のエッジ密度を計算
    masked_edges = cv2.bitwise_and(edges, edges, mask=mask)

    total_pixels = np.sum(mask > 0)
    if total_pixels == 0:
        return 0.0

    edge_pixels = np.sum(masked_edges > 0)
    return edge_pixels / total_pixels


def calculate_texture_variance(image: np.ndarray, region: np.ndarray) -> float:
    """
    指定領域のテクスチャ分散を計算

    高い分散 = 複雑なパターン（文字など）が残っている可能性

    Args:
        image: 入力画像（BGR）
        region: 領域のポリゴン座標

    Returns:
        テクスチャ分散
    """
    # グレースケールに変換
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # マスクを作成
    mask = np.zeros(gray.shape, dtype=np.uint8)
    cv2.fillPoly(mask, [region.reshape(-1, 1, 2).astype(np.int32)], 255)

    # Laplacianでテクスチャを抽出
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)

    # マスク領域内の分散を計算
    masked_laplacian = laplacian[mask > 0]

    if len(masked_laplacian) == 0:
        return 0.0

    return np.var(masked_laplacian)


def check_mask_completeness(
    original: np.ndarray,
    masked: np.ndarray,
    region: np.ndarray,
) -> Tuple[bool, dict]:
    """
    マスク処理が完全かチェック

    Args:
        original: 元画像（BGR）
        masked: マスク後の画像（BGR）
        region: マスク領域のポリゴン座標

    Returns:
        (合格フラグ, 詳細情報)
    """
    info = {
        "edge_density_before": 0.0,
        "edge_density_after": 0.0,
        "texture_variance_before": 0.0,
        "texture_variance_after": 0.0,
        "reduction_ratio": 0.0,
        "passed": False,
    }

    # 元画像のエッジ密度とテクスチャ
    info["edge_density_before"] = calculate_edge_density(original, region)
    info["texture_variance_before"] = calculate_texture_variance(original, region)

    # マスク後のエッジ密度とテクスチャ
    info["edge_density_after"] = calculate_edge_density(masked, region)
    info["texture_variance_after"] = calculate_texture_variance(masked, region)

    # 削減率を計算
    if info["edge_density_before"] > 0:
        info["reduction_ratio"] = 1 - (
            info["edge_density_after"] / info["edge_density_before"]
        )
    else:
        info["reduction_ratio"] = 1.0

    # 判定
    # エッジ密度が80%以上削減されていれば合格
    # または、マスク後のエッジ密度が十分低ければ合格
    if info["reduction_ratio"] >= 0.8 or info["edge_density_after"] < 0.02:
        info["passed"] = True

    return info["passed"], info


def check_all_regions(
    original: np.ndarray,
    masked: np.ndarray,
    regions: List[np.ndarray],
) -> Tuple[bool, List[dict]]:
    """
    全ての領域のマスク品質をチェック

    Args:
        original: 元画像（BGR）
        masked: マスク後の画像（BGR）
        regions: マスク領域のリスト

    Returns:
        (全て合格フラグ, 各領域の詳細情報リスト)
    """
    all_passed = True
    results = []

    for region in regions:
        passed, info = check_mask_completeness(original, masked, region)
        results.append(info)
        if not passed:
            all_passed = False

    return all_passed, results


def verify_no_plate_leak(
    masked_image: np.ndarray,
    detection_regions: List[np.ndarray],
    edge_threshold: float = 0.05,
    variance_threshold: float = 500.0,
) -> Tuple[bool, str]:
    """
    マスク後の画像にナンバープレートが残っていないか検証

    Args:
        masked_image: マスク後の画像
        detection_regions: 検出された領域
        edge_threshold: エッジ密度の閾値
        variance_threshold: テクスチャ分散の閾値

    Returns:
        (安全フラグ, メッセージ)
    """
    for i, region in enumerate(detection_regions):
        edge_density = calculate_edge_density(masked_image, region)
        texture_variance = calculate_texture_variance(masked_image, region)

        if edge_density > edge_threshold:
            return (
                False,
                f"領域{i + 1}: エッジ密度が高すぎます ({edge_density:.3f} > {edge_threshold})",
            )

        if texture_variance > variance_threshold:
            return (
                False,
                f"領域{i + 1}: テクスチャ分散が高すぎます ({texture_variance:.1f} > {variance_threshold})",
            )

    return True, "OK"
