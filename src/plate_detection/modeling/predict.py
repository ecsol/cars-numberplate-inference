"""
Inference utilities for license plate detection.

推論ユーティリティ
"""

from pathlib import Path
from typing import Optional, List, Tuple
import numpy as np
import cv2

from ultralytics import YOLO

from ..config import settings, PROJECT_ROOT


class PlateDetector:
    """
    ナンバープレート検出器クラス
    """
    
    def __init__(
        self,
        model_path: Optional[Path] = None,
        confidence: Optional[float] = None,
        device: Optional[str] = None,
    ):
        """
        検出器を初期化する。
        
        Args:
            model_path: モデルファイルパス
            confidence: 信頼度閾値
            device: 推論デバイス
        """
        self.model_path = model_path or settings.model_path
        self.confidence = confidence or settings.confidence_threshold
        self.device = device or settings.device
        
        # モデルをロード
        self.model = YOLO(str(self.model_path))
    
    def predict(
        self,
        image: np.ndarray,
        verbose: bool = False,
    ) -> List[dict]:
        """
        画像からナンバープレートを検出する。
        
        Args:
            image: BGR形式の画像 (numpy array)
            verbose: 詳細出力フラグ
            
        Returns:
            検出結果のリスト。各要素は以下を含む:
            - 'mask': セグメンテーションマスク (numpy array)
            - 'bbox': バウンディングボックス [x1, y1, x2, y2]
            - 'confidence': 信頼度スコア
        """
        results = self.model.predict(
            source=image,
            conf=self.confidence,
            save=False,
            verbose=verbose,
        )
        
        detections = []
        
        for r in results:
            if r.masks is not None:
                for mask, box, conf in zip(
                    r.masks.xy,
                    r.boxes.xyxy.cpu().numpy(),
                    r.boxes.conf.cpu().numpy(),
                ):
                    detections.append({
                        "mask": np.array(mask, dtype=np.int32),
                        "bbox": box.tolist(),
                        "confidence": float(conf),
                    })
        
        return detections
    
    def mask_plates(
        self,
        image: np.ndarray,
        color: Tuple[int, int, int] = (255, 255, 255),
    ) -> np.ndarray:
        """
        検出したナンバープレートをマスクする。
        
        Args:
            image: 入力画像 (BGR)
            color: マスク色 (BGR)
            
        Returns:
            マスク済み画像
        """
        detections = self.predict(image)
        result = image.copy()
        
        for det in detections:
            mask = det["mask"]
            
            # 凸包を計算
            hull = cv2.convexHull(mask)
            
            # 多角形近似
            epsilon = 0.02 * cv2.arcLength(hull, True)
            approx = cv2.approxPolyDP(hull, epsilon, True)
            
            # 4角形に変換
            if len(approx) != 4:
                pts = hull.reshape(-1, 2)
                sum_pts = pts.sum(axis=1)
                diff_pts = np.diff(pts, axis=1).reshape(-1)
                
                tl = pts[np.argmin(sum_pts)]
                br = pts[np.argmax(sum_pts)]
                tr = pts[np.argmin(diff_pts)]
                bl = pts[np.argmax(diff_pts)]
                
                approx = np.array([tl, tr, br, bl], dtype=np.int32)
            
            quad = approx.reshape(-1, 2)
            
            # 平行四辺形に調整
            tl, tr, br, bl = quad[:4] if len(quad) >= 4 else (quad[0], quad[0], quad[0], quad[0])
            
            top_vec = tr - tl
            bottom_vec = br - bl
            avg_vec = (top_vec + bottom_vec) / 2
            
            tr = tl + avg_vec
            br = bl + avg_vec
            
            left_vec = bl - tl
            right_vec = br - tr
            avg_lr = (left_vec + right_vec) / 2
            
            bl = tl + avg_lr
            br = tr + avg_lr
            
            quad_parallel = np.array(
                [tl, tr, br, bl], dtype=np.int32
            ).reshape(-1, 1, 2)
            
            # マスクを塗りつぶし
            cv2.fillPoly(result, [quad_parallel], color=color)
        
        return result


# グローバルインスタンス（API用）
_detector: Optional[PlateDetector] = None


def get_detector() -> PlateDetector:
    """
    シングルトン検出器インスタンスを取得する。
    """
    global _detector
    if _detector is None:
        _detector = PlateDetector()
    return _detector
