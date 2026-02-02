"""
Two-Stage License Plate Detection Pipeline

Stage 1: YOLO-Seg - Detect vùng biển số
Stage 2: YOLO-Pose - Detect 4 góc chính xác

Output: 4 corners với độ chính xác cao nhất
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from ultralytics import YOLO


class TwoStageDetector:
    """
    Two-Stage License Plate Detector
    
    Stage 1: Segmentation model để detect vùng biển số
    Stage 2: Pose model để detect 4 góc chính xác
    """
    
    def __init__(
        self,
        seg_model_path: str,
        pose_model_path: str,
        device: str = "cpu",
    ):
        """
        初期化
        
        Args:
            seg_model_path: Stage 1 segmentation model path
            pose_model_path: Stage 2 pose model path
            device: cuda / mps / cpu
        """
        self.seg_model = YOLO(seg_model_path)
        self.pose_model = YOLO(pose_model_path)
        self.device = device
        
        # パディング設定（crop時に余裕を持たせる）
        self.crop_padding = 0.2  # 20% padding
    
    def detect(
        self,
        image: np.ndarray,
        seg_conf: float = 0.3,
        pose_conf: float = 0.5,
    ) -> List[Dict]:
        """
        Two-Stage検出を実行
        
        Args:
            image: 入力画像 (BGR)
            seg_conf: Stage 1の信頼度閾値
            pose_conf: Stage 2の信頼度閾値
        
        Returns:
            検出結果のリスト:
            [
                {
                    'bbox': [x1, y1, x2, y2],
                    'corners': [[x1,y1], [x2,y2], [x3,y3], [x4,y4]],  # tl, tr, br, bl
                    'confidence': float,
                    'seg_mask': np.array (optional)
                },
                ...
            ]
        """
        h, w = image.shape[:2]
        results = []
        
        # === Stage 1: Segmentation ===
        seg_results = self.seg_model.predict(
            image,
            conf=seg_conf,
            device=self.device,
            verbose=False,
        )
        
        if not seg_results or len(seg_results[0].boxes) == 0:
            return results
        
        # 各検出に対してStage 2を実行
        for i, box in enumerate(seg_results[0].boxes):
            # Bounding box取得
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            seg_conf_score = float(box.conf[0])
            
            # Segmentation mask取得（あれば）
            seg_mask = None
            if seg_results[0].masks is not None and i < len(seg_results[0].masks):
                seg_mask = seg_results[0].masks[i].xy[0]
            
            # === Stage 2: Pose (Corner Detection) ===
            # Crop with padding
            crop_x1, crop_y1, crop_x2, crop_y2, crop_img = self._crop_with_padding(
                image, x1, y1, x2, y2
            )
            
            # Pose detection on cropped image
            pose_results = self.pose_model.predict(
                crop_img,
                conf=pose_conf,
                device=self.device,
                verbose=False,
            )
            
            corners = None
            pose_conf_score = 0.0
            
            if pose_results and len(pose_results[0].keypoints) > 0:
                # Keypoints取得
                kpts = pose_results[0].keypoints
                if kpts.xy is not None and len(kpts.xy) > 0:
                    # 最初の検出のkeypoints
                    keypoints = kpts.xy[0].cpu().numpy()  # (4, 2)
                    
                    if len(keypoints) == 4:
                        # Crop座標を元画像座標に変換
                        corners = []
                        for kp in keypoints:
                            orig_x = kp[0] + crop_x1
                            orig_y = kp[1] + crop_y1
                            corners.append([orig_x, orig_y])
                        corners = np.array(corners)
                        
                        # Confidence
                        if kpts.conf is not None:
                            pose_conf_score = float(kpts.conf[0].mean())
            
            # Cornersが取得できなかった場合、bboxの4隅を使用
            if corners is None:
                corners = np.array([
                    [x1, y1],  # tl
                    [x2, y1],  # tr
                    [x2, y2],  # br
                    [x1, y2],  # bl
                ])
                pose_conf_score = seg_conf_score * 0.5  # Lower confidence
            
            results.append({
                'bbox': [x1, y1, x2, y2],
                'corners': corners,
                'confidence': (seg_conf_score + pose_conf_score) / 2,
                'seg_confidence': seg_conf_score,
                'pose_confidence': pose_conf_score,
                'seg_mask': seg_mask,
            })
        
        return results
    
    def _crop_with_padding(
        self,
        image: np.ndarray,
        x1: float, y1: float,
        x2: float, y2: float,
    ) -> Tuple[int, int, int, int, np.ndarray]:
        """
        パディング付きでクロップ
        
        Returns:
            (crop_x1, crop_y1, crop_x2, crop_y2, cropped_image)
        """
        h, w = image.shape[:2]
        
        # Bboxサイズ
        bw = x2 - x1
        bh = y2 - y1
        
        # パディング追加
        pad_x = bw * self.crop_padding
        pad_y = bh * self.crop_padding
        
        # クリップ
        crop_x1 = max(0, int(x1 - pad_x))
        crop_y1 = max(0, int(y1 - pad_y))
        crop_x2 = min(w, int(x2 + pad_x))
        crop_y2 = min(h, int(y2 + pad_y))
        
        crop_img = image[crop_y1:crop_y2, crop_x1:crop_x2]
        
        return crop_x1, crop_y1, crop_x2, crop_y2, crop_img


def mask_plate_with_corners(
    image: np.ndarray,
    corners: np.ndarray,
    mask_image: np.ndarray,
    opacity: float = 1.0,
) -> np.ndarray:
    """
    4コーナーを使ってマスク画像を合成
    
    Args:
        image: 元画像 (BGR)
        corners: 4コーナー座標 [[x1,y1], [x2,y2], [x3,y3], [x4,y4]] (tl, tr, br, bl)
        mask_image: マスク画像 (BGRA)
        opacity: 透明度
    
    Returns:
        合成後の画像
    """
    h, w = image.shape[:2]
    mask_h, mask_w = mask_image.shape[:2]
    
    # 目的座標（4コーナー）
    dst_pts = corners.astype(np.float32)
    
    # マスク画像の4コーナー
    src_pts = np.array([
        [0, 0],
        [mask_w - 1, 0],
        [mask_w - 1, mask_h - 1],
        [0, mask_h - 1]
    ], dtype=np.float32)
    
    # 射影変換
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    
    warped = cv2.warpPerspective(
        mask_image, M, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0)
    )
    
    # アルファブレンディング
    if warped.shape[2] == 4:
        alpha = warped[:, :, 3:4].astype(np.float32) / 255.0 * opacity
        overlay_bgr = warped[:, :, :3].astype(np.float32)
        background_f = image.astype(np.float32)
        result = alpha * overlay_bgr + (1 - alpha) * background_f
        return result.astype(np.uint8)
    else:
        return warped[:, :, :3]
