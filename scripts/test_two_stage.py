#!/usr/bin/env python3
"""
Test Two-Stage License Plate Detection Pipeline

Stage 1: YOLO-Seg (detect vùng biển số)
Stage 2: YOLO-Pose (detect 4 góc chính xác)
"""

import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np
import argparse
from ultralytics import YOLO


def order_points(pts: np.ndarray) -> np.ndarray:
    """4点を左上、右上、右下、左下の順に並べ替え"""
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # tl
    rect[2] = pts[np.argmax(s)]  # br
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # tr
    rect[3] = pts[np.argmax(diff)]  # bl
    return rect


def mask_plate_with_corners(
    image: np.ndarray,
    corners: np.ndarray,
    mask_image: np.ndarray,
) -> np.ndarray:
    """4コーナーを使ってマスク画像を合成"""
    h, w = image.shape[:2]
    mask_h, mask_w = mask_image.shape[:2]

    # 順序を正規化
    dst_pts = order_points(corners.astype(np.float32))

    # マスク画像の4コーナー
    src_pts = np.array(
        [[0, 0], [mask_w - 1, 0], [mask_w - 1, mask_h - 1], [0, mask_h - 1]],
        dtype=np.float32,
    )

    # 射影変換
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv2.warpPerspective(
        mask_image,
        M,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0),
    )

    # アルファブレンディング
    if warped.shape[2] == 4:
        alpha = warped[:, :, 3:4].astype(np.float32) / 255.0
        overlay_bgr = warped[:, :, :3].astype(np.float32)
        background_f = image.astype(np.float32)
        result = alpha * overlay_bgr + (1 - alpha) * background_f
        return result.astype(np.uint8)
    return image


def two_stage_detect(
    image: np.ndarray,
    seg_model: YOLO,
    pose_model: YOLO,
    seg_conf: float = 0.3,
    pose_conf: float = 0.3,
):
    """
    Two-Stage Detection (FIXED: run pose on FULL image for better accuracy)

    Returns:
        List of detections with corners
    """
    results = []

    # Run POSE on FULL image (not crop!) - much better accuracy
    pose_results = pose_model.predict(image, conf=pose_conf, verbose=False)

    if pose_results and len(pose_results[0].boxes) > 0:
        for box, kpts in zip(pose_results[0].boxes, pose_results[0].keypoints):
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            box_conf = float(box.conf[0])

            corners = None
            kpt_conf = 0.0

            if kpts.xy is not None and len(kpts.xy[0]) >= 4:
                corners = kpts.xy[0].cpu().numpy()[:4]
                if kpts.conf is not None:
                    kpt_conf = float(kpts.conf[0][:4].mean())

            if corners is not None:
                results.append(
                    {
                        "bbox": [x1, y1, x2, y2],
                        "corners": corners,
                        "seg_conf": box_conf,
                        "pose_conf": kpt_conf,
                    }
                )

    # Fallback to segmentation if pose failed
    if not results:
        seg_results = seg_model.predict(image, conf=seg_conf, verbose=False)
        if seg_results and len(seg_results[0].boxes) > 0:
            for box in seg_results[0].boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                seg_conf_score = float(box.conf[0])
                corners = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])
                results.append(
                    {
                        "bbox": [x1, y1, x2, y2],
                        "corners": corners,
                        "seg_conf": seg_conf_score,
                        "pose_conf": 0.0,
                    }
                )

    return results


def main():
    parser = argparse.ArgumentParser(description="Test Two-Stage Pipeline")
    parser.add_argument("--input", type=str, required=True, help="Input image path")
    parser.add_argument("--output", type=str, default="output/two_stage_test.jpg")
    parser.add_argument(
        "--seg-model", type=str, default="models/yolo26x_lambda_best.pt"
    )
    parser.add_argument("--pose-model", type=str, default="models/yolo26x_pose_best.pt")
    parser.add_argument("--mask", type=str, default="assets/plate_mask.png")
    parser.add_argument("--debug", action="store_true", help="Show debug visualization")

    args = parser.parse_args()

    # Load models
    print("Loading models...")
    seg_model = YOLO(args.seg_model)
    pose_model = YOLO(args.pose_model)

    # Load image
    image = cv2.imread(args.input)
    if image is None:
        print(f"Error: Cannot load image {args.input}")
        return

    print(f"Image size: {image.shape[1]}x{image.shape[0]}")

    # Load mask
    mask_image = cv2.imread(args.mask, cv2.IMREAD_UNCHANGED)
    if mask_image is None:
        print(f"Warning: Cannot load mask {args.mask}, using white fill")
        mask_image = np.ones((100, 200, 4), dtype=np.uint8) * 255

    # Two-stage detection
    print("Running two-stage detection...")
    detections = two_stage_detect(image, seg_model, pose_model)

    print(f"Detected {len(detections)} plates")

    # Process results
    result = image.copy()
    debug_img = image.copy() if args.debug else None

    for i, det in enumerate(detections):
        corners = det["corners"]
        print(
            f"  Plate {i}: seg_conf={det['seg_conf']:.2f}, pose_conf={det['pose_conf']:.2f}"
        )

        if args.debug:
            # Draw corners on debug image
            for j, pt in enumerate(corners):
                color = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0)][j]
                cv2.circle(debug_img, (int(pt[0]), int(pt[1])), 8, color, -1)
                cv2.putText(
                    debug_img,
                    str(j),
                    (int(pt[0]) + 10, int(pt[1])),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    color,
                    2,
                )

            # Draw polygon
            pts = corners.astype(np.int32)
            cv2.polylines(debug_img, [pts], True, (0, 255, 0), 2)

        # Apply mask
        result = mask_plate_with_corners(result, corners, mask_image)

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(output_path), result)
    print(f"Saved: {output_path}")

    if args.debug and debug_img is not None:
        debug_path = output_path.parent / f"debug_{output_path.name}"
        cv2.imwrite(str(debug_path), debug_img)
        print(f"Debug: {debug_path}")


if __name__ == "__main__":
    main()
