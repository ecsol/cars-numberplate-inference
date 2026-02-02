#!/usr/bin/env python3
"""
Compare original image with masked result side by side.
ảnh gốc và ảnh đã mask hiển thị song song để dễ kiểm tra.
"""
import sys
from pathlib import Path

# プロジェクトルートをパスに追加
sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2  # type: ignore
import numpy as np
from scripts.process_image import process_image


def compare_images(input_path: str, confidence: float = 0.1) -> None:
    """
    Hiển thị ảnh gốc và ảnh mask side-by-side.
    
    Args:
        input_path: Đường dẫn ảnh input
        confidence: Ngưỡng confidence cho detection
    """
    input_path = Path(input_path)
    if not input_path.exists():
        print(f"Error: File không tồn tại: {input_path}")
        return
    
    # Đọc ảnh gốc
    original = cv2.imread(str(input_path))
    if original is None:
        print(f"Error: Không thể đọc ảnh: {input_path}")
        return
    
    # Process và lấy ảnh đã mask
    output_path = f"/tmp/compare_output_{input_path.stem}.jpg"
    result = process_image(
        input_path=str(input_path),
        output_path=output_path,
        is_masking=True,
        model_path="models/best.pt",
        confidence=confidence,
    )
    
    # Đọc ảnh đã mask
    masked = cv2.imread(output_path)
    if masked is None:
        print("Error: Không thể tạo ảnh mask")
        return
    
    # Resize để 2 ảnh cùng kích thước
    h1, w1 = original.shape[:2]
    h2, w2 = masked.shape[:2]
    
    # Scale để fit màn hình (max 800px height)
    max_height = 800
    if h1 > max_height:
        scale = max_height / h1
        original = cv2.resize(original, (int(w1 * scale), int(h1 * scale)))
        masked = cv2.resize(masked, (int(w2 * scale), int(h2 * scale)))
    
    # Tạo ảnh side-by-side
    h, w = original.shape[:2]
    combined = np.zeros((h, w * 2 + 10, 3), dtype=np.uint8)
    combined[:, :w] = original
    combined[:, w + 10:] = masked
    
    # Thêm label
    cv2.putText(combined, "ORIGINAL", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(combined, "MASKED", (w + 20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Hiển thị thông tin detection
    det_count = result.get("detections", 0)
    cv2.putText(combined, f"Detections: {det_count}", (w + 20, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    
    print(f"\n=== COMPARISON ===")
    print(f"Input: {input_path}")
    print(f"Detections: {det_count}")
    print(f"\nControls:")
    print("  [Q/ESC] - Quit")
    print("  [S] - Save comparison image")
    
    window_name = f"Compare: {input_path.name}"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    cv2.imshow(window_name, combined)
    
    while True:
        key = cv2.waitKey(0) & 0xFF
        if key in [ord('q'), 27]:  # Q or ESC
            break
        elif key == ord('s'):
            save_path = f"/tmp/comparison_{input_path.stem}.jpg"
            cv2.imwrite(save_path, combined)
            print(f"Saved: {save_path}")
    
    cv2.destroyAllWindows()


def compare_folder(folder_path: str, confidence: float = 0.1) -> None:
    """
    So sánh tất cả ảnh trong folder.
    
    Args:
        folder_path: Đường dẫn folder chứa ảnh
        confidence: Ngưỡng confidence
    """
    folder = Path(folder_path)
    if not folder.exists():
        print(f"Error: Folder không tồn tại: {folder}")
        return
    
    # Lấy danh sách ảnh
    extensions = [".jpg", ".jpeg", ".png"]
    images = sorted([f for f in folder.iterdir() if f.suffix.lower() in extensions])
    
    if not images:
        print(f"Không tìm thấy ảnh trong: {folder}")
        return
    
    print(f"Found {len(images)} images")
    print("\nControls:")
    print("  [N/RIGHT] - Next image")
    print("  [P/LEFT] - Previous image")
    print("  [Q/ESC] - Quit")
    print("  [S] - Save current comparison")
    
    idx = 0
    while True:
        img_path = images[idx]
        print(f"\n[{idx + 1}/{len(images)}] {img_path.name}")
        
        # Đọc ảnh gốc
        original = cv2.imread(str(img_path))
        if original is None:
            print(f"Skip: Cannot read {img_path}")
            idx = (idx + 1) % len(images)
            continue
        
        # Process
        output_path = f"/tmp/compare_output_{img_path.stem}.jpg"
        result = process_image(
            input_path=str(img_path),
            output_path=output_path,
            is_masking=True,
            model_path="models/best.pt",
            confidence=confidence,
        )
        
        masked = cv2.imread(output_path)
        if masked is None:
            idx = (idx + 1) % len(images)
            continue
        
        # Scale
        h1, w1 = original.shape[:2]
        max_height = 700
        if h1 > max_height:
            scale = max_height / h1
            original = cv2.resize(original, (int(w1 * scale), int(h1 * scale)))
            masked = cv2.resize(masked, (int(masked.shape[1] * scale), int(masked.shape[0] * scale)))
        
        # Side-by-side
        h, w = original.shape[:2]
        combined = np.zeros((h, w * 2 + 10, 3), dtype=np.uint8)
        combined[:, :w] = original
        combined[:, w + 10:] = masked
        
        # Labels
        cv2.putText(combined, "ORIGINAL", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(combined, "MASKED", (w + 20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(combined, f"[{idx+1}/{len(images)}] {img_path.name}", (10, h - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        det_count = result.get("detections", 0)
        cv2.putText(combined, f"Det: {det_count}", (w + 20, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        cv2.imshow("Compare Results", combined)
        
        key = cv2.waitKey(0) & 0xFF
        if key in [ord('q'), 27]:
            break
        elif key in [ord('n'), 83, 3]:  # N, Right arrow
            idx = (idx + 1) % len(images)
        elif key in [ord('p'), 81, 2]:  # P, Left arrow
            idx = (idx - 1) % len(images)
        elif key == ord('s'):
            save_path = f"/tmp/comparison_{img_path.stem}.jpg"
            cv2.imwrite(save_path, combined)
            print(f"Saved: {save_path}")
    
    cv2.destroyAllWindows()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Compare original vs masked images")
    parser.add_argument("path", help="Image file or folder path")
    parser.add_argument("--confidence", "-c", type=float, default=0.1, 
                        help="Detection confidence threshold")
    
    args = parser.parse_args()
    
    path = Path(args.path)
    if path.is_dir():
        compare_folder(str(path), args.confidence)
    else:
        compare_images(str(path), args.confidence)
