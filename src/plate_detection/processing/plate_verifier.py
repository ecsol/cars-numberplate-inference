#!/usr/bin/env python3
"""
Plate Verifier using OCR
ナンバープレートを検証する - 数字/文字が含まれているかチェック

Biển số xe PHẢI có số bên trong. Module này dùng OCR để verify
vùng được detect có phải là biển số thật hay không.
"""
import re
from typing import Optional, Tuple

import cv2  # type: ignore
import numpy as np

# EasyOCR - lazy load để tránh slow startup
_ocr_reader = None


def get_ocr_reader():
    """
    Lazy load EasyOCR reader (chỉ load khi cần)
    
    Returns:
        EasyOCR Reader instance
    """
    global _ocr_reader
    if _ocr_reader is None:
        import easyocr
        # Hỗ trợ tiếng Nhật (ja) và tiếng Anh (en) cho số/chữ
        _ocr_reader = easyocr.Reader(['ja', 'en'], gpu=False, verbose=False)
    return _ocr_reader


def extract_plate_region(
    image: np.ndarray,
    polygon: np.ndarray,
    padding: float = 0.05
) -> np.ndarray:
    """
    Crop vùng biển số từ ảnh gốc
    
    Args:
        image: Ảnh gốc (BGR)
        polygon: Polygon coordinates của plate
        padding: Padding thêm xung quanh (0.0 - 0.2)
    
    Returns:
        Cropped plate region
    """
    # Lấy bounding box từ polygon
    x_coords = polygon[:, 0]
    y_coords = polygon[:, 1]
    
    x_min, x_max = int(x_coords.min()), int(x_coords.max())
    y_min, y_max = int(y_coords.min()), int(y_coords.max())
    
    # Thêm padding
    h, w = image.shape[:2]
    pad_x = int((x_max - x_min) * padding)
    pad_y = int((y_max - y_min) * padding)
    
    x_min = max(0, x_min - pad_x)
    x_max = min(w, x_max + pad_x)
    y_min = max(0, y_min - pad_y)
    y_max = min(h, y_max + pad_y)
    
    return image[y_min:y_max, x_min:x_max]


def contains_plate_characters(text: str) -> bool:
    """
    Kiểm tra text có chứa đặc trưng biển số không
    
    Điều kiện:
    - BẮT BUỘC: ít nhất 1 số (0-9)
    - BONUS: có hiragana, kanji càng tốt nhưng không bắt buộc
    
    Biển số Nhật có thể chỉ có 1 số (VD: biển đặc biệt)
    
    Args:
        text: Text từ OCR
    
    Returns:
        True nếu có ít nhất 1 số
    """
    if not text:
        return False
    
    # Đếm số lượng chữ số - ĐIỀU KIỆN BẮT BUỘC
    digit_count = len(re.findall(r'\d', text))
    
    # Biển số PHẢI có ít nhất 1 số
    return digit_count >= 1


def verify_plate_with_ocr(
    image: np.ndarray,
    polygon: np.ndarray,
    min_confidence: float = 0.3
) -> Tuple[bool, str, float]:
    """
    Verify xem vùng detect có phải là biển số thật không (có số bên trong)
    
    Args:
        image: Ảnh gốc (BGR)
        polygon: Polygon của vùng detect
        min_confidence: Ngưỡng confidence OCR tối thiểu
    
    Returns:
        Tuple[is_valid, detected_text, confidence]
        - is_valid: True nếu là biển số thật (có số)
        - detected_text: Text OCR được
        - confidence: Độ tin cậy trung bình
    """
    try:
        # Crop vùng plate
        plate_region = extract_plate_region(image, polygon)
        
        if plate_region.size == 0:
            return False, "", 0.0
        
        # Resize nếu quá nhỏ (OCR cần ảnh đủ lớn)
        min_height = 50
        if plate_region.shape[0] < min_height:
            scale = min_height / plate_region.shape[0]
            plate_region = cv2.resize(plate_region, None, fx=scale, fy=scale)
        
        # Chạy OCR
        reader = get_ocr_reader()
        results = reader.readtext(plate_region)
        
        if not results:
            return False, "", 0.0
        
        # Ghép tất cả text và tính confidence trung bình
        texts = []
        confidences = []
        
        for (bbox, text, conf) in results:
            if conf >= min_confidence:
                texts.append(text)
                confidences.append(conf)
        
        if not texts:
            return False, "", 0.0
        
        combined_text = " ".join(texts)
        avg_confidence = sum(confidences) / len(confidences)
        
        # Kiểm tra có ký tự biển số không
        is_valid = contains_plate_characters(combined_text)
        
        return is_valid, combined_text, avg_confidence
        
    except Exception as e:
        print(f"OCR verification error: {e}")
        return False, "", 0.0


def filter_valid_plates(
    image: np.ndarray,
    detections: list,
    min_confidence: float = 0.3
) -> list:
    """
    Lọc chỉ giữ lại các detection là biển số thật (có số bên trong)
    
    Args:
        image: Ảnh gốc (BGR)
        detections: List các detection từ YOLO [{'polygon': np.array, ...}, ...]
        min_confidence: Ngưỡng OCR confidence
    
    Returns:
        List các detection đã được verify là biển số thật
    """
    valid_detections = []
    
    for det in detections:
        polygon = det.get('polygon')
        if polygon is None:
            continue
        
        is_valid, text, conf = verify_plate_with_ocr(image, polygon, min_confidence)
        
        if is_valid:
            # Thêm thông tin OCR vào detection
            det['ocr_text'] = text
            det['ocr_confidence'] = conf
            valid_detections.append(det)
        else:
            print(f"Filtered out non-plate detection (OCR: '{text}', conf: {conf:.2f})")
    
    return valid_detections


# === Quick test ===
if __name__ == "__main__":
    import sys
    from pathlib import Path
    
    if len(sys.argv) < 2:
        print("Usage: python plate_verifier.py <image_path>")
        sys.exit(1)
    
    img_path = sys.argv[1]
    img = cv2.imread(img_path)
    
    if img is None:
        print(f"Cannot read image: {img_path}")
        sys.exit(1)
    
    # Test OCR trên toàn bộ ảnh
    print("Testing OCR on full image...")
    reader = get_ocr_reader()
    results = reader.readtext(img)
    
    print(f"\nFound {len(results)} text regions:")
    for (bbox, text, conf) in results:
        print(f"  '{text}' (confidence: {conf:.2f})")
    
    # Check if contains plate characters
    all_text = " ".join([r[1] for r in results])
    is_plate = contains_plate_characters(all_text)
    print(f"\nIs likely a plate: {is_plate}")
