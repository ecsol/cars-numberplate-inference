"""
OCR検証モジュール - 日本のナンバープレートのテキスト検証

Japanese license plate format:
- Line 1: 地名 (2-4 chars) + 分類番号 (3 digits) - 例: "品川 500"
- Line 2: ひらがな (1 char) + 一連番号 (4 digits with hyphen) - 例: "あ 12-34"
"""

import re
from typing import Optional, Tuple
import numpy as np
import cv2


# グローバルOCRリーダー（遅延初期化）
_ocr_reader = None


def get_ocr_reader():
    """OCRリーダーを取得（シングルトン）"""
    global _ocr_reader
    if _ocr_reader is None:
        import easyocr
        # 日本語と英数字をサポート
        _ocr_reader = easyocr.Reader(['ja', 'en'], gpu=False, verbose=False)
    return _ocr_reader


# 日本語文字パターン
HIRAGANA_PATTERN = re.compile(r'[\u3040-\u309F]')  # ひらがな
KATAKANA_PATTERN = re.compile(r'[\u30A0-\u30FF]')  # カタカナ
KANJI_PATTERN = re.compile(r'[\u4E00-\u9FFF]')     # 漢字
NUMBER_PATTERN = re.compile(r'\d')                  # 数字

# 地名リスト（一部）- 検出精度向上用
PREFECTURE_NAMES = [
    "札幌", "函館", "旭川", "室蘭", "釧路", "帯広", "北見",
    "青森", "八戸", "岩手", "盛岡", "宮城", "仙台", "秋田",
    "山形", "庄内", "福島", "会津", "いわき",
    "水戸", "土浦", "つくば", "宇都宮", "とちぎ",
    "群馬", "高崎", "前橋", "大宮", "川口", "所沢", "熊谷", "春日部", "越谷", "川越",
    "千葉", "成田", "習志野", "市川", "船橋", "柏", "野田",
    "品川", "練馬", "足立", "八王子", "多摩", "世田谷", "杉並",
    "横浜", "川崎", "相模", "湘南",
    "新潟", "長岡", "富山", "石川", "金沢", "福井",
    "山梨", "長野", "松本", "諏訪",
    "岐阜", "飛騨", "静岡", "浜松", "沼津",
    "名古屋", "豊橋", "三河", "尾張小牧", "岡崎", "豊田", "一宮",
    "三重", "鈴鹿",
    "滋賀", "京都",
    "大阪", "なにわ", "和泉", "堺",
    "神戸", "姫路",
    "奈良", "和歌山",
    "鳥取", "島根", "岡山", "倉敷",
    "広島", "福山", "山口", "下関",
    "徳島", "香川", "愛媛", "高知",
    "福岡", "北九州", "久留米", "筑豊",
    "佐賀", "長崎", "佐世保", "熊本", "大分",
    "宮崎", "鹿児島", "沖縄",
]


def extract_plate_region(img: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    """ポリゴンから画像領域を切り出す"""
    # バウンディングボックスを取得
    x, y, w, h = cv2.boundingRect(polygon)
    
    # 少しパディングを追加
    padding = 5
    x = max(0, x - padding)
    y = max(0, y - padding)
    w = min(img.shape[1] - x, w + 2 * padding)
    h = min(img.shape[0] - y, h + 2 * padding)
    
    # 領域を切り出し
    roi = img[y:y+h, x:x+w]
    
    return roi


def preprocess_for_ocr(img: np.ndarray) -> np.ndarray:
    """OCR用に画像を前処理"""
    # グレースケール変換
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    
    # コントラスト強調
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    # リサイズ（OCR精度向上）
    h, w = enhanced.shape
    if w < 200:
        scale = 200 / w
        enhanced = cv2.resize(enhanced, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    
    return enhanced


def validate_plate_text(text: str) -> Tuple[bool, float]:
    """
    テキストが日本のナンバープレート形式かどうかを検証
    
    Returns:
        (is_valid, confidence_score)
    """
    if not text or len(text) < 3:
        return False, 0.0
    
    # スコア計算
    score = 0.0
    
    # 数字があるか
    if NUMBER_PATTERN.search(text):
        score += 0.3
    
    # 日本語文字があるか
    has_hiragana = bool(HIRAGANA_PATTERN.search(text))
    has_katakana = bool(KATAKANA_PATTERN.search(text))
    has_kanji = bool(KANJI_PATTERN.search(text))
    
    if has_hiragana:
        score += 0.2
    if has_kanji:
        score += 0.3
    if has_katakana:
        score += 0.1
    
    # 地名が含まれているか
    for name in PREFECTURE_NAMES:
        if name in text:
            score += 0.3
            break
    
    # ナンバープレート形式のパターン
    # 例: "品川 500 あ 12-34" または部分的なマッチ
    plate_pattern = re.compile(r'(\d{1,4}[-・]?\d{0,4})')
    if plate_pattern.search(text):
        score += 0.2
    
    # スコアを0-1に正規化
    score = min(1.0, score)
    
    # 閾値判定（0.3以上で有効とみなす）
    is_valid = score >= 0.3
    
    return is_valid, score


def ocr_validate_plate(
    img: np.ndarray, 
    polygon: np.ndarray,
    min_confidence: float = 0.3,
) -> Tuple[bool, str, float]:
    """
    OCRでナンバープレートを検証
    
    Args:
        img: 元画像
        polygon: 検出されたポリゴン
        min_confidence: 最小信頼度
    
    Returns:
        (is_valid, detected_text, score)
    """
    try:
        # 領域を切り出し
        roi = extract_plate_region(img, polygon)
        
        if roi.size == 0:
            return False, "", 0.0
        
        # 前処理
        processed = preprocess_for_ocr(roi)
        
        # OCR実行
        reader = get_ocr_reader()
        results = reader.readtext(processed, detail=1)
        
        if not results:
            return False, "", 0.0
        
        # テキストを結合
        texts = []
        total_conf = 0.0
        for (bbox, text, conf) in results:
            texts.append(text)
            total_conf += conf
        
        combined_text = " ".join(texts)
        avg_conf = total_conf / len(results) if results else 0.0
        
        # テキスト検証
        is_valid, text_score = validate_plate_text(combined_text)
        
        # 総合スコア
        final_score = (avg_conf + text_score) / 2
        
        return is_valid and final_score >= min_confidence, combined_text, final_score
        
    except Exception as e:
        print(f"OCR error: {e}")
        return False, "", 0.0
