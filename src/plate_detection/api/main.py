"""
ナンバープレート検出API
"""

import base64
from typing import List, Optional
from enum import Enum

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import numpy as np
import cv2

from ..config import settings
from ..modeling.predict import get_detector
from ..processing.overlay import overlay_banner
from ..processing.ocr_validator import ocr_validate_plate
from ..processing.plate_masker import mask_plate_with_image, load_mask_image
from ..processing.image_preprocessor import auto_preprocess_for_detection
from ..processing.quality_checker import verify_no_plate_leak
from ..processing.exif_handler import auto_orient_image


# 定数（日本の軽自動車の正方形ナンバーにも対応）
ASPECT_RATIO_MIN = 1.0  # 軽自動車の正方形ナンバー対応
ASPECT_RATIO_MAX = 2.5
FILL_COLOR = (255, 255, 255)
OCR_MIN_CONFIDENCE = 0.3  # OCR検証の最小信頼度

# マスク画像をロード
MASK_IMAGE = load_mask_image()


# レスポンスモデル
class HealthResponse(BaseModel):
    """ヘルスチェックレスポンス"""
    status: str = Field(example="healthy", description="サービス状態")
    model_loaded: bool = Field(example=True, description="YOLOモデルのロード状態")


class DetectionItem(BaseModel):
    """検出アイテム（/predict用）"""
    confidence: float = Field(example=0.95, description="検出信頼度（0.0〜1.0）")
    polygon: List[List[int]] = Field(
        example=[[100, 200], [300, 200], [300, 280], [100, 280]],
        description="マスク領域の4点座標 [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]"
    )
    ocr_text: Optional[str] = Field(
        default=None, 
        example="品川 500 あ 12-34",
        description="OCR読み取りテキスト（ocr_check=true時のみ）"
    )
    ocr_score: Optional[float] = Field(
        default=None,
        example=0.85,
        description="OCR信頼度（ocr_check=true時のみ）"
    )


class PredictResponse(BaseModel):
    """マスク処理レスポンス"""
    image: str = Field(description="マスク済み画像（Base64エンコード）")
    detections: List[DetectionItem] = Field(description="検出されたナンバープレートのリスト")
    count: int = Field(example=1, description="検出数")


class DetectItem(BaseModel):
    """検出アイテム（/detect用）"""
    bbox: List[float] = Field(
        example=[100.0, 200.0, 200.0, 80.0],
        description="バウンディングボックス [x, y, width, height]"
    )
    confidence: float = Field(example=0.95, description="検出信頼度（0.0〜1.0）")
    mask_points: List[List[float]] = Field(description="セグメンテーションポリゴンの座標リスト")


class DetectResponse(BaseModel):
    """検出結果レスポンス"""
    count: int = Field(example=1, description="検出数")
    detections: List[DetectItem] = Field(description="検出されたナンバープレートのリスト")


class OverlayMode(str, Enum):
    """
    オーバーレイモード
    
    - overlay: バナーを画像に重ねる（画像サイズ維持）
    - extend: 画像を拡張してバナーを追加
    - fit: 画像を縮小してバナーを追加（画像サイズ維持）
    """
    overlay = "overlay"
    extend = "extend"
    fit = "fit"


class OverlayPosition(str, Enum):
    """
    バナー位置
    
    - bottom: 画像の下部
    - top: 画像の上部
    """
    bottom = "bottom"
    top = "top"


class OverlayResponse(BaseModel):
    """オーバーレイ処理レスポンス"""
    image: str = Field(description="合成画像（Base64エンコード）")
    mode: str = Field(example="overlay", description="使用したオーバーレイモード")
    position: str = Field(example="bottom", description="バナー位置")
    opacity: float = Field(example=1.0, description="適用した透明度")
    plate_masked: bool = Field(example=False, description="ナンバーマスク処理の有無")
    plates_count: int = Field(example=0, description="マスクしたナンバープレート数")
    output_width: int = Field(example=640, description="出力画像の幅")
    output_height: int = Field(example=480, description="出力画像の高さ")


# アプリケーション
app = FastAPI(
    title="ナンバープレート検出API",
    description="""
## 概要
日本のナンバープレートを検出し、プライバシー保護のためにマスク（白塗り）するAPI。
YOLO11 Instance Segmentationモデルを使用。

## 主な機能

### 1. マスキング (`/predict`)
画像内のナンバープレートを検出して白色でマスク処理。

### 2. 検出のみ (`/detect`)  
マスクせずに検出結果（座標・信頼度）をJSONで返す。

### 3. バナーオーバーレイ (`/overlay`)
画像にバナーを合成。オプションでナンバーマスクも可能。

## 対応フォーマット
- 入力: JPEG, PNG, GIF, WebP
- 出力: Base64エンコードされたJPEG画像

## 注意事項
- 最大ファイルサイズ: 10MB
- 傾いた画像は `auto_rotate=true` で自動補正
""",
    version="0.1.0",
)


@app.get("/")
async def root():
    """API情報"""
    return {
        "name": "ナンバープレート検出API",
        "version": "0.1.0",
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """ヘルスチェック"""
    return {"status": "healthy", "model_loaded": True}


def validate_aspect_ratio(polygon: np.ndarray) -> bool:
    """アスペクト比を検証"""
    x, y, w, h = cv2.boundingRect(polygon)
    if h == 0 or w == 0:
        return False
    ratio = max(w, h) / min(w, h)
    return ASPECT_RATIO_MIN <= ratio <= ASPECT_RATIO_MAX


def polygon_to_quadrilateral(polygon: np.ndarray) -> np.ndarray:
    """ポリゴンを四角形に変換"""
    hull = cv2.convexHull(polygon)
    epsilon = 0.02 * cv2.arcLength(hull, True)
    approx = cv2.approxPolyDP(hull, epsilon, True)
    
    if len(approx) != 4:
        pts = hull.reshape(-1, 2)
        sum_pts = pts.sum(axis=1)
        diff_pts = np.diff(pts, axis=1).reshape(-1)
        
        tl = pts[np.argmin(sum_pts)]
        br = pts[np.argmax(sum_pts)]
        tr = pts[np.argmin(diff_pts)]
        bl = pts[np.argmax(diff_pts)]
        
        approx = np.array([tl, tr, br, bl], dtype=np.int32).reshape(-1, 1, 2)
    
    return approx


def rotate_back(img: np.ndarray, rotation_used: str) -> np.ndarray:
    """回転した画像を元の向きに戻す"""
    reverse_rotations = {
        "original": None,
        "90_cw": cv2.ROTATE_90_COUNTERCLOCKWISE,  # 90CW → 90CCWで戻す
        "90_ccw": cv2.ROTATE_90_CLOCKWISE,        # 90CCW → 90CWで戻す
        "180": cv2.ROTATE_180,                     # 180 → 180で戻す
    }
    
    reverse = reverse_rotations.get(rotation_used)
    if reverse is not None:
        return cv2.rotate(img, reverse)
    return img


def try_detect_with_rotations(detector, img: np.ndarray) -> tuple:
    """
    複数の回転で検出を試みる（batch_mask.pyと同じロジック）
    
    Returns:
        (best_detections, best_img, rotation_used)
    """
    rotations = [
        (None, "original"),
        (cv2.ROTATE_90_CLOCKWISE, "90_cw"),
        (cv2.ROTATE_90_COUNTERCLOCKWISE, "90_ccw"),
        (cv2.ROTATE_180, "180"),
    ]
    
    best_detections = []
    best_img = img
    best_conf = 0
    rotation_used = "original"
    
    for rotation, name in rotations:
        if rotation is not None:
            test_img = cv2.rotate(img, rotation)
        else:
            test_img = img
        
        detections_raw = detector.predict(test_img)
        
        # 有効な検出のみフィルタリング
        valid_detections = []
        max_conf = 0
        for det in detections_raw:
            polygon = det["mask"]
            if validate_aspect_ratio(polygon):
                valid_detections.append(det)
                max_conf = max(max_conf, det["confidence"])
        
        # 最高信頼度の回転を選択
        if max_conf > best_conf:
            best_conf = max_conf
            best_detections = valid_detections
            best_img = test_img
            rotation_used = name
    
    return best_detections, best_img, rotation_used


@app.post("/predict", response_model=PredictResponse)
async def predict(
    image: UploadFile = File(..., description="画像ファイル（JPEG/PNG/GIF/WebP）"),
    auto_rotate: bool = Form(True, description="傾いた画像の自動回転検出"),
    ocr_check: bool = Form(False, description="OCRでテキスト検証（処理が遅くなる）"),
    mask_mode: str = Form("fill", description="マスクモード: fill（白塗り）/ image（画像）"),
    low_light_fix: bool = Form(False, description="低照度画像の自動補正"),
    quality_check: bool = Form(False, description="マスク品質チェック（未マスク流出防止）"),
):
    """
    ナンバープレートを検出してマスク処理
    
    ## パラメータ詳細
    
    ### auto_rotate（推奨: true）
    - `true`: 画像を0°/90°/180°/270°回転して検出を試み、最も信頼度の高い結果を採用。
    - `false`: 回転検出なし。処理は高速。
    
    ### ocr_check（推奨: false）
    - `false`: 検出されたすべてのナンバープレートをマスク。
    - `true`: OCRでテキスト検証し、日本のナンバープレートパターンに一致する場合のみマスク。
    
    ### mask_mode
    - `fill`: 白色で塗りつぶし（デフォルト）
    - `image`: マスク画像を射影変換して合成
    
    ### low_light_fix
    - `true`: 低照度画像を自動補正して検出精度を向上
    - `false`: 補正なし（デフォルト）
    
    ### quality_check
    - `true`: マスク後の画像を検証し、ナンバーが残っていないかチェック
    - `false`: チェックなし（デフォルト）
    
    ## 使用例
    ```
    curl -X POST "/predict" -F "image=@car.jpg" -F "mask_mode=image"
    ```
    """
    if not image.content_type or image.content_type.split("/")[0] != "image":
        raise HTTPException(status_code=400, detail="画像ファイルが必要です")
    
    contents = await image.read()
    if len(contents) > settings.max_file_size_bytes:
        raise HTTPException(status_code=400, detail="ファイルサイズが大きすぎます")
    
    # EXIF向き補正を適用
    img, exif_info = auto_orient_image(contents)
    
    if img is None:
        raise HTTPException(status_code=400, detail="画像を読み込めません")
    
    # 低照度補正（オプション）
    if low_light_fix:
        img, preprocess_info = auto_preprocess_for_detection(img)
    
    detector = get_detector()
    
    if auto_rotate:
        # 回転検出を試みる
        valid_detections, best_img, rotation_used = try_detect_with_rotations(detector, img)
    else:
        # 通常検出のみ
        detections_raw = detector.predict(img)
        valid_detections = [d for d in detections_raw if validate_aspect_ratio(d["mask"])]
        best_img = img
        rotation_used = "original"
    
    detections = []
    result_img = best_img.copy()
    original_img = best_img.copy()  # 品質チェック用
    masked_regions = []  # 品質チェック用
    
    for det in valid_detections:
        polygon = det["mask"]
        
        # OCR検証（オプション）
        if ocr_check:
            is_valid, ocr_text, ocr_score = ocr_validate_plate(
                best_img, polygon, OCR_MIN_CONFIDENCE
            )
            if not is_valid:
                continue  # OCR検証に失敗したらスキップ
        
        quad = polygon_to_quadrilateral(polygon)
        
        # マスク処理
        if mask_mode == "image":
            # 射影変換でマスク画像を合成
            result_img = mask_plate_with_image(result_img, polygon, MASK_IMAGE, opacity=1.0)
        else:
            # 白塗り（デフォルト）
            cv2.fillPoly(result_img, [quad], color=FILL_COLOR)
        
        masked_regions.append(quad)
        
        quad_points = quad.reshape(-1, 2).tolist()
        detection_info = {
            "confidence": det["confidence"],
            "polygon": quad_points
        }
        
        # OCR結果を追加
        if ocr_check:
            detection_info["ocr_text"] = ocr_text
            detection_info["ocr_score"] = ocr_score
        
        detections.append(detection_info)
    
    # 品質チェック（オプション）
    quality_ok = True
    quality_message = "OK"
    if quality_check and masked_regions:
        quality_ok, quality_message = verify_no_plate_leak(result_img, masked_regions)
        if not quality_ok:
            # 品質チェックNG: 白塗りで強制マスク
            for quad in masked_regions:
                cv2.fillPoly(result_img, [quad], color=FILL_COLOR)
            quality_message = f"品質チェックNG、白塗りで再マスク: {quality_message}"
    
    # 元の向きに戻す
    result_img = rotate_back(result_img, rotation_used)
    
    _, buffer = cv2.imencode(".jpg", result_img)
    img_base64 = base64.b64encode(buffer).decode("utf-8")
    
    return JSONResponse(content={
        "image": img_base64,
        "detections": detections,
        "count": len(detections)
    })


@app.post("/detect", response_model=DetectResponse)
async def detect(image: UploadFile = File(..., description="画像ファイル（JPEG/PNG/GIF/WebP）")):
    """
    ナンバープレートを検出（マスクなし、座標のみ）
    
    ## 概要
    画像内のナンバープレートを検出し、座標情報をJSONで返す。
    マスク処理は行わないため、独自の後処理が必要な場合に使用。
    
    ## レスポンス
    - `count`: 検出数
    - `detections`: 検出リスト
      - `bbox`: バウンディングボックス [x, y, width, height]
      - `confidence`: 信頼度（0.0〜1.0）
      - `mask_points`: セグメンテーションポリゴン座標
    
    ## 注意
    - このAPIは回転検出（auto_rotate）非対応
    - 傾いた画像には `/predict` を使用してください
    """
    if not image.content_type or image.content_type.split("/")[0] != "image":
        raise HTTPException(status_code=400, detail="画像ファイルが必要です")
    
    contents = await image.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        raise HTTPException(status_code=400, detail="画像を読み込めません")
    
    detector = get_detector()
    detections = detector.predict(img)
    
    results = []
    for det in detections:
        results.append({
            "bbox": det["bbox"],
            "confidence": det["confidence"],
            "mask_points": det["mask"].tolist(),
        })
    
    return JSONResponse(content={
        "count": len(results),
        "detections": results,
    })


@app.post("/overlay", response_model=OverlayResponse)
async def overlay(
    image: UploadFile = File(..., description="画像ファイル（JPEG/PNG/GIF/WebP）"),
    mode: OverlayMode = Form(OverlayMode.overlay, description="オーバーレイモード"),
    position: OverlayPosition = Form(OverlayPosition.bottom, description="バナー位置"),
    opacity: float = Form(1.0, description="バナー透明度（0.0〜1.0）"),
    bg_color: str = Form("#FFFFFF", description="背景色（fitモード用、HEX形式）"),
    mask_plate: bool = Form(False, description="ナンバープレートをマスクする"),
    auto_rotate: bool = Form(True, description="傾いた画像の自動回転検出"),
):
    """
    画像にバナーをオーバーレイ合成
    
    ## モード詳細
    
    ### overlay（デフォルト）
    バナーを画像の上に重ねる。画像サイズは変わらない。
    バナーは画像幅に合わせて自動リサイズ。
    
    ### extend
    画像を拡張してバナーを追加。画像の高さが増える。
    元の画像は切り取られない。
    
    ### fit
    画像を縮小してバナーを追加。元のサイズを維持。
    余白は `bg_color` で塗りつぶし。
    
    ## パラメータ詳細
    
    | パラメータ | 説明 | 例 |
    |-----------|------|-----|
    | mode | overlay/extend/fit | overlay |
    | position | bottom/top | bottom |
    | opacity | 0.0（透明）〜1.0（不透明） | 0.8 |
    | bg_color | HEX色コード（fitモード用） | #000000 |
    | mask_plate | ナンバーマスク有無 | true |
    | auto_rotate | 回転検出（mask_plate時のみ） | true |
    
    ## レスポンス
    - `image`: 合成画像（Base64）
    - `mode`: 使用したモード
    - `opacity`: 適用した透明度
    - `plate_masked`: マスク処理の有無
    - `plates_count`: マスクしたナンバー数
    
    ## 使用例
    ```
    # バナーオーバーレイ + ナンバーマスク
    curl -X POST "/overlay" \\
      -F "image=@car.jpg" \\
      -F "mode=extend" \\
      -F "position=bottom" \\
      -F "opacity=0.9" \\
      -F "mask_plate=true"
    ```
    """
    if not image.content_type or image.content_type.split("/")[0] != "image":
        raise HTTPException(status_code=400, detail="画像ファイルが必要です")
    
    contents = await image.read()
    if len(contents) > settings.max_file_size_bytes:
        raise HTTPException(status_code=400, detail="ファイルサイズが大きすぎます")
    
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        raise HTTPException(status_code=400, detail="画像を読み込めません")
    
    # opacity範囲チェック
    opacity = max(0.0, min(1.0, opacity))
    
    plates_count = 0
    
    # マスク処理（オプション）
    if mask_plate:
        detector = get_detector()
        
        if auto_rotate:
            # 回転検出対応
            valid_detections, best_img, rotation_used = try_detect_with_rotations(detector, img)
        else:
            # 通常検出のみ
            detections_raw = detector.predict(img)
            valid_detections = [d for d in detections_raw if validate_aspect_ratio(d["mask"])]
            best_img = img
            rotation_used = "original"
        
        # マスク処理
        for det in valid_detections:
            polygon = det["mask"]
            quad = polygon_to_quadrilateral(polygon)
            cv2.fillPoly(best_img, [quad], color=FILL_COLOR)
            plates_count += 1
        
        # 元の向きに戻す
        img = rotate_back(best_img, rotation_used)
    
    # オーバーレイ処理
    result_img = overlay_banner(
        image=img,
        mode=mode.value,
        position=position.value,
        opacity=opacity,
        bg_color=bg_color,
    )
    
    _, buffer = cv2.imencode(".jpg", result_img)
    img_base64 = base64.b64encode(buffer).decode("utf-8")
    
    # 出力画像サイズ
    output_h, output_w = result_img.shape[:2]
    
    return JSONResponse(content={
        "image": img_base64,
        "mode": mode.value,
        "position": position.value,
        "opacity": opacity,
        "plate_masked": mask_plate,
        "plates_count": plates_count,
        "output_width": output_w,
        "output_height": output_h,
    })
