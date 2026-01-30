# 🚗 Plate Detection Service

> Production-ready API service cho ナンバープレート検出 (Inference Only)

## 📁 Cấu trúc thư mục

```
plate-detection-service/
├── src/
│   └── plate_detection/
│       ├── api/
│       │   ├── __init__.py
│       │   └── main.py          # FastAPI endpoints
│       ├── modeling/
│       │   ├── __init__.py
│       │   └── predict.py       # YOLO inference
│       ├── processing/
│       │   ├── __init__.py
│       │   ├── overlay.py       # Banner overlay
│       │   ├── plate_masker.py  # Masking logic
│       │   ├── exif_handler.py  # EXIF orientation
│       │   ├── image_preprocessor.py
│       │   ├── quality_checker.py
│       │   └── ocr_validator.py
│       └── config.py            # Settings
├── models/
│   └── best.pt                  # YOLO model (required)
├── assets/
│   ├── plate_mask.png           # Mask image
│   └── banner_sample.png        # Banner image
├── tests/
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .env.example
└── README.md
```

## 🚀 Deployment

### Option 1: Docker (推奨)

```bash
# 1. Copy model file vào thư mục models/
cp /path/to/best.pt models/

# 2. Build và chạy
docker-compose up -d

# 3. Kiểm tra
curl http://localhost:8000/health
```

### Option 2: Manual (Amazon Linux)

```bash
# 1. Cài đặt dependencies
sudo dnf install -y python3.10 python3.10-pip mesa-libGL

# 2. Tạo virtual environment
python3.10 -m venv venv
source venv/bin/activate

# 3. Cài đặt packages
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

# 4. Copy model
cp /path/to/best.pt models/

# 5. Cấu hình
cp .env.example .env
# Edit .env nếu cần

# 6. Chạy
uvicorn src.plate_detection.api.main:app --host 0.0.0.0 --port 8000
```

### Option 3: Systemd Service

```bash
# Tạo service file
sudo tee /etc/systemd/system/plate-api.service << 'EOF'
[Unit]
Description=Plate Detection API
After=network.target

[Service]
Type=simple
User=ec2-user
WorkingDirectory=/home/ec2-user/plate-detection-service
Environment="PATH=/home/ec2-user/plate-detection-service/venv/bin"
ExecStart=/home/ec2-user/plate-detection-service/venv/bin/uvicorn \
    src.plate_detection.api.main:app \
    --host 0.0.0.0 --port 8000 --workers 2
Restart=always

[Install]
WantedBy=multi-user.target
EOF

# Enable và start
sudo systemctl daemon-reload
sudo systemctl enable plate-api
sudo systemctl start plate-api
```

## 🛠️ Scripts Usage

### process_image.py - 画像処理スクリプト

ナンバープレートを検出してマスキング処理を行います。

```bash
# 仮想環境を有効化
source venv/bin/activate

# === 単一ファイル処理 ===
python scripts/process_image.py --input=car.jpg --output=result.jpg

# === フォルダ一括処理 ===
# outputフォルダは自動作成されます
python scripts/process_image.py --input=/path/to/images --output=/path/to/output

# === オプション ===
# バナーなし（マスキングのみ）
python scripts/process_image.py --input=folder --output=output --is-masking=false

# 信頼度閾値を変更
python scripts/process_image.py --input=car.jpg --output=result.jpg --confidence=0.3

# モデルを指定
python scripts/process_image.py --input=car.jpg --output=result.jpg --model=models/custom.pt
```

**オプション一覧:**
| Option | Default | Description |
|--------|---------|-------------|
| `--input` | (必須) | 入力画像またはフォルダ |
| `--output` | (必須) | 出力画像またはフォルダ |
| `--is-masking` | `true` | バナー追加 (true/false) |
| `--model` | `models/best.pt` | モデルファイルパス |
| `--confidence` | `0.1` | 検出信頼度 (0.0~1.0) |

---

### fetch_today_images.py - バッチ処理スクリプト

DBから画像を取得し、自動でマスキング処理を行います（crontab用）。

```bash
# 仮想環境を有効化
source venv/bin/activate

# === 基本実行 ===
# 今日の画像、最大10件
python scripts/fetch_today_images.py

# === 日付指定 ===
# 昨日の画像
python scripts/fetch_today_images.py --days-ago 1

# 1週間前の画像
python scripts/fetch_today_images.py --days-ago 7

# === 処理件数指定 ===
# 最大50件
python scripts/fetch_today_images.py --limit 50

# 全件処理（制限なし）
python scripts/fetch_today_images.py --limit 0

# === 組み合わせ ===
python scripts/fetch_today_images.py --days-ago 3 --limit 100
```

**オプション一覧:**
| Option | Default | Description |
|--------|---------|-------------|
| `--days-ago` | `0` | 何日前の画像を処理するか (0=今日) |
| `--limit` | `10` | 最大処理件数 (0=無制限) |

**crontab設定例:**
```bash
# 毎分実行
* * * * * /home/ec2-user/plate-detection-service/venv/bin/python /home/ec2-user/plate-detection-service/scripts/fetch_today_images.py >> /dev/null 2>&1
```

**ログファイル:**
```
logs/
├── process.log              # 処理ログ
└── tracking/
    └── processed_20260130.json  # 日次トラッキング
```

---

## 📡 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API info |
| `/health` | GET | Health check |
| `/predict` | POST | Detect & mask plates |
| `/detect` | POST | Detect only (no mask) |
| `/overlay` | POST | Add banner overlay |

### Ví dụ sử dụng

```bash
# Health check
curl http://localhost:8000/health

# Detect và mask
curl -X POST "http://localhost:8000/predict" \
  -F "image=@car.jpg" \
  -F "auto_rotate=true"

# Detect only
curl -X POST "http://localhost:8000/detect" \
  -F "image=@car.jpg"

# Overlay banner
curl -X POST "http://localhost:8000/overlay" \
  -F "image=@car.jpg" \
  -F "mode=extend" \
  -F "mask_plate=true"
```

## ⚙️ Cấu hình (.env)

```bash
cp .env.example .env
vi .env
```

### API設定
| Variable | Default | Description |
|----------|---------|-------------|
| `API_HOST` | `0.0.0.0` | APIホスト |
| `API_PORT` | `8000` | APIポート |
| `DEBUG` | `false` | デバッグモード |

### モデル設定
| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_PATH` | `models/best.pt` | モデルファイルパス |
| `CONFIDENCE_THRESHOLD` | `0.1` | 検出信頼度 (0.0~1.0) |
| `DEVICE` | `cpu` | デバイス: cpu / cuda / mps |
| `MAX_FILE_SIZE_MB` | `10` | 最大ファイルサイズ |

### データベース設定 (fetch_today_images.py用)
| Variable | Default | Description |
|----------|---------|-------------|
| `DB_HOST` | - | DBホスト |
| `DB_NAME` | `cartrading` | DB名 |
| `DB_USER` | - | DBユーザー |
| `DB_PASSWORD` | - | DBパスワード |

### S3設定
| Variable | Default | Description |
|----------|---------|-------------|
| `S3_MOUNT` | - | S3マウントパス |

### バックアップ設定
| Variable | Default | Description |
|----------|---------|-------------|
| `BACKUP_MODE` | `local` | バックアップ先: local / s3 |
| `BACKUP_DIR` | `/home/ec2-user/backup` | ローカルバックアップ先 |

### ログ設定
| Variable | Default | Description |
|----------|---------|-------------|
| `LOG_DIR` | `logs` | ログディレクトリ（相対/絶対パス）|

## 📦 Files cần thiết

1. **Model file** (`models/best.pt`) - **BẮT BUỘC**
2. `assets/plate_mask.png` - Mask image (optional)
3. `assets/banner_sample.png` - Banner (optional)

## 🔍 Troubleshooting

### Lỗi: Model not found
```bash
# Kiểm tra model file
ls -la models/best.pt
```

### Lỗi: libGL not found
```bash
sudo dnf install -y mesa-libGL
```

### Kiểm tra logs
```bash
# Docker
docker-compose logs -f

# Systemd
sudo journalctl -u plate-api -f
```
