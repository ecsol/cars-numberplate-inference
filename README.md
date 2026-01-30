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

## ⚙️ Cấu hình

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_PATH` | `models/best.pt` | Đường dẫn model |
| `CONFIDENCE_THRESHOLD` | `0.1` | Ngưỡng confidence |
| `DEVICE` | `cpu` | Device: cpu/cuda |
| `MAX_FILE_SIZE_MB` | `10` | Max file size |

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
