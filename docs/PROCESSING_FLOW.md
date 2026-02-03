# Image Processing Flow

## Overview

Script `fetch_today_images.py` xử lý ảnh xe từ database, detect biển số và mask/banner.

## Directory Structure

```
/mnt/cs1es3/webroot/upfile/{car_id_prefix}/{car_id_suffix}/
├── original_image.jpg          # File gốc (first file: banner only, no mask)
├── .backup/
│   └── original_image.jpg      # Backup của file gốc (KHÔNG BAO GIỜ bị ghi đè)
└── .detect/
    └── original_image.jpg      # File đã xử lý (masked) - cho các file không phải first
```

## Processing Rules

### 1. Backup Logic (KHÔNG THAY ĐỔI)
- Trước khi xử lý, backup file gốc vào `.backup/`
- Backup chỉ tạo MỘT LẦN - không bao giờ ghi đè
- Nếu backup đã tồn tại → restore từ backup trước khi xử lý

### 2. First File (is_first=True)
```
Input:  /upfile/1041/8430/xxx.jpg (file gốc)
Output: /upfile/1041/8430/xxx.jpg (GHI ĐÈ file gốc)

Processing:
- Detect biển số (YOLO)
- is_masking = FALSE (KHÔNG che biển số)
- add_banner = TRUE (có banner overlay)
- Ghi đè trực tiếp lên file gốc
```

### 3. Non-First Files (is_first=False)
```
Input:  /upfile/1041/8430/yyy.jpg (file gốc)
Output: /upfile/1041/8430/.detect/yyy.jpg (file mới trong .detect/)

Processing:
- Detect biển số (YOLO)
- is_masking = TRUE (có che biển số)
- add_banner = TRUE (có banner overlay)
- Lưu vào .detect/ folder
- File gốc KHÔNG BỊ THAY ĐỔI
```

## Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                        START PROCESSING                              │
└─────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│  1. Check backup exists in .backup/                                  │
│     - If NOT exists → Create backup (copy original to .backup/)     │
│     - If exists → Restore from backup (ensure clean original)       │
└─────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│  2. Load image and run detection (YOLO)                             │
└─────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
                    ┌──────────────┴──────────────┐
                    │      is_first == True?       │
                    └──────────────┬──────────────┘
                          │                │
                         YES              NO
                          │                │
                          ▼                ▼
┌─────────────────────────────┐  ┌─────────────────────────────┐
│  3a. FIRST FILE             │  │  3b. NON-FIRST FILE         │
│                             │  │                             │
│  - Banner overlay ONLY      │  │  - Apply masking            │
│  - NO masking               │  │  - Banner overlay           │
│  - Overwrite original       │  │  - Save to .detect/         │
│                             │  │  - Original unchanged       │
└─────────────────────────────┘  └─────────────────────────────┘
                          │                │
                          └────────┬───────┘
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│  4. Update tracking file                                            │
└─────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│                          END PROCESSING                              │
└─────────────────────────────────────────────────────────────────────┘
```

## Example

### Car 10418430 with 3 images:

```
Before processing:
/upfile/1041/8430/
├── 10418430001.jpg  (first - branch_no=1)
├── 10418430002.jpg  (branch_no=2)
└── 10418430003.jpg  (branch_no=3)

After processing:
/upfile/1041/8430/
├── 10418430001.jpg  ← Banner ONLY (no mask), overwritten
├── 10418430002.jpg  ← UNCHANGED (original)
├── 10418430003.jpg  ← UNCHANGED (original)
├── .backup/
│   ├── 10418430001.jpg  ← Backup of original
│   ├── 10418430002.jpg  ← Backup of original
│   └── 10418430003.jpg  ← Backup of original
└── .detect/
    ├── 10418430002.jpg  ← Masked version
    └── 10418430003.jpg  ← Masked version
```

## Configuration

Environment variables:
- `S3_MOUNT`: S3 mount point (e.g., `/mnt/cs1es3/webroot`)
- `BACKUP_S3_BUCKET`: S3 bucket for backup via boto3 (e.g., `cs1es3`)
- `BACKUP_S3_PREFIX`: Backup folder name (default: `.backup`)

## Scripts

| Script | Purpose |
|--------|---------|
| `fetch_today_images.py` | Main processing script |
| `restore_from_backup.py` | Restore originals from backup |
| `process_image_v2.py` | Image detection & masking logic |
