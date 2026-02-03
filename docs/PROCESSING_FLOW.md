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

### 0. First File Determination (Cách xác định ảnh đầu tiên)

**First file được xác định từ DATABASE:**

```sql
SELECT ... FROM upload_files
WHERE ...
ORDER BY 
    COALESCE(inspresultdata_cd, car_cd::text),  -- Group by car
    branch_no ASC                                -- Sort by branch_no
```

**Logic:**
1. Query DB lấy tất cả files của ngày
2. Nhóm theo xe (`car_cd` hoặc `inspresultdata_cd`)
3. **Chỉ file có `branch_no = 1` mới là First file**

**Code:**
```python
# branch_no == 1 のみ first file として扱う
is_first = file_info["branch_no"] == 1
```

**Ví dụ:**
| File    | branch_no | is_first |
| ------- | --------- | -------- |
| 001.jpg | 1         | TRUE     |
| 002.jpg | 2         | FALSE    |
| 003.jpg | 3         | FALSE    |
| 004.jpg | NULL      | FALSE    |

**Lưu ý:** Nếu xe không có file `branch_no = 1`, sẽ không có first file → tất cả chỉ output vào `.detect/`

---

### 1. Backup Logic (KHÔNG THAY ĐỔI)
- Trước khi xử lý, backup file gốc vào `.backup/`
- Backup chỉ tạo MỘT LẦN - không bao giờ ghi đè
- Nếu backup đã tồn tại → restore từ backup trước khi xử lý

### 2. First File (is_first=True) - XỬ LÝ 2 LẦN
```
Input:  /upfile/1041/8430/xxx.jpg (file gốc)

Output 1: /upfile/1041/8430/.detect/xxx.jpg
  - is_masking = TRUE (có che biển số)
  - add_banner = TRUE (có banner overlay)

Output 2: /upfile/1041/8430/xxx.jpg (GHI ĐÈ file gốc)
  - is_masking = FALSE (KHÔNG che biển số)
  - add_banner = TRUE (có banner overlay)
  - Dùng để hiển thị trên website (không che biển số)
```

### 3. Non-First Files (is_first=False)
```
Input:  /upfile/1041/8430/yyy.jpg (file gốc)
Output: /upfile/1041/8430/.detect/yyy.jpg

Processing:
- Detect biển số (YOLO)
- is_masking = TRUE (có che biển số)
- add_banner = TRUE (có banner overlay)
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
├── 10418430001.jpg  (first - branch_no=1, determined by DB)
├── 10418430002.jpg  (branch_no=2)
└── 10418430003.jpg  (branch_no=3)

After processing:
/upfile/1041/8430/
├── 10418430001.jpg  ← Banner ONLY (no mask) - for website display
├── 10418430002.jpg  ← UNCHANGED (original)
├── 10418430003.jpg  ← UNCHANGED (original)
├── .backup/
│   ├── 10418430001.jpg  ← Backup of original (clean)
│   ├── 10418430002.jpg  ← Backup of original (clean)
│   └── 10418430003.jpg  ← Backup of original (clean)
└── .detect/
    ├── 10418430001.jpg  ← Masked + Banner (full processing)
    ├── 10418430002.jpg  ← Masked + Banner
    └── 10418430003.jpg  ← Masked + Banner
```

**Giải thích:**
- **Website hiển thị**: Dùng `10418430001.jpg` (banner only, không che biển số) - ảnh đại diện
- **Download/Export**: Dùng `.detect/` folder (tất cả đã được mask)

## Configuration

Environment variables:
- `S3_MOUNT`: S3 mount point (e.g., `/mnt/cs1es3/webroot`)
- `BACKUP_S3_BUCKET`: S3 bucket for backup via boto3 (e.g., `cs1es3`)
- `BACKUP_S3_PREFIX`: Backup folder name (default: `.backup`)

---

## Tracking File Design

### File Location
```
{LOG_DIR}/tracking/processed_YYYYMMDD.json
```
Ví dụ: `logs/tracking/processed_20260203.json`

### File Structure
```json
{
  "date": "2026-02-03",
  "created_at": "2026-02-03T00:00:00",
  "last_processed_time": "2026-02-03T15:30:00",
  "processed": {
    "12345": {
      "file_id": 12345,
      "path": "/upfile/1041/8430/10418430001.jpg",
      "branch_no": 1,
      "processed_at": "2026-02-03 10:30:00",
      "status": "success",
      "detections": 2,
      "is_first": true
    },
    "12346": {
      "file_id": 12346,
      "path": "/upfile/1041/8430/10418430002.jpg",
      "branch_no": 2,
      "processed_at": "2026-02-03 10:30:05",
      "status": "success",
      "detections": 1,
      "is_first": false
    },
    "12347": {
      "file_id": 12347,
      "path": "/upfile/1041/8430/10418430003.jpg",
      "branch_no": 3,
      "processed_at": "2026-02-03 10:30:10",
      "status": "error",
      "detections": 0,
      "is_first": false,
      "error": "file_not_found"
    }
  }
}
```

### Field Definitions

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `date` | string | Yes | Ngày xử lý (ISO format) |
| `created_at` | string | Yes | Thời gian tạo file tracking |
| `last_processed_time` | string | No | Thời gian xử lý cuối (dùng cho incremental fetch) |
| `processed` | object | Yes | Map của file_id -> record |

### Record Fields

| Field | Type | Required | Description | Used by Restore |
|-------|------|----------|-------------|-----------------|
| `file_id` | int | Yes | ID trong database | No |
| `path` | string | Yes | Đường dẫn relative (e.g., `/upfile/1041/8430/xxx.jpg`) | **YES** - để xác định file cần restore |
| `branch_no` | int/null | Yes | Số thứ tự ảnh trong xe | No (debug only) |
| `processed_at` | string | Yes | Thời gian xử lý | No |
| `status` | string | Yes | `success` / `error` / `skip` | **YES** - để filter |
| `detections` | int | No | Số biển số phát hiện | No |
| `is_first` | bool | Yes | Có phải ảnh đầu tiên không | No |
| `error` | string | No | Lý do lỗi (nếu status=error) | No |

### Restore Script Usage

`restore_from_backup.py` sử dụng tracking file để:

1. **Lấy danh sách files cần restore** từ `processed` object
2. **Filter theo status**: `--status success` / `--status error` / `--status all`
3. **Filter theo car_id**: Extract từ `path` field
4. **Xác định backup path**: Từ `path` field → tính ra `.backup/` location

```python
# Restore script chỉ dùng 2 fields:
path = record.get("path", "")      # Required
status = record.get("status", "")  # Optional filter
```

### Rules

1. **Mỗi ngày có 1 tracking file riêng** - không ghi đè ngày khác
2. **file_id là unique key** - mỗi file chỉ có 1 record
3. **Không xóa records** - chỉ thêm mới hoặc update
4. **path field là critical** - restore script phụ thuộc vào field này
5. **last_processed_time** - dùng cho incremental DB fetch, giảm load

---

## Scripts

| Script                   | Purpose                         |
| ------------------------ | ------------------------------- |
| `fetch_today_images.py`  | Main processing script          |
| `restore_from_backup.py` | Restore originals from backup   |
| `process_image_v2.py`    | Image detection & masking logic |
