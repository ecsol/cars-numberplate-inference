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

## Scripts

| Script                   | Purpose                         |
| ------------------------ | ------------------------------- |
| `fetch_today_images.py`  | Main processing script          |
| `restore_from_backup.py` | Restore originals from backup   |
| `process_image_v2.py`    | Image detection & masking logic |
