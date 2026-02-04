# --force Flag Documentation

## Overview

Flag `--force` được sử dụng để **tái tạo lại thư mục `.detect/`** và **cập nhật original với banner-only** (cho `branch_no=1`).

```bash
python fetch_today_images.py --force
```

## Mục đích sử dụng

| Use Case          | Giải thích                                              |
| ----------------- | ------------------------------------------------------- |
| Sửa lỗi banner    | Khi `.detect/` của `branch_no != 1` bị thêm banner nhầm |
| Cập nhật model    | Sau khi train model mới, cần re-process tất cả          |
| Debug/Test        | Kiểm tra kết quả detection mà không ảnh hưởng original  |
| Fix detection lỗi | Khi model cũ detect sai, cần chạy lại với model mới     |

---

## So sánh với các mode khác

| Đặc điểm                        | Normal Mode                | `--force` Mode               | `--force-overlay` Mode   |
| ------------------------------- | -------------------------- | ---------------------------- | ------------------------ |
| **Mục đích**                    | Xử lý ảnh mới              | Tái tạo `.detect/` + original | Thêm banner vào original |
| **Kiểm tra `.detect/` tồn tại** | Skip nếu đã có             | **Overwrite**                | N/A                      |
| **Tạo backup mới**              | Có (nếu chưa có)           | **Skip**                     | Skip                     |
| **Thay đổi original (branch_no=1)** | banner only            | **banner only** ✅           | banner only              |
| **Output**                      | `.detect/` + original      | `.detect/` + original        | original only            |
| **Đầu vào**                     | `.backup`                  | `.backup`                    | original hiện tại        |

---

## Processing Rules

### Quy tắc xử lý theo branch_no

```
┌─────────────────────────────────────────────────────────────────────┐
│                         --force MODE                                 │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  branch_no = 1:                                                      │
│    Input:  .backup/xxx.jpg                                          │
│    Output 1: .detect/xxx.jpg                                        │
│      - is_masking = TRUE  ✅                                        │
│      - add_banner = TRUE  ✅                                        │
│    Output 2: Original (overwrite)                                   │
│      - is_masking = FALSE ✅ (không che biển số)                    │
│      - add_banner = TRUE  ✅                                        │
│                                                                      │
│  branch_no != 1:                                                     │
│    Input:  .backup/yyy.jpg                                          │
│    Output: .detect/yyy.jpg                                          │
│    Processing:                                                       │
│      - is_masking = TRUE  ✅                                        │
│      - add_banner = FALSE ⛔ (TUYỆT ĐỐI CẤM!)                       │
│    Original: KHÔNG THAY ĐỔI                                         │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### Bảng tóm tắt

| branch_no | Input     | .detect/ Output        | .detect/ Masking | .detect/ Banner | Original Output      |
| --------- | --------- | ---------------------- | ---------------- | --------------- | -------------------- |
| `= 1`     | `.backup` | overwrite              | ✅ Có            | ✅ Có           | ✅ banner only (overwrite) |
| `!= 1`    | `.backup` | overwrite              | ✅ Có            | ⛔ **CẤM**      | ❌ Không đổi         |

---

## Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                    START: --force flag called                        │
└─────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│  1. SKIP backup creation                                            │
│     - Không tạo backup mới                                          │
│     - Sử dụng backup hiện có (nếu có)                               │
└─────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
                    ┌──────────────┴──────────────┐
                    │      branch_no == 1?         │
                    └──────────────┬──────────────┘
                          │                │
                        YES               NO
                          │                │
                          ▼                ▼
┌─────────────────────────────────┐  ┌─────────────────────────────────────┐
│  2a. FIRST FILE (branch_no=1)   │  │  2b. NON-FIRST FILE (branch_no!=1)  │
│                                 │  │                                     │
│  Input: .backup/xxx.jpg         │  │  Input: .backup/yyy.jpg             │
│                                 │  │                                     │
│  Output 1: .detect/xxx.jpg      │  │  Output: .detect/yyy.jpg            │
│    - mask = TRUE                │  │    - mask = TRUE                    │
│    - banner = TRUE              │  │    - banner = FALSE ⛔              │
│                                 │  │                                     │
│  Output 2: Original (overwrite) │  │  Original: KHÔNG THAY ĐỔI          │
│    - mask = FALSE               │  │                                     │
│    - banner = TRUE ✅           │  │                                     │
└─────────────────────────────────┘  └─────────────────────────────────────┘
                          │                │
                          └────────┬───────┘
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│  3. Kết thúc xử lý                                                  │
│     - branch_no=1: Original được ghi đè với banner-only             │
│     - branch_no!=1: Original giữ nguyên                             │
└─────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│                              END                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Input Source Logic

### Thứ tự xử lý input (--force mode)

```python
# --force mode: Tự động tạo backup nếu chưa có, sau đó dùng backup làm input

if BACKUP_S3_BUCKET:
    backup_s3_key = f"webroot/{dir_part}/.backup/{file_name}"
    
    # Bước 1: Tạo backup nếu chưa có
    if not s3_backup_exists(backup_s3_key):
        s3_upload_backup(full_path, backup_s3_key)
        logger.debug("--force: バックアップなし、作成")
    
    # Bước 2: Download backup làm input
    input_path = download_from_s3(backup_s3_key)
else:
    backup_path = os.path.join(BACKUP_DIR, relative_path)
    
    # Bước 1: Tạo backup nếu chưa có
    if not os.path.exists(backup_path):
        shutil.copy(full_path, backup_path)
        logger.debug("--force: バックアップ作成")
    
    # Bước 2: Dùng backup làm input
    input_path = backup_path
```

### Tại sao dùng .backup làm input?

| Lý do                  | Giải thích                                                        |
| ---------------------- | ----------------------------------------------------------------- |
| **Detection accuracy** | File `.backup` là ảnh gốc chưa qua xử lý, detection chính xác hơn |
| **Tránh artifacts**    | Nếu dùng original đã có banner → detection có thể bị ảnh hưởng    |
| **Consistency**        | Đảm bảo kết quả giống nhau mỗi lần chạy                           |
| **Auto-create**        | `--force` tự động tạo backup nếu chưa có                          |

---

## Code Reference

### Argument Definition

```python
# Line 1271-1275
parser.add_argument(
    "--force",
    action="store_true",
    help=".detect/が存在しても強制的に再処理",
)
```

### Main Processing Logic

```python
# Line 910-995 in backup_and_process()
if force:
    # .detect/ は常にマスクあり
    # branch_no=1: マスク+バナー
    # branch_no!=1: マスクのみ（バナー【絶対禁止】）
    use_masking = True  # .detect/ は常にマスクあり
    use_banner = is_first_image  # branch_no=1 のみバナー
    
    # ... download from .backup ...
    
    result = process_image(
        input_path=temp_input_path,
        output_path=temp_detect_path,
        seg_model=seg_model,
        pose_model=pose_model,
        mask_image=mask_image,
        is_masking=True,      # .detect/は常にマスクあり
        add_banner=use_banner,  # branch_no=1のみバナー【それ以外は絶対禁止】
    )
```

---

## Usage Examples

### 1. Re-process tất cả ảnh hôm nay

```bash
python fetch_today_images.py --force --limit 50
```

### 2. Re-process thư mục cụ thể

```bash
python fetch_today_images.py --path /1554913G --force
```

### 3. Re-process ngày cụ thể

```bash
python fetch_today_images.py --date 2026-02-01 --force
```

### 4. Re-process với limit cao

```bash
python fetch_today_images.py --force --limit 500 --days-ago 1
```

---

## Example Scenario

### Trước khi chạy --force

```
/upfile/1041/8430/
├── 10418430001.jpg      ← Original (có banner - từ normal mode)
├── 10418430002.jpg      ← Original (không đổi)
├── 10418430003.jpg      ← Original (không đổi)
├── .backup/
│   ├── 10418430001.jpg  ← Backup gốc (clean)
│   ├── 10418430002.jpg  ← Backup gốc (clean)
│   └── 10418430003.jpg  ← Backup gốc (clean)
└── .detect/
    ├── 10418430001.jpg  ← ❌ CŨ: có thể sai (model cũ hoặc banner lỗi)
    ├── 10418430002.jpg  ← ❌ CŨ: có thể sai
    └── 10418430003.jpg  ← ❌ CŨ: có thể sai
```

### Sau khi chạy --force

```
/upfile/1041/8430/
├── 10418430001.jpg      ← ✅ CẬP NHẬT: banner only (branch_no=1)
├── 10418430002.jpg      ← KHÔNG ĐỔI
├── 10418430003.jpg      ← KHÔNG ĐỔI
├── .backup/
│   ├── 10418430001.jpg  ← KHÔNG ĐỔI
│   ├── 10418430002.jpg  ← KHÔNG ĐỔI
│   └── 10418430003.jpg  ← KHÔNG ĐỔI
└── .detect/
    ├── 10418430001.jpg  ← ✅ MỚI: mask + banner (branch_no=1)
    ├── 10418430002.jpg  ← ✅ MỚI: mask ONLY (branch_no=2)
    └── 10418430003.jpg  ← ✅ MỚI: mask ONLY (branch_no=3)
```

---

## Important Notes

### ⚠️ Quy tắc TUYỆT ĐỐI

1. **Banner trên `branch_no != 1` là CẤM TUYỆT ĐỐI**
   - Chỉ ảnh đầu tiên (branch_no=1) mới được phép có banner
   - Các ảnh còn lại chỉ được mask, KHÔNG có banner

2. **`--force` xử lý cả `.detect/` và original (cho branch_no=1)**
   - `.detect/`: tái tạo với mask + banner (branch_no=1) hoặc mask only (branch_no!=1)
   - Original: ghi đè với banner-only cho branch_no=1, không đổi cho các ảnh khác

3. **Input luôn từ .backup**
   - Đảm bảo detection từ ảnh sạch
   - Nếu không có .backup → dùng original (có warning)

### 🔄 Khi nào KHÔNG nên dùng --force

| Trường hợp                  | Lý do                   | Giải pháp                     |
| --------------------------- | ----------------------- | ----------------------------- |
| Muốn restore về ảnh gốc     | `--force` không restore | Dùng `restore_from_backup.py` |
| File mới chưa có `.detect/` | Không cần force         | Chạy normal mode              |

---

## Related Scripts

| Script                   | Mục đích            | Liên quan đến --force                     |
| ------------------------ | ------------------- | ----------------------------------------- |
| `fetch_today_images.py`  | Main processing     | Chứa --force flag                         |
| `restore_from_backup.py` | Restore từ backup   | Dùng trước --force nếu cần reset original |
| `process_image_v2.py`    | Detection & masking | Được gọi bởi --force                      |

---

## Changelog

| Date       | Version | Description                                                  |
| ---------- | ------- | ------------------------------------------------------------ |
| 2026-02-03 | 1.1     | `--force` giờ cũng tạo banner-only cho original (branch_no=1) |
| 2026-02-03 | 1.0     | Initial documentation                                        |
