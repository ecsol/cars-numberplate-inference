# Hướng dẫn sử dụng fetch_today_images.py

## Mục đích

Script tự động xử lý ảnh xe:
- Phát hiện và che biển số xe (masking)
- Thêm banner vào ảnh đầu tiên (branch_no=1)
- Lưu backup và ảnh đã xử lý

---

## Cách chạy cơ bản

```bash
# Vào thư mục inference
cd /home/ec2-user/plate-detection-service

# Chạy xử lý ảnh hôm nay
./venv/bin/python scripts/fetch_today_images.py
```

---

## Tất cả Options

### Filter Options (Chỉ lọc dữ liệu, không thay đổi logic)

| Option | Mô tả | Ví dụ |
|--------|-------|-------|
| `--days-ago N` | Xử lý ảnh của N ngày trước | `--days-ago 1` (hôm qua) |
| `--date YYYY-MM-DD` | Xử lý ngày cụ thể | `--date 2026-02-01` |
| `--limit N` | Giới hạn số xe xử lý | `--limit 50` |
| `--path FOLDER` | Xử lý 1 folder cụ thể | `--path 1041/0765` |

### Logic Options (Thay đổi cách xử lý)

| Option | Mô tả | Khi nào dùng |
|--------|-------|--------------|
| `--force` | Xử lý lại TẤT CẢ ảnh | Khi cần tái tạo `.detect/` hoặc banner |
| `--force-overlay` | Chỉ thêm banner, không tạo `.detect/` | Khi chỉ cần cập nhật banner |

---

## Ví dụ thực tế

### 1. Xử lý ảnh mới (hàng ngày)

```bash
# Xử lý tối đa 10 xe mới nhất hôm nay
./venv/bin/python scripts/fetch_today_images.py --limit 10
```

### 2. Xử lý lại 1 folder cụ thể

```bash
# Khi biết folder có vấn đề
./venv/bin/python scripts/fetch_today_images.py --path 1041/0765 --force
```

### 3. Xử lý lại tất cả ảnh hôm nay

```bash
# Tái tạo lại .detect/ và banner cho tất cả
./venv/bin/python scripts/fetch_today_images.py --force --limit 100
```

### 4. Xử lý ảnh hôm qua

```bash
./venv/bin/python scripts/fetch_today_images.py --days-ago 1 --limit 50
```

### 5. Xử lý ngày cụ thể

```bash
./venv/bin/python scripts/fetch_today_images.py --date 2026-02-01 --limit 50
```

### 6. Chỉ cập nhật banner (không tạo lại .detect/)

```bash
./venv/bin/python scripts/fetch_today_images.py --force-overlay --limit 50
```

---

## Cron Job (Tự động)

```bash
# Chạy mỗi phút, xử lý 5 xe mới
* * * * * flock -n /tmp/fetch_today.lock /home/ec2-user/plate-detection-service/venv/bin/python /home/ec2-user/plate-detection-service/scripts/fetch_today_images.py --limit 5 >> /home/ec2-user/plate-detection-service/logs/cron.log 2>&1
```

**Giải thích:**
- `flock -n /tmp/fetch_today.lock`: Tránh chạy trùng nếu job trước chưa xong
- `--limit 5`: Xử lý tối đa 5 xe mỗi lần
- `>> .../cron.log 2>&1`: Ghi log vào file

---

## Cấu trúc thư mục output

```
/upfile/1041/0765/
├── 10410765001.jpg      ← Original (có banner nếu branch_no=1)
├── 10410765002.jpg      ← Original (không thay đổi)
├── 10410765030.jpg      ← Original (Photo30 - meter panel)
├── .backup/
│   ├── 10410765001.jpg  ← Backup gốc (không banner, không mask)
│   ├── 10410765002.jpg
│   └── 10410765030.jpg
└── .detect/
    ├── 10410765001.jpg  ← Masked + Banner
    ├── 10410765002.jpg  ← Masked only
    └── 10410765030.jpg  ← Copy trực tiếp (không detect)
```

---

## Ảnh đặc biệt (Skip Detection)

Các branch_no sau **không chạy qua model detection** (chỉ copy):

| branch_no | Tên | Lý do |
|-----------|-----|-------|
| 30 | メーターパネル (Meter Panel) | Không phải ảnh xe |
| 31 | コーションプレート (Caution Plate) | Không phải ảnh xe |
| 32 | 車検証 (Vehicle Inspection) | Không phải ảnh xe |

Các ảnh này vẫn được:
- ✅ Copy vào `.backup/`
- ✅ Copy vào `.detect/`
- ❌ KHÔNG chạy detection/masking

---

## Xử lý lỗi thường gặp

### 1. Option sai tên

```bash
$ ./venv/bin/python scripts/fetch_today_images.py --limt 5
[ERROR] 不正なオプション: ['--limt']
[ERROR] 有効なオプション: --days-ago, --date, --limit, --path, --force, --force-overlay
```

**Sửa:** Kiểm tra lại tên option (`--limit` không phải `--limt`)

### 2. Ảnh bị miss không được xử lý

```bash
# Dùng --force để xử lý lại tất cả
./venv/bin/python scripts/fetch_today_images.py --force --limit 100
```

### 3. Muốn xử lý lại 1 xe cụ thể

```bash
# Tìm folder của xe (ví dụ: 1041/0765)
./venv/bin/python scripts/fetch_today_images.py --path 1041/0765 --force
```

### 4. Chỉ muốn cập nhật banner (không tạo lại .detect/)

```bash
./venv/bin/python scripts/fetch_today_images.py --force-overlay --limit 50
```

---

## So sánh --force vs --force-overlay

| Đặc điểm | `--force` | `--force-overlay` |
|----------|-----------|-------------------|
| Tạo `.backup/` | ✅ Có | ✅ Có |
| Tạo `.detect/` | ✅ Có (overwrite) | ❌ Không |
| Masking | ✅ Có | ❌ Không |
| Banner trên `.detect/` | ✅ branch_no=1 | ❌ Không |
| Banner trên original | ✅ branch_no=1 | ✅ branch_no=1 |
| Xử lý branch_no≠1 | ✅ Có | ❌ Skip |

---

## Khôi phục ảnh gốc

Nếu cần khôi phục ảnh về trạng thái ban đầu:

```bash
./venv/bin/python scripts/restore_from_backup.py --path 1041/0765
```

---

## Xem log

```bash
# Log mới nhất
tail -f /home/ec2-user/plate-detection-service/logs/process.log

# Log cron
tail -f /home/ec2-user/plate-detection-service/logs/cron.log
```

---

## Liên hệ hỗ trợ

Nếu gặp vấn đề không giải quyết được, liên hệ:
- Check code: `/home/ec2-user/plate-detection-service/scripts/fetch_today_images.py`
- Docs: `/home/ec2-user/plate-detection-service/docs/`

---

*Cập nhật: 2026-02-04*
