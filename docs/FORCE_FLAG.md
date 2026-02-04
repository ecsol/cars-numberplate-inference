# --force & --force-overlay Documentation

## Overview

Script `fetch_today_images.py` có 2 flag đặc biệt để xử lý lại ảnh:

| Flag | Mục đích |
| ---- | -------- |
| `--force` | Tái tạo `.detect/` + overlay banner lên original (branch_no=1) |
| `--force-overlay` | Chỉ overlay banner lên original (branch_no=1), không tạo `.detect/` |

---

## So sánh chi tiết

| Đặc điểm | Normal Mode | `--force` | `--force-overlay` |
| -------- | ----------- | --------- | ----------------- |
| **Mục đích** | Xử lý ảnh mới | Tái tạo tất cả | Chỉ thêm banner |
| **Tạo backup** | ✅ Có (nếu chưa có) | ✅ Có (nếu chưa có) | ✅ Có (nếu chưa có) |
| **Tạo `.detect/`** | ✅ Có | ✅ Có (overwrite) | ❌ Không |
| **Masking** | ✅ Có (trong .detect/) | ✅ Có (trong .detect/) | ❌ Không |
| **Overlay banner** | ✅ branch_no=1 only | ✅ branch_no=1 only | ✅ branch_no=1 only |
| **Input cho .detect/** | `.backup` | `.backup` | N/A |
| **Input cho overlay** | `.backup` | Original hiện tại | Original hiện tại |
| **Skip nếu đã có** | ✅ Skip nếu .detect/ có | ❌ Không skip | ❌ Không skip |

---

# `--force` Flag

## Mục đích

Tái tạo lại `.detect/` và overlay banner lên original cho `branch_no=1`.

```bash
python fetch_today_images.py --force
python fetch_today_images.py --force --limit 50
python fetch_today_images.py --path /1554913G --force
```

## Flow xử lý

```
┌──────────────────────────────────────────────────────────────────┐
│                         --force MODE                              │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│  BƯỚC 1: TẠO BACKUP (nếu chưa có) ← BẮT BUỘC                     │
│    └─ copy original → .backup/                                   │
│                                                                   │
│  BƯỚC 2: TẠO .detect/ (từ BACKUP)                                │
│    ├─ Input: .backup (ảnh gốc sạch, detection chính xác)         │
│    ├─ Output: .detect/ (overwrite nếu đã có)                     │
│    ├─ branch_no=1: mask + banner                                 │
│    └─ branch_no≠1: mask ONLY (banner CẤM!)                       │
│                                                                   │
│  BƯỚC 3: OVERLAY BANNER (branch_no=1 only, SAU CÙNG)             │
│    ├─ Input: ORIGINAL HIỆN TẠI (không phải backup)               │
│    ├─ Output: ghi đè original                                    │
│    ├─ masking = FALSE (không che biển số)                        │
│    └─ banner = TRUE                                              │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
```

## Xử lý theo branch_no

| branch_no | .detect/ | Original |
| --------- | -------- | -------- |
| `= 1` | ✅ mask + banner (từ backup) | ✅ banner only (từ original hiện tại) |
| `≠ 1` | ✅ mask ONLY (từ backup) | ❌ Không đổi |

## Tại sao dùng backup cho .detect/?

| Lý do | Giải thích |
| ----- | ---------- |
| **Detection accuracy** | Backup là ảnh gốc sạch, không có banner → detection chính xác |
| **Tránh artifacts** | Original có thể đã có banner → detection bị ảnh hưởng |
| **Consistency** | Kết quả giống nhau mỗi lần chạy |

## Tại sao dùng original hiện tại cho overlay?

| Lý do | Giải thích |
| ----- | ---------- |
| **Backup được bảo toàn** | Backup giữ nguyên để restore nếu cần |
| **Cập nhật trạng thái hiện tại** | Overlay banner lên trạng thái hiện tại của ảnh |

---

# `--force-overlay` Flag

## Mục đích

**CHỈ** overlay banner lên original cho `branch_no=1`. Không làm gì khác.

```bash
python fetch_today_images.py --force-overlay
python fetch_today_images.py --force-overlay --limit 50
```

## Flow xử lý

```
┌──────────────────────────────────────────────────────────────────┐
│                     --force-overlay MODE                          │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│  BƯỚC 1: KIỂM TRA branch_no                                      │
│    └─ branch_no ≠ 1 → SKIP (không làm gì)                        │
│                                                                   │
│  BƯỚC 2: TẠO BACKUP (nếu chưa có) ← BẮT BUỘC                     │
│    └─ copy original → .backup/                                   │
│                                                                   │
│  BƯỚC 3: OVERLAY BANNER                                          │
│    ├─ Input: ORIGINAL HIỆN TẠI                                   │
│    ├─ Output: GHI ĐÈ ORIGINAL                                    │
│    ├─ masking = FALSE (KHÔNG che biển số)                        │
│    └─ banner = TRUE                                              │
│                                                                   │
│  ❌ KHÔNG tạo .detect/                                           │
│  ❌ KHÔNG masking                                                │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
```

## Xử lý theo branch_no

| branch_no | Hành động |
| --------- | --------- |
| `= 1` | ✅ Tạo backup (nếu chưa có) → Overlay banner lên original |
| `≠ 1` | ❌ SKIP hoàn toàn |

---

# Ví dụ thực tế

## Trước khi chạy

```
/upfile/1041/8430/
├── 10418430001.jpg      ← Original (có thể có banner cũ)
├── 10418430002.jpg      ← Original
├── 10418430003.jpg      ← Original
├── .backup/
│   ├── 10418430001.jpg  ← Backup gốc (clean, không banner)
│   ├── 10418430002.jpg  ← Backup gốc
│   └── 10418430003.jpg  ← Backup gốc
└── .detect/
    ├── 10418430001.jpg  ← Cũ (có thể sai)
    ├── 10418430002.jpg  ← Cũ
    └── 10418430003.jpg  ← Cũ
```

## Sau khi chạy `--force`

```
/upfile/1041/8430/
├── 10418430001.jpg      ← ✅ OVERLAY: banner (từ original cũ)
├── 10418430002.jpg      ← Không đổi
├── 10418430003.jpg      ← Không đổi
├── .backup/
│   ├── 10418430001.jpg  ← Không đổi (bảo toàn)
│   ├── 10418430002.jpg  ← Không đổi
│   └── 10418430003.jpg  ← Không đổi
└── .detect/
    ├── 10418430001.jpg  ← ✅ MỚI: mask + banner (từ backup)
    ├── 10418430002.jpg  ← ✅ MỚI: mask ONLY (từ backup)
    └── 10418430003.jpg  ← ✅ MỚI: mask ONLY (từ backup)
```

## Sau khi chạy `--force-overlay`

```
/upfile/1041/8430/
├── 10418430001.jpg      ← ✅ OVERLAY: banner (từ original cũ)
├── 10418430002.jpg      ← Không đổi (skip vì branch_no≠1)
├── 10418430003.jpg      ← Không đổi (skip vì branch_no≠1)
├── .backup/
│   └── ...              ← Tạo mới nếu chưa có
└── .detect/
    └── ...              ← KHÔNG THAY ĐỔI (--force-overlay không tạo .detect/)
```

---

# Quy tắc quan trọng

## ⚠️ QUY TẮC TUYỆT ĐỐI

1. **Banner trên `branch_no ≠ 1` là CẤM TUYỆT ĐỐI**
   - Chỉ ảnh đầu tiên (`branch_no=1`) mới được phép có banner
   - Các ảnh còn lại: mask only, KHÔNG banner

2. **Backup là BẮT BUỘC**
   - Cả `--force` và `--force-overlay` đều tự động tạo backup nếu chưa có
   - Backup được bảo toàn, không bao giờ ghi đè

3. **Overlay banner dùng ORIGINAL HIỆN TẠI**
   - Không dùng backup để overlay
   - Overlay lên trạng thái hiện tại của ảnh

4. **`.detect/` dùng BACKUP làm input**
   - Backup là ảnh sạch → detection chính xác
   - Không dùng original (có thể đã có banner)

---

# Khi nào dùng flag nào?

| Trường hợp | Dùng |
| ---------- | ---- |
| Cần tái tạo `.detect/` + overlay banner | `--force` |
| Chỉ cần thêm/cập nhật banner, không cần `.detect/` | `--force-overlay` |
| Xử lý ảnh mới | Normal mode (không cần flag) |
| Restore về ảnh gốc | `restore_from_backup.py` |

---

# Scripts liên quan

| Script | Mục đích |
| ------ | -------- |
| `fetch_today_images.py` | Main processing script |
| `restore_from_backup.py` | Restore original từ backup |
| `process_image_v2.py` | Detection & masking logic |

---

# Changelog

| Date | Version | Description |
| ---- | ------- | ----------- |
| 2026-02-04 | 2.0 | Cập nhật documentation đầy đủ cho --force và --force-overlay |
| 2026-02-04 | 1.3 | --force-overlay tạo backup bắt buộc |
| 2026-02-04 | 1.2 | --force overlay dùng original hiện tại |
| 2026-02-03 | 1.1 | --force tự động tạo backup |
| 2026-02-03 | 1.0 | Initial documentation |
