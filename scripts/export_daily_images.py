#!/usr/bin/env python3
"""
指定日の車両画像情報をJSON形式でエクスポート

Usage:
    python export_daily_images.py                    # 今日の画像
    python export_daily_images.py --date 2026-02-03  # 特定日
    python export_daily_images.py --days-ago 1       # 昨日
    python export_daily_images.py --output result.json  # 出力ファイル指定
    python export_daily_images.py --chatwork         # Chatworkにファイル送信
"""

import argparse
import json
import os
import sys
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict

import psycopg2
import requests

# スクリプトディレクトリ
SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_DIR))


def load_env_file():
    """環境変数ファイルを読み込み"""
    env_file = PROJECT_DIR / ".env"
    if env_file.exists():
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    if key not in os.environ:
                        os.environ[key] = value


load_env_file()

# ======================
# 設定
# ======================
DB_CONFIG = {
    "host": os.getenv("DB_HOST", ""),
    "database": os.getenv("DB_NAME", "cartrading"),
    "user": os.getenv("DB_USER", ""),
    "password": os.getenv("DB_PASSWORD", ""),
}

IMAGE_BASE_URL = os.getenv("IMAGE_BASE_URL", "https://www.autobacs-cars-system.com")

# Chatwork設定
CHATWORK_API_KEY = os.getenv("CHATWORK_API_KEY", "")
CHATWORK_ROOM_ID = os.getenv("CHATWORK_ROOM_ID", "")


def send_file_to_chatwork(file_path: str, message: str = "") -> bool:
    """
    Chatworkにファイルをアップロードしてメッセージ送信
    
    Args:
        file_path: 送信するファイルのパス
        message: 添付メッセージ
    
    Returns:
        bool: 成功したらTrue
    """
    if not CHATWORK_API_KEY or not CHATWORK_ROOM_ID:
        print("[WARN] CHATWORK_API_KEY または CHATWORK_ROOM_ID が未設定")
        return False
    
    url = f"https://api.chatwork.com/v2/rooms/{CHATWORK_ROOM_ID}/files"
    headers = {"X-ChatWorkToken": CHATWORK_API_KEY}
    
    try:
        with open(file_path, "rb") as f:
            files = {"file": (os.path.basename(file_path), f, "application/json")}
            data = {"message": message} if message else {}
            
            response = requests.post(url, headers=headers, files=files, data=data)
            
            if response.status_code == 200:
                print(f"[OK] Chatworkにファイル送信完了: {os.path.basename(file_path)}")
                return True
            else:
                print(f"[ERROR] Chatwork送信失敗: {response.status_code} - {response.text}")
                return False
    except Exception as e:
        print(f"[ERROR] Chatwork送信エラー: {e}")
        return False


def get_images_by_date(target_date) -> list:
    """
    指定日に作成/更新された画像を取得

    Returns:
        list: [(id, car_cd, inspresultdata_cd, branch_no, save_file_name, created, modified), ...]
    """
    query = """
        SELECT 
            id,
            car_cd,
            inspresultdata_cd,
            branch_no,
            save_file_name,
            created,
            modified
        FROM upload_files
        WHERE (DATE(created) = %s OR DATE(modified) = %s)
          AND delete_flg = 0
          AND save_file_name IS NOT NULL
          AND save_file_name != ''
        ORDER BY 
            COALESCE(inspresultdata_cd, car_cd::text),
            branch_no ASC
    """
    params = (target_date, target_date)

    try:
        print(f"[INFO] DB接続: {DB_CONFIG['host']}")
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()
        cur.execute(query, params)
        rows = cur.fetchall()
        cur.close()
        conn.close()
        print(f"[INFO] DB取得完了: {len(rows)}件")
        return rows
    except Exception as e:
        print(f"[ERROR] DB接続失敗: {e}")
        return []


def build_car_structure(images: list, target_date) -> dict:
    """
    画像リストを車両ごとの構造に変換

    Returns:
        dict: {
            "date": "2026-02-03",
            "total_cars": 10,
            "total_images": 100,
            "cars": {
                "car_id": {
                    "car_cd": "12345678",
                    "inspresultdata_cd": "1554913G",
                    "total_images": 6,
                    "images": [
                        {"branch_no": 1, "path": "/upfile/...", "url": "https://...", "filename": "..."},
                        ...
                    ]
                },
                ...
            }
        }
    """
    cars = defaultdict(
        lambda: {
            "car_cd": None,
            "inspresultdata_cd": None,
            "created": None,
            "modified": None,
            "total_images": 0,
            "images": [],
        }
    )

    for row in images:
        (
            file_id,
            car_cd,
            inspresultdata_cd,
            branch_no,
            save_file_name,
            created,
            modified,
        ) = row

        # car_keyを決定
        car_key = inspresultdata_cd if inspresultdata_cd else str(car_cd)

        # 画像情報を追加
        cars[car_key]["car_cd"] = str(car_cd) if car_cd else None
        cars[car_key]["inspresultdata_cd"] = inspresultdata_cd
        cars[car_key]["total_images"] += 1

        # 車両の作成日・更新日を追跡（最も古いcreated、最も新しいmodified）
        if created:
            created_str = (
                created.isoformat() if hasattr(created, "isoformat") else str(created)
            )
            if (
                cars[car_key]["created"] is None
                or created_str < cars[car_key]["created"]
            ):
                cars[car_key]["created"] = created_str
        if modified:
            modified_str = (
                modified.isoformat()
                if hasattr(modified, "isoformat")
                else str(modified)
            )
            if (
                cars[car_key]["modified"] is None
                or modified_str > cars[car_key]["modified"]
            ):
                cars[car_key]["modified"] = modified_str

        cars[car_key]["images"].append(
            {
                "branch_no": branch_no,
                "path": save_file_name,
                "url": f"{IMAGE_BASE_URL}{save_file_name}",
                "filename": os.path.basename(save_file_name),
                "file_id": file_id,
                "created": (
                    created.isoformat()
                    if created and hasattr(created, "isoformat")
                    else str(created) if created else None
                ),
                "modified": (
                    modified.isoformat()
                    if modified and hasattr(modified, "isoformat")
                    else str(modified) if modified else None
                ),
            }
        )

    # branch_noでソート
    for car_key in cars:
        cars[car_key]["images"].sort(key=lambda x: x["branch_no"] or 999)

    return {
        "date": str(target_date),
        "exported_at": datetime.now().isoformat(),
        "total_cars": len(cars),
        "total_images": len(images),
        "image_base_url": IMAGE_BASE_URL,
        "cars": dict(cars),
    }


def main():
    parser = argparse.ArgumentParser(
        description="指定日の車両画像情報をJSON形式でエクスポート",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  python export_daily_images.py                    # 今日の画像
  python export_daily_images.py --date 2026-02-03  # 特定日
  python export_daily_images.py --days-ago 1       # 昨日
  python export_daily_images.py --output result.json  # 出力ファイル指定
        """,
    )
    parser.add_argument(
        "--days-ago",
        type=int,
        default=0,
        help="何日前の画像を取得するか (0=今日, 1=昨日, ...) [デフォルト: 0]",
    )
    parser.add_argument(
        "--date",
        type=str,
        default=None,
        help="対象日を指定 (YYYY-MM-DD形式、--days-agoより優先)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="出力JSONファイルパス (未指定: images_YYYYMMDD.json)",
    )
    parser.add_argument(
        "--car-id",
        type=str,
        default=None,
        help="特定の車両IDのみ出力",
    )
    parser.add_argument(
        "--chatwork",
        action="store_true",
        help="Chatworkにファイルを送信",
    )

    args = parser.parse_args()

    # 対象日を計算
    if args.date:
        try:
            target_date = datetime.strptime(args.date, "%Y-%m-%d").date()
        except ValueError:
            print(f"[ERROR] 日付形式が不正: {args.date} (YYYY-MM-DD形式で指定)")
            return 1
    else:
        target_date = datetime.now().date() - timedelta(days=args.days_ago)

    # 出力ファイル名
    if args.output:
        output_file = args.output
    else:
        output_file = f"images_{target_date.strftime('%Y%m%d')}.json"

    print("=" * 60)
    print(f"車両画像エクスポート")
    print(f"  対象日: {target_date}")
    print(f"  出力先: {output_file}")
    if args.car_id:
        print(f"  車両ID: {args.car_id}")
    print("=" * 60)

    # 画像を取得
    images = get_images_by_date(target_date)

    if not images:
        print(f"[INFO] {target_date} の画像はありません")
        return 0

    # 構造化
    result = build_car_structure(images, target_date)

    # 特定車両のみフィルタ
    if args.car_id:
        if args.car_id in result["cars"]:
            filtered_car = result["cars"][args.car_id]
            result["cars"] = {args.car_id: filtered_car}
            result["total_cars"] = 1
            result["total_images"] = filtered_car["total_images"]
        else:
            print(f"[ERROR] 車両ID '{args.car_id}' が見つかりません")
            return 1

    # JSON出力
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print("=" * 60)
    print(f"[OK] エクスポート完了")
    print(f"  車両数: {result['total_cars']}台")
    print(f"  画像数: {result['total_images']}枚")
    print(f"  出力先: {output_file}")
    print("=" * 60)

    # サマリー表示
    print("\n[車両一覧]")
    for car_id, car_info in list(result["cars"].items())[:20]:
        print(f"  {car_id}: {car_info['total_images']}枚")
    if len(result["cars"]) > 20:
        print(f"  ... 他 {len(result['cars']) - 20}台")

    # Chatworkにファイル送信
    if args.chatwork:
        message = f"📊 車両画像エクスポート ({target_date})\n車両数: {result['total_cars']}台 / 画像数: {result['total_images']}枚"
        send_file_to_chatwork(output_file, message)

    return 0


if __name__ == "__main__":
    sys.exit(main())
