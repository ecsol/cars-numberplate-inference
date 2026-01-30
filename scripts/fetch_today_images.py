#!/usr/bin/env python3
"""
今日アップロードされた車両の最初の画像のS3パスを取得するスクリプト
crontab: * * * * * /path/to/venv/bin/python /path/to/fetch_today_images.py
"""

import os
import psycopg2
from datetime import datetime

# 環境変数または直接設定
DB_CONFIG = {
    "host": os.getenv("DB_HOST", "cs1adb99-instance-1.cdr2jtnoao7j.ap-northeast-1.rds.amazonaws.com"),
    "database": os.getenv("DB_NAME", "cartrading"),
    "user": os.getenv("DB_USER", "cars_hitosuke"),
    "password": os.getenv("DB_PASSWORD", ""),
}

S3_BUCKET = os.getenv("S3_BUCKET", "your-bucket-name")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "/tmp/plate_images")


def get_today_first_images():
    """今日の各車両の最初の画像を取得"""
    query = """
        SELECT DISTINCT ON (inspresultdata_cd)
            id,
            inspresultdata_cd,
            save_file_name,
            upload_timestamp
        FROM upload_files
        WHERE DATE(upload_timestamp) = CURRENT_DATE
          AND branch_no = 1
          AND delete_flg = 0
        ORDER BY inspresultdata_cd, upload_timestamp ASC
    """
    
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    cur.execute(query)
    rows = cur.fetchall()
    cur.close()
    conn.close()
    
    return rows


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    images = get_today_first_images()
    
    if not images:
        print(f"[{datetime.now()}] 今日の画像はありません")
        return
    
    print(f"[{datetime.now()}] {len(images)}件の画像を取得")
    
    for row in images:
        file_id, car_cd, s3_path, timestamp = row
        print(f"  車両: {car_cd}, パス: s3://{S3_BUCKET}{s3_path}")
        
        # TODO: S3からダウンロードして処理
        # aws s3 cp s3://{S3_BUCKET}{s3_path} {OUTPUT_DIR}/


if __name__ == "__main__":
    main()
