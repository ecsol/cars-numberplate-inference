#!/usr/bin/env python3
"""
車両画像を取得し、ナンバープレートをマスキングするバッチスクリプト

処理フロー:
1. DBから指定日に作成/更新された画像を取得
2. ローカルトラッキングファイルで処理済みをスキップ
3. 各車両の最初の画像(branch_no=1): マスク + バナー追加
4. その他の画像: マスクのみ
5. 元画像を.backupフォルダにバックアップ
6. 処理済みファイルをトラッキングに記録

Usage:
    python fetch_today_images.py                    # 今日の画像、最大10件
    python fetch_today_images.py --days-ago 1      # 昨日の画像
    python fetch_today_images.py --limit 50        # 最大50件処理
    python fetch_today_images.py --days-ago 7 --limit 100

crontab: * * * * * /path/to/venv/bin/python /path/to/fetch_today_images.py
"""

import argparse
import json
import os
import sys
import shutil
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional

import psycopg2

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

from scripts.process_image import process_image

# ======================
# 設定
# ======================
DB_CONFIG = {
    "host": os.getenv("DB_HOST", ""),
    "database": os.getenv("DB_NAME", "cartrading"),
    "user": os.getenv("DB_USER", ""),
    "password": os.getenv("DB_PASSWORD", ""),
}

S3_MOUNT = os.getenv("S3_MOUNT", "")
MODEL_PATH = os.getenv("MODEL_PATH", str(PROJECT_DIR / "models" / "best.pt"))
LOG_DIR = Path(os.getenv("LOG_DIR", "/var/log/plate-detection"))
LOG_FILE = LOG_DIR / "process.log"

# バックアップディレクトリ（S3に書き込めない場合はローカルに保存）
BACKUP_DIR = os.getenv("BACKUP_DIR", "")


# ======================
# ロギング
# ======================
class Logger:
    """詳細ログ出力クラス"""
    
    def __init__(self, log_file: Path):
        self.log_file = log_file
        try:
            self.log_file.parent.mkdir(parents=True, exist_ok=True)
        except PermissionError:
            # フォールバック: カレントディレクトリ
            self.log_file = Path("./process.log")
            print(f"[WARN] ログディレクトリ作成不可、フォールバック: {self.log_file}")
    
    def _write(self, level: str, message: str):
        """ログ出力"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        log_line = f"[{timestamp}] [{level}] {message}"
        print(log_line)
        
        try:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(log_line + "\n")
        except Exception:
            pass
    
    def info(self, message: str):
        self._write("INFO", message)
    
    def error(self, message: str):
        self._write("ERROR", message)
    
    def warn(self, message: str):
        self._write("WARN", message)
    
    def debug(self, message: str):
        self._write("DEBUG", message)
    
    def success(self, message: str):
        self._write("OK", message)


logger = Logger(LOG_FILE)


# ======================
# トラッキング
# ======================
class ProcessingTracker:
    """
    処理済みファイルのトラッキング
    
    日付ごとにJSONファイルで管理:
    /var/log/plate-detection/tracking/processed_20260130.json
    
    形式:
    {
        "date": "2026-01-30",
        "processed": {
            "12345": {
                "file_id": 12345,
                "path": "/upfile/1007/4856/xxx.jpg",
                "processed_at": "2026-01-30 12:34:56",
                "status": "success",
                "detections": 1,
                "is_first": true
            },
            ...
        }
    }
    """
    
    def __init__(self, log_dir: Path):
        self.tracking_dir = log_dir / "tracking"
        try:
            self.tracking_dir.mkdir(parents=True, exist_ok=True)
        except PermissionError:
            # フォールバック: カレントディレクトリ
            self.tracking_dir = Path("./tracking")
            self.tracking_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_tracking_file(self, target_date: datetime.date) -> Path:
        """トラッキングファイルパスを取得"""
        return self.tracking_dir / f"processed_{target_date.strftime('%Y%m%d')}.json"
    
    def load(self, target_date: datetime.date) -> dict:
        """トラッキングデータを読み込み"""
        tracking_file = self._get_tracking_file(target_date)
        
        if not tracking_file.exists():
            return {
                "date": target_date.isoformat(),
                "created_at": datetime.now().isoformat(),
                "processed": {}
            }
        
        try:
            with open(tracking_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"トラッキングファイル読み込み失敗: {e}")
            return {
                "date": target_date.isoformat(),
                "created_at": datetime.now().isoformat(),
                "processed": {}
            }
    
    def save(self, target_date: datetime.date, data: dict):
        """トラッキングデータを保存"""
        tracking_file = self._get_tracking_file(target_date)
        data["updated_at"] = datetime.now().isoformat()
        
        try:
            with open(tracking_file, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"トラッキングファイル保存失敗: {e}")
    
    def is_processed(self, target_date: datetime.date, file_id: int) -> bool:
        """ファイルが処理済みかどうか"""
        data = self.load(target_date)
        return str(file_id) in data["processed"]
    
    def mark_processed(
        self,
        target_date: datetime.date,
        file_id: int,
        path: str,
        status: str,
        detections: int = 0,
        is_first: bool = False,
        error_reason: Optional[str] = None,
    ):
        """ファイルを処理済みとしてマーク"""
        data = self.load(target_date)
        
        record = {
            "file_id": file_id,
            "path": path,
            "processed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "status": status,
            "detections": detections,
            "is_first": is_first,
        }
        
        if error_reason:
            record["error"] = error_reason
        
        data["processed"][str(file_id)] = record
        self.save(target_date, data)
    
    def get_stats(self, target_date: datetime.date) -> dict:
        """統計情報を取得"""
        data = self.load(target_date)
        processed = data.get("processed", {})
        
        stats = {
            "total": len(processed),
            "success": 0,
            "error": 0,
            "skip": 0,
        }
        
        for record in processed.values():
            status = record.get("status", "unknown")
            if status in stats:
                stats[status] += 1
        
        return stats


tracker = ProcessingTracker(LOG_DIR)


# ======================
# データベース
# ======================
def get_images_by_date(days_ago: int = 0) -> list:
    """
    指定日に作成/更新された画像を取得
    
    Args:
        days_ago: 何日前の画像を取得するか (0=今日, 1=昨日, ...)
    
    Returns:
        list: [(id, car_cd, inspresultdata_cd, branch_no, save_file_name, created, modified), ...]
    """
    target_date = datetime.now().date() - timedelta(days=days_ago)
    
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
    
    try:
        logger.debug(f"DB接続: {DB_CONFIG['host']}")
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()
        cur.execute(query, (target_date, target_date))
        rows = cur.fetchall()
        cur.close()
        conn.close()
        logger.debug(f"DB取得完了: {len(rows)}件")
        return rows
    except Exception as e:
        logger.error(f"DB接続失敗: {e}")
        return []


# ======================
# 画像処理
# ======================
def backup_and_process(
    file_path: str,
    is_first_image: bool = False,
) -> dict:
    """
    画像をバックアップして処理
    
    Args:
        file_path: S3上のファイルパス (例: /upfile/1007/4856/20220824190333_1.jpg)
        is_first_image: 最初の画像かどうか (True=バナー追加)
    
    Returns:
        dict: 処理結果
    """
    # フルパスを構築
    full_path = os.path.join(S3_MOUNT, file_path.lstrip("/"))
    
    logger.debug(f"処理開始: {full_path}")
    
    if not os.path.exists(full_path):
        logger.warn(f"ファイル未検出: {full_path}")
        return {"status": "skip", "reason": "file_not_found", "path": full_path}
    
    # バックアップディレクトリ（BACKUP_DIRが設定されていればローカル、なければS3上）
    file_name = os.path.basename(full_path)
    relative_dir = os.path.dirname(file_path.lstrip("/"))
    
    if BACKUP_DIR:
        # ローカルバックアップ（S3が書き込み不可の場合）
        backup_dir = os.path.join(BACKUP_DIR, relative_dir)
    else:
        # S3上にバックアップ（従来の動作）
        file_dir = os.path.dirname(full_path)
        backup_dir = os.path.join(file_dir, ".backup")
    
    backup_path = os.path.join(backup_dir, file_name)
    
    # バックアップディレクトリ作成
    try:
        os.makedirs(backup_dir, exist_ok=True)
    except Exception as e:
        logger.error(f"バックアップディレクトリ作成失敗: {backup_dir} - {e}")
        return {"status": "error", "reason": f"mkdir_failed: {e}", "path": full_path}
    
    # 元画像をバックアップ（既にあれば上書きしない）
    if not os.path.exists(backup_path):
        try:
            logger.debug(f"バックアップ作成: {backup_path}")
            shutil.copy2(full_path, backup_path)
        except Exception as e:
            logger.error(f"バックアップ失敗: {e}")
            return {"status": "error", "reason": f"backup_failed: {e}", "path": full_path}
    else:
        logger.debug(f"バックアップ既存: {backup_path}")
    
    # 処理実行
    try:
        logger.debug(f"モデル推論開始: is_first={is_first_image}")
        result = process_image(
            input_path=full_path,
            output_path=full_path,
            is_masking=is_first_image,
            model_path=MODEL_PATH,
            confidence=0.1,
        )
        result["status"] = "success"
        result["backup_path"] = backup_path
        logger.debug(f"処理完了: 検出数={result.get('detections', 0)}")
        return result
    except Exception as e:
        logger.error(f"処理失敗: {e}")
        # エラー時はバックアップから復元
        try:
            shutil.copy2(backup_path, full_path)
            logger.debug("バックアップから復元完了")
        except Exception:
            pass
        return {"status": "error", "reason": str(e), "path": full_path}


# ======================
# 設定検証
# ======================
def validate_config() -> bool:
    """設定を検証"""
    errors = []
    
    if not DB_CONFIG["host"]:
        errors.append("DB_HOST が設定されていません")
    if not DB_CONFIG["user"]:
        errors.append("DB_USER が設定されていません")
    if not DB_CONFIG["password"]:
        errors.append("DB_PASSWORD が設定されていません")
    if not S3_MOUNT:
        errors.append("S3_MOUNT が設定されていません")
    elif not os.path.exists(S3_MOUNT):
        errors.append(f"S3_MOUNT パスが存在しません: {S3_MOUNT}")
    
    if errors:
        for error in errors:
            logger.error(f"設定エラー: {error}")
        logger.error(".env ファイルまたは環境変数を確認してください")
        return False
    
    return True


# ======================
# メイン処理
# ======================
def main():
    # 引数パース
    parser = argparse.ArgumentParser(
        description="車両画像のナンバープレートをマスキング",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  python fetch_today_images.py                    # 今日の画像、最大10件
  python fetch_today_images.py --days-ago 1      # 昨日の画像
  python fetch_today_images.py --limit 50        # 最大50件処理
  python fetch_today_images.py --days-ago 7 --limit 100
        """
    )
    parser.add_argument(
        "--days-ago",
        type=int,
        default=0,
        help="何日前の画像を処理するか (0=今日, 1=昨日, ...) [デフォルト: 0]"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="1回の実行で処理する最大画像数 [デフォルト: 10]"
    )
    
    args = parser.parse_args()
    
    # 対象日を計算
    target_date = datetime.now().date() - timedelta(days=args.days_ago)
    
    logger.info("=" * 60)
    logger.info(f"バッチ処理開始")
    logger.info(f"  対象日: {target_date}")
    logger.info(f"  最大処理数: {args.limit}件")
    logger.info(f"  S3マウント: {S3_MOUNT}")
    logger.info(f"  モデル: {MODEL_PATH}")
    logger.info("=" * 60)
    
    # 設定検証
    if not validate_config():
        return
    
    # 既存のトラッキング統計
    existing_stats = tracker.get_stats(target_date)
    logger.info(f"トラッキング状況: 処理済み {existing_stats['total']}件 "
                f"(成功: {existing_stats['success']}, エラー: {existing_stats['error']})")
    
    # 画像を取得
    images = get_images_by_date(days_ago=args.days_ago)
    
    if not images:
        logger.info(f"{target_date} の画像はありません")
        return
    
    logger.info(f"DB取得: {len(images)}件")
    
    # 車両ごとにグループ化
    car_images = {}
    for row in images:
        file_id, car_cd, inspresultdata_cd, branch_no, save_file_name, created, modified = row
        
        car_key = inspresultdata_cd if inspresultdata_cd else str(car_cd)
        
        if car_key not in car_images:
            car_images[car_key] = []
        
        car_images[car_key].append({
            "id": file_id,
            "branch_no": branch_no,
            "path": save_file_name,
            "created": created,
            "modified": modified,
        })
    
    logger.info(f"車両数: {len(car_images)}台")
    
    # 処理カウンター
    stats = {"success": 0, "skip_tracked": 0, "skip_other": 0, "error": 0}
    processed_count = 0
    
    # 各車両を処理
    for car_key, car_files in car_images.items():
        # limit到達チェック
        if processed_count >= args.limit:
            logger.info(f"処理上限到達: {args.limit}件")
            break
        
        # branch_noでソート
        car_files.sort(key=lambda x: x["branch_no"] or 999)
        
        logger.debug(f"車両処理開始: {car_key} ({len(car_files)}枚)")
        
        for idx, file_info in enumerate(car_files):
            file_id = file_info["id"]
            
            # limit到達チェック
            if processed_count >= args.limit:
                break
            
            # トラッキングで処理済みかチェック
            if tracker.is_processed(target_date, file_id):
                stats["skip_tracked"] += 1
                continue
            
            is_first = (idx == 0)
            
            # 処理実行
            result = backup_and_process(
                file_path=file_info["path"],
                is_first_image=is_first,
            )
            
            status = result.get("status", "error")
            
            if status == "success":
                stats["success"] += 1
                processed_count += 1
                
                # トラッキングに記録
                tracker.mark_processed(
                    target_date=target_date,
                    file_id=file_id,
                    path=file_info["path"],
                    status="success",
                    detections=result.get("detections", 0),
                    is_first=is_first,
                )
                
                logger.success(
                    f"[{processed_count}/{args.limit}] {file_info['path']} "
                    f"(検出: {result.get('detections', 0)}, "
                    f"バナー: {'あり' if is_first else 'なし'})"
                )
                
            elif status == "error":
                stats["error"] += 1
                processed_count += 1
                
                # エラーもトラッキングに記録
                tracker.mark_processed(
                    target_date=target_date,
                    file_id=file_id,
                    path=file_info["path"],
                    status="error",
                    error_reason=result.get("reason", "unknown"),
                )
                
                logger.error(f"{file_info['path']} - {result.get('reason', 'unknown')}")
                
            else:  # skip
                stats["skip_other"] += 1
                logger.debug(f"スキップ: {file_info['path']} - {result.get('reason', '')}")
    
    # 最終統計
    logger.info("=" * 60)
    logger.info("処理完了")
    logger.info(f"  成功: {stats['success']}件")
    logger.info(f"  エラー: {stats['error']}件")
    logger.info(f"  スキップ（処理済み）: {stats['skip_tracked']}件")
    logger.info(f"  スキップ（その他）: {stats['skip_other']}件")
    
    # 最終トラッキング統計
    final_stats = tracker.get_stats(target_date)
    logger.info(f"トラッキング累計: {final_stats['total']}件処理済み")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
