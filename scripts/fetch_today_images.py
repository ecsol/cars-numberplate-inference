#!/usr/bin/env python3
"""
車両画像を取得し、ナンバープレートをマスキングするバッチスクリプト

╔══════════════════════════════════════════════════════════════════════════╗
║  【重要ルール】バナー/オーバーレイの追加条件                              ║
║  ──────────────────────────────────────────────────────────────────────── ║
║  ● branch_no=1 (先頭画像) → バナー追加OK                                 ║
║  ● branch_no!=1 (2枚目以降) → バナー追加【絶対禁止】マスクのみ           ║
╚══════════════════════════════════════════════════════════════════════════╝

処理フロー:
1. DBから指定日に作成/更新された画像を取得
2. ローカルトラッキングファイルで処理済みをスキップ
3. 各車両の最初の画像(branch_no=1): マスク + バナー追加
4. その他の画像(branch_no!=1): マスクのみ【バナー禁止】
5. 元画像を.backupフォルダにバックアップ（初回のみ、復元は手動）
6. 処理済みファイルをトラッキングに記録

Usage:
    python fetch_today_images.py                    # 今日の画像、最大10件
    python fetch_today_images.py --days-ago 1      # 昨日の画像
    python fetch_today_images.py --limit 50        # 最大50件処理
    python fetch_today_images.py --days-ago 7 --limit 100
    python fetch_today_images.py --path /1554913G  # 特定フォルダを直接処理
    python fetch_today_images.py --force           # .detect/ + original強制再作成
    /home/ec2-user/plate-detection-service/venv/bin/python fetch_today_images.py --force-overlay   # 元画像に強制バナー（branch_no=1のみ）

crontab: * * * * * /path/to/venv/bin/python /path/to/fetch_today_images.py
"""

import argparse
import json
import os
import sys
import shutil
import warnings
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional

# Suppress boto3 Python 3.9 deprecation warning
warnings.filterwarnings("ignore", message=".*Boto3 will no longer support Python 3.9.*")

import boto3
from botocore.exceptions import ClientError
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

from scripts.process_image_v2 import process_image

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

# Two-Stage モデルパス
SEG_MODEL_PATH = os.getenv(
    "SEG_MODEL_PATH", str(PROJECT_DIR / "models" / "best_yolo26x_lambda_20260201.pt")
)
POSE_MODEL_PATH = os.getenv(
    "POSE_MODEL_PATH", str(PROJECT_DIR / "models" / "yolo26x_pose_best.pt")
)
PLATE_MASK_PATH = os.getenv(
    "PLATE_MASK_PATH", str(PROJECT_DIR / "assets" / "plate_mask.png")
)

# 検出スキップ対象のbranch_no
# これらの画像はナンバープレート検出をスキップし、そのまま.detect/にコピーする
# - 30: メーターパネル (Photo30)
# - 31: コーションプレート (Photo31)
# - 32: 車検証 (Photo32)
SKIP_DETECTION_BRANCH_NOS = {30, 31, 32}

# ログディレクトリ（デフォルト: プロジェクトフォルダ内 logs/）
_log_dir_env = os.getenv("LOG_DIR", "")
if _log_dir_env:
    LOG_DIR = Path(_log_dir_env)
    if not LOG_DIR.is_absolute():
        LOG_DIR = PROJECT_DIR / _log_dir_env
else:
    LOG_DIR = PROJECT_DIR / "logs"
LOG_FILE = LOG_DIR / "process.log"

# バックアップ設定
# BACKUP_S3_BUCKET: boto3でS3に直接バックアップ（推奨 - mountpoint-s3の制限を回避）
# BACKUP_DIR: ローカルにバックアップ
# どちらも未設定: エラー
BACKUP_S3_BUCKET = os.getenv("BACKUP_S3_BUCKET", "")  # 例: cs1es3
BACKUP_S3_PREFIX = os.getenv("BACKUP_S3_PREFIX", ".backup")  # S3 key prefix
BACKUP_DIR = os.getenv("BACKUP_DIR", "")

# boto3 S3 client (lazy init)
_s3_client = None

# Chatwork通知設定（オプション）
CHATWORK_API_KEY = os.getenv("CHATWORK_API_KEY", "")
CHATWORK_ROOM_ID = os.getenv("CHATWORK_ROOM_ID", "")
# 画像のベースURL（Chatwork通知用）
IMAGE_BASE_URL = os.getenv("IMAGE_BASE_URL", "https://www.autobacs-cars-system.com")
# 担当者リスト（Chatworkメンション用）
# ファムタイズオン (8892649) は除外
CHATWORK_MENTION_USERS = [
    ("11055639", "BaoNTV"),
    ("11055644", "Nguyen Duc Thang"),
    ("11055661", "MinhDV"),
]


def get_s3_client():
    """S3クライアントを取得（遅延初期化）"""
    global _s3_client
    if _s3_client is None:
        _s3_client = boto3.client("s3")
    return _s3_client


def send_chatwork_notification(message: str) -> bool:
    """
    Chatworkに通知を送信

    Args:
        message: 送信するメッセージ

    Returns:
        bool: 送信成功かどうか
    """
    if not CHATWORK_API_KEY or not CHATWORK_ROOM_ID:
        return False

    try:
        url = f"https://api.chatwork.com/v2/rooms/{CHATWORK_ROOM_ID}/messages"
        headers = {"X-ChatWorkToken": CHATWORK_API_KEY}
        data = {"body": message}

        response = requests.post(url, headers=headers, data=data, timeout=10)
        response.raise_for_status()
        return True
    except Exception as e:
        logger.error(f"Chatwork通知失敗: {e}")
        return False


def build_processing_summary(
    target_date: datetime.date,
    stats: dict,
    car_results: list,
) -> str:
    """
    処理結果のサマリーメッセージを作成

    Args:
        target_date: 対象日
        stats: 統計情報
        car_results: 車両ごとの処理結果
            [(car_id, success_count, error_count, detections, images_list), ...]
            images_list: [(branch_no, path), ...] sorted by branch_no

    Returns:
        str: Chatwork用メッセージ
    """
    lines = [
        "[info][title]🚗 ナンバープレート処理完了[/title]",
        f"📅 対象日: {target_date}",
        f"✅ 成功: {stats['success']}件",
        f"❌ エラー: {stats['error']}件",
        f"⏭️ スキップ: {stats['skip_tracked'] + stats['skip_other']}件",
        "[/info]",
        "",
    ]

    if car_results:
        lines.append("[info][title]📊 車両別結果[/title]")
        for idx, (car_id, success, error, detections, car_images) in enumerate(
            car_results[:10]
        ):
            status_icon = "✅" if error == 0 else "⚠️"

            # 担当者をローテーション
            if CHATWORK_MENTION_USERS:
                user_idx = idx % len(CHATWORK_MENTION_USERS)
                user_id, user_name = CHATWORK_MENTION_USERS[user_idx]
                mention = f"[To:{user_id}]{user_name}さん"
                lines.append(
                    f"{status_icon} {car_id}: {success}枚処理, 検出{detections}件 担当:{mention}"
                )
            else:
                lines.append(
                    f"{status_icon} {car_id}: {success}枚処理, 検出{detections}件"
                )

            # 全画像のURLをbranch_no順で表示（オリジナル + .detect/マスク済み）
            # branch_noではなく連番で表示（1から開始）
            for seq_no, (branch_no, path) in enumerate(car_images, start=1):
                dir_path = os.path.dirname(path)
                file_name = os.path.basename(path)
                # オリジナルURL
                original_url = f"{IMAGE_BASE_URL}{path}"
                # マスク済みURL (.detect/)
                detect_url = f"{IMAGE_BASE_URL}{dir_path}/.detect/{file_name}"
                lines.append(f"  {seq_no}. 元: {original_url}")
                lines.append(f"     検: {detect_url}")
            lines.append("")

        if len(car_results) > 10:
            lines.append(f"... 他 {len(car_results) - 10}台")
        lines.append("[/info]")

    return "\n".join(lines)


def s3_backup_exists(s3_key: str) -> bool:
    """S3にバックアップが存在するか確認"""
    try:
        get_s3_client().head_object(Bucket=BACKUP_S3_BUCKET, Key=s3_key)
        return True
    except ClientError as e:
        if e.response["Error"]["Code"] == "404":
            return False
        raise


def s3_upload_backup(local_path: str, s3_key: str):
    """ローカルファイルをS3にバックアップ"""
    get_s3_client().upload_file(local_path, BACKUP_S3_BUCKET, s3_key)


def s3_download_backup(s3_key: str, local_path: str):
    """S3からバックアップをダウンロード（mountpoint-s3対応）"""
    # mountpoint-s3はshutil.copyが動かないため、
    # バイト単位で直接書き込む
    response = get_s3_client().get_object(Bucket=BACKUP_S3_BUCKET, Key=s3_key)
    data = response["Body"].read()

    with open(local_path, "wb") as f:
        f.write(data)


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

# ログローテーション設定
LOG_RETENTION_DAYS = int(os.getenv("LOG_RETENTION_DAYS", "60"))


# ======================
# トラッキング
# ======================
# ステータス定義:
#   pending    - DBから取得、処理待ち
#   processing - 処理中
#   verified   - 出力ファイル確認済み
#   done       - 完了（成功）
#   error      - エラー発生
TRACKING_STATUS = {
    "pending": "pending",
    "processing": "processing",
    "verified": "verified",
    "done": "done",
    "error": "error",
}


class ProcessingTracker:
    """
    処理済みファイルのトラッキング（状態管理付き）

    日付ごとにJSONファイルで管理:
    {PROJECT_DIR}/logs/tracking/processed_20260130.json

    ステータスフロー:
        pending → processing → verified → done
                      ↓
                    error

    形式:
    {
        "date": "2026-01-30",
        "processed": {
            "12345": {
                "file_id": 12345,
                "path": "/upfile/1007/4856/xxx.jpg",
                "status": "done",
                "status_history": [
                    {"status": "pending", "at": "2026-01-30 12:34:50"},
                    {"status": "processing", "at": "2026-01-30 12:34:51"},
                    {"status": "verified", "at": "2026-01-30 12:34:55"},
                    {"status": "done", "at": "2026-01-30 12:34:56"}
                ],
                "detections": 1,
                "is_first": true,
                "output_paths": {
                    "detect": "/upfile/.../xxx.jpg",
                    "original": "/upfile/.../xxx.jpg"  # branch_no=1のみ
                }
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
                "processed": {},
            }

        try:
            with open(tracking_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"トラッキングファイル読み込み失敗: {e}")
            return {
                "date": target_date.isoformat(),
                "created_at": datetime.now().isoformat(),
                "processed": {},
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

    def get_status(self, target_date: datetime.date, file_id: int) -> Optional[str]:
        """ファイルの現在のステータスを取得"""
        data = self.load(target_date)
        record = data["processed"].get(str(file_id))
        if record:
            return record.get("status")
        return None

    def is_done(self, target_date: datetime.date, file_id: int) -> bool:
        """ファイルが完了済み（done）かどうか"""
        status = self.get_status(target_date, file_id)
        return status == TRACKING_STATUS["done"]

    def is_processed(self, target_date: datetime.date, file_id: int) -> bool:
        """ファイルが処理済み（done または verified）かどうか"""
        status = self.get_status(target_date, file_id)
        return status in [TRACKING_STATUS["done"], TRACKING_STATUS["verified"]]

    def needs_processing(self, target_date: datetime.date, file_id: int) -> bool:
        """ファイルが処理必要かどうか（pending, processing, または未登録）"""
        status = self.get_status(target_date, file_id)
        if status is None:
            return True
        return status in [TRACKING_STATUS["pending"], TRACKING_STATUS["processing"]]

    def has_car_any_done(
        self, target_date: datetime.date, car_path_prefix: str
    ) -> bool:
        """車両のいずれかのファイルがdone状態かどうか（パスパターンでチェック）

        Args:
            car_path_prefix: 例 "/upfile/1041/8430/"
        """
        data = self.load(target_date)
        for record in data.get("processed", {}).values():
            path = record.get("path", "")
            status = record.get("status", "")
            if path.startswith(car_path_prefix) and status == TRACKING_STATUS["done"]:
                return True
        return False

    def has_car_all_done(
        self, target_date: datetime.date, car_path_prefix: str
    ) -> bool:
        """車両の全ファイルがdone状態かどうか"""
        data = self.load(target_date)
        car_files = []
        for record in data.get("processed", {}).values():
            path = record.get("path", "")
            if path.startswith(car_path_prefix):
                car_files.append(record)

        if not car_files:
            return False

        return all(r.get("status") == TRACKING_STATUS["done"] for r in car_files)

    def _add_status_history(self, record: dict, new_status: str):
        """ステータス履歴を追加"""
        if "status_history" not in record:
            record["status_history"] = []
        record["status_history"].append({
            "status": new_status,
            "at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        })

    def mark_pending(
        self,
        target_date: datetime.date,
        file_id: int,
        path: str,
        branch_no: Optional[int] = None,
        car_id: Optional[str] = None,
        is_first: bool = False,
    ):
        """ファイルをpending状態としてマーク"""
        data = self.load(target_date)
        file_key = str(file_id)

        # 既存レコードがあればスキップ（再処理しない）
        if file_key in data["processed"]:
            return

        record = {
            "file_id": file_id,
            "car_id": car_id,
            "path": path,
            "branch_no": branch_no,
            "is_first": is_first,
            "status": TRACKING_STATUS["pending"],
            "status_history": [],
        }
        self._add_status_history(record, TRACKING_STATUS["pending"])

        data["processed"][file_key] = record
        self.save(target_date, data)

    def mark_processing(self, target_date: datetime.date, file_id: int):
        """ファイルをprocessing状態としてマーク"""
        data = self.load(target_date)
        file_key = str(file_id)

        if file_key not in data["processed"]:
            logger.warn(f"ファイル {file_id} がトラッキングに存在しません")
            return

        record = data["processed"][file_key]
        record["status"] = TRACKING_STATUS["processing"]
        self._add_status_history(record, TRACKING_STATUS["processing"])

        self.save(target_date, data)

    def mark_verified(
        self,
        target_date: datetime.date,
        file_id: int,
        detections: int = 0,
        output_paths: Optional[dict] = None,
    ):
        """ファイルをverified状態としてマーク（出力確認済み）"""
        data = self.load(target_date)
        file_key = str(file_id)

        if file_key not in data["processed"]:
            logger.warn(f"ファイル {file_id} がトラッキングに存在しません")
            return

        record = data["processed"][file_key]
        record["status"] = TRACKING_STATUS["verified"]
        record["detections"] = detections
        if output_paths:
            record["output_paths"] = output_paths
        self._add_status_history(record, TRACKING_STATUS["verified"])

        self.save(target_date, data)

    def mark_done(self, target_date: datetime.date, file_id: int):
        """ファイルをdone状態としてマーク（完了）"""
        data = self.load(target_date)
        file_key = str(file_id)

        if file_key not in data["processed"]:
            logger.warn(f"ファイル {file_id} がトラッキングに存在しません")
            return

        record = data["processed"][file_key]
        record["status"] = TRACKING_STATUS["done"]
        record["completed_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._add_status_history(record, TRACKING_STATUS["done"])

        self.save(target_date, data)

    def mark_error(
        self,
        target_date: datetime.date,
        file_id: int,
        error_reason: str,
    ):
        """ファイルをerror状態としてマーク"""
        data = self.load(target_date)
        file_key = str(file_id)

        if file_key not in data["processed"]:
            logger.warn(f"ファイル {file_id} がトラッキングに存在しません")
            return

        record = data["processed"][file_key]
        record["status"] = TRACKING_STATUS["error"]
        record["error"] = error_reason
        self._add_status_history(record, TRACKING_STATUS["error"])

        self.save(target_date, data)

    def mark_car_done(self, target_date: datetime.date, car_path_prefix: str):
        """車両の全ファイルをdone状態としてマーク"""
        data = self.load(target_date)

        for file_key, record in data["processed"].items():
            path = record.get("path", "")
            status = record.get("status", "")
            if path.startswith(car_path_prefix) and status == TRACKING_STATUS["verified"]:
                record["status"] = TRACKING_STATUS["done"]
                record["completed_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                self._add_status_history(record, TRACKING_STATUS["done"])

        self.save(target_date, data)

    # Legacy method for backward compatibility
    def has_car_any_processed(
        self, target_date: datetime.date, car_path_prefix: str
    ) -> bool:
        """車両のいずれかのファイルが処理済みかどうか（後方互換性のため維持）"""
        return self.has_car_any_done(target_date, car_path_prefix)

    def mark_processed(
        self,
        target_date: datetime.date,
        file_id: int,
        path: str,
        status: str,
        detections: int = 0,
        is_first: bool = False,
        branch_no: Optional[int] = None,
        car_id: Optional[str] = None,
        error_reason: Optional[str] = None,
    ):
        """ファイルを処理済みとしてマーク（後方互換性のため維持）"""
        data = self.load(target_date)

        record = {
            "file_id": file_id,
            "car_id": car_id,
            "path": path,
            "branch_no": branch_no,
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
            "pending": 0,
            "processing": 0,
            "verified": 0,
            "done": 0,
            "error": 0,
            # Legacy compatibility
            "success": 0,
            "skip": 0,
        }

        for record in processed.values():
            status = record.get("status", "unknown")
            if status in stats:
                stats[status] += 1
            # Legacy: count 'done' as 'success' too
            if status == "done":
                stats["success"] += 1

        return stats

    def get_last_processed_time(self, target_date: datetime.date) -> Optional[datetime]:
        """最後に処理した時刻を取得"""
        data = self.load(target_date)
        last_time_str = data.get("last_processed_time")
        if last_time_str:
            try:
                return datetime.fromisoformat(last_time_str)
            except ValueError:
                return None
        return None

    def set_last_processed_time(self, target_date: datetime.date, last_time: datetime):
        """最後に処理した時刻を保存"""
        data = self.load(target_date)
        data["last_processed_time"] = last_time.isoformat()
        self.save(target_date, data)

    def cleanup_old_files(self, retention_days: int = 60):
        """古いトラッキングファイルを削除"""
        if not self.tracking_dir.exists():
            return

        cutoff_date = datetime.now().date() - timedelta(days=retention_days)
        deleted_count = 0

        for file_path in self.tracking_dir.glob("processed_*.json"):
            try:
                # ファイル名から日付を抽出: processed_20260130.json
                date_str = file_path.stem.replace("processed_", "")
                file_date = datetime.strptime(date_str, "%Y%m%d").date()

                if file_date < cutoff_date:
                    file_path.unlink()
                    deleted_count += 1
                    logger.debug(f"トラッキングファイル削除: {file_path.name}")
            except (ValueError, OSError) as e:
                logger.debug(f"ファイル処理スキップ: {file_path.name} - {e}")

        if deleted_count > 0:
            logger.info(
                f"トラッキングローテーション: {deleted_count}件削除 ({retention_days}日以前)"
            )


tracker = ProcessingTracker(LOG_DIR)


# ======================
# 出力ファイル検証
# ======================
def verify_output_exists(
    file_path: str,
    is_first_image: bool = False,
) -> dict:
    """
    出力ファイルが存在するか検証

    Args:
        file_path: 元ファイルパス (例: /upfile/1041/8430/xxx.jpg)
        is_first_image: 最初の画像かどうか

    Returns:
        dict: {
            "verified": bool,
            "detect_exists": bool,
            "detect_path": str,
            "original_exists": bool,  # is_first_imageの場合のみ
            "original_path": str,
            "missing": list  # 欠損ファイルリスト
        }
    """
    full_path = os.path.join(S3_MOUNT, file_path.lstrip("/"))
    file_name = os.path.basename(full_path)
    dir_path = os.path.dirname(full_path)

    result = {
        "verified": False,
        "detect_exists": False,
        "detect_path": None,
        "original_exists": False,
        "original_path": full_path,
        "missing": [],
    }

    # .detect/ ファイルチェック
    detect_path = os.path.join(dir_path, ".detect", file_name)
    result["detect_path"] = detect_path

    if BACKUP_S3_BUCKET:
        # S3の場合はboto3で確認
        relative_path = file_path.lstrip("/")
        dir_part = os.path.dirname(relative_path)
        detect_s3_key = f"webroot/{dir_part}/.detect/{file_name}"
        try:
            result["detect_exists"] = s3_backup_exists(detect_s3_key)
        except Exception:
            result["detect_exists"] = False
    else:
        result["detect_exists"] = os.path.exists(detect_path)

    if not result["detect_exists"]:
        result["missing"].append(detect_path)

    # First imageの場合はoriginalもチェック
    if is_first_image:
        if BACKUP_S3_BUCKET:
            relative_path = file_path.lstrip("/")
            original_s3_key = f"webroot/{relative_path}"
            try:
                result["original_exists"] = s3_backup_exists(original_s3_key)
            except Exception:
                result["original_exists"] = False
        else:
            result["original_exists"] = os.path.exists(full_path)

        # originalは常に存在するはずなので、missing には追加しない
        # （存在しない場合は別の問題）

    # 全て存在すればverified
    if is_first_image:
        result["verified"] = result["detect_exists"] and result["original_exists"]
    else:
        result["verified"] = result["detect_exists"]

    return result


# ======================
# モデル・マスク読み込み（グローバル - 起動時に1回のみ）
# ======================
seg_model = None
pose_model = None
mask_image = None


def load_models():
    """モデルとマスク画像を読み込み（初回のみ）"""
    global seg_model, pose_model, mask_image

    if seg_model is None:
        from ultralytics import YOLO

        logger.info(f"Segモデル読み込み: {SEG_MODEL_PATH}")
        seg_model = YOLO(SEG_MODEL_PATH)

    if pose_model is None:
        from ultralytics import YOLO

        logger.info(f"Poseモデル読み込み: {POSE_MODEL_PATH}")
        pose_model = YOLO(POSE_MODEL_PATH)

    if mask_image is None:
        import cv2

        logger.info(f"マスク画像読み込み: {PLATE_MASK_PATH}")
        mask_image = cv2.imread(PLATE_MASK_PATH, cv2.IMREAD_UNCHANGED)
        if mask_image is None:
            raise ValueError(f"マスク画像を読み込めません: {PLATE_MASK_PATH}")

    return seg_model, pose_model, mask_image


# ======================
# データベース
# ======================
def get_images_from_path(folder_path: str) -> list:
    """
    指定フォルダから画像ファイルを取得（DBからbranch_noを取得）

    【重要】branch_noは必ずDBから取得する（ファイル名から推測しない）

    Args:
        folder_path: フォルダパス (例: /1554913G または 1554913G)

    Returns:
        list: [(id, car_cd, inspresultdata_cd, branch_no, save_file_name, created, modified), ...]
              get_images_by_dateと同じ形式で返す

    Note:
        DBに存在しないファイルはスキップされる
    """
    # パスを正規化
    folder_path = folder_path.strip("/")
    full_folder_path = os.path.join(S3_MOUNT, "upfile", folder_path)

    if not os.path.exists(full_folder_path):
        logger.error(f"フォルダが存在しません: {full_folder_path}")
        return []

    # car_id を抽出（フォルダ名）
    car_id = os.path.basename(folder_path)

    # ファイルシステムから画像ファイル一覧を取得（存在確認用）
    image_extensions = {".jpg", ".jpeg", ".png", ".gif", ".webp"}
    file_names = []

    for file_name in sorted(os.listdir(full_folder_path)):
        file_path = os.path.join(full_folder_path, file_name)

        # ファイルのみ、画像拡張子のみ
        if not os.path.isfile(file_path):
            continue

        _, ext = os.path.splitext(file_name)
        if ext.lower() not in image_extensions:
            continue

        # .backup, .detect フォルダ内はスキップ
        if ".backup" in file_path or ".detect" in file_path:
            continue

        file_names.append(file_name)

    if not file_names:
        logger.info(f"フォルダスキャン: {full_folder_path} - 画像なし")
        return []

    # DBからbranch_noを取得（絶対にファイル名から推測しない！）
    # フォルダIDには2種類ある:
    #   1. car_cd形式: 1041/9302 (数値/数値)
    #   2. inspresultdata_cd形式: 1555316G (英数字)
    # 両方のパターンで検索する
    like_pattern = f"/upfile/{folder_path}/%"

    # inspresultdata_cdの場合はフォルダ名がそのままID
    # car_cdの場合はフォルダ名の最後の部分がID (例: 1041/9302 → 9302)
    folder_id = os.path.basename(folder_path)

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
        WHERE (save_file_name LIKE %s OR inspresultdata_cd = %s)
          AND delete_flg = 0
        ORDER BY branch_no ASC
    """

    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()
        cur.execute(query, (like_pattern, folder_id))
        rows = cur.fetchall()
        cur.close()
        conn.close()

        # DBの結果をそのまま返す（branch_noはDBの値を使用）
        logger.info(
            f"フォルダスキャン: {full_folder_path} - "
            f"ファイル{len(file_names)}枚, DB登録{len(rows)}件"
        )

        if len(rows) != len(file_names):
            logger.warn(
                f"ファイル数とDB登録数が一致しません: "
                f"ファイル={len(file_names)}, DB={len(rows)}"
            )

        return rows

    except Exception as e:
        logger.error(f"DB接続エラー: {e}")
        return []


def get_images_by_date(
    target_date: datetime.date,
    last_fetch_time: Optional[datetime] = None,
    only_first: bool = False,
    order: str = "newest",
) -> list:
    """
    指定日に作成/更新された画像を取得

    Args:
        target_date: 対象日
        last_fetch_time: この時刻以降に作成/更新された画像のみ取得（増分取得でDB負荷軽減）
        only_first: Trueの場合、branch_no=1のみ取得（--force-overlay用）
        order: 並び順 "newest"=新しい順, "oldest"=古い順

    Returns:
        list: [(id, car_cd, inspresultdata_cd, branch_no, save_file_name, created, modified), ...]
    """

    # branch_no=1のみ取得する条件
    branch_condition = "AND branch_no = 1" if only_first else ""

    # 並び順: newest=新しい順(DESC), oldest=古い順(ASC)
    date_order = "DESC" if order == "newest" else "ASC"

    if last_fetch_time:
        # 増分取得: last_fetch_time以降の新規/更新ファイルのみ
        query = f"""
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
              AND (created > %s OR modified > %s)
              AND delete_flg = 0
              AND save_file_name IS NOT NULL
              AND save_file_name != ''
              {branch_condition}
            ORDER BY 
                GREATEST(created, modified) {date_order},
                COALESCE(inspresultdata_cd, car_cd::text),
                branch_no ASC
        """
        params = (target_date, target_date, last_fetch_time, last_fetch_time)
    else:
        # 初回: 全件取得
        query = f"""
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
              {branch_condition}
            ORDER BY 
                GREATEST(created, modified) {date_order},
                COALESCE(inspresultdata_cd, car_cd::text),
                branch_no ASC
        """
        params = (target_date, target_date)

    try:
        logger.debug(f"DB接続: {DB_CONFIG['host']}")
        logger.debug(f"対象日: {target_date}, only_first: {only_first}")
        if last_fetch_time:
            logger.debug(f"増分取得: {last_fetch_time} 以降")
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()
        cur.execute(query, params)
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
    force_overlay: bool = False,
    force: bool = False,
    branch_no: Optional[int] = None,
) -> dict:
    """
    画像をバックアップして処理

    ╔══════════════════════════════════════════════════════════════════╗
    ║  【重要】バナー/オーバーレイのルール                              ║
    ║  ─────────────────────────────────────────────────────────────── ║
    ║  ● branch_no=1 (先頭画像):                                       ║
    ║    → バナー追加OK（元画像・.detect/両方）                        ║
    ║                                                                  ║
    ║  ● branch_no!=1 (2枚目以降):                                     ║
    ║    → バナー追加【絶対禁止】                                      ║
    ║    → マスク処理のみ（.detect/に保存）                            ║
    ║    → 元画像は変更しない                                          ║
    ║                                                                  ║
    ║  ● branch_no=30,31,32 (特殊画像):                                ║
    ║    → 検出スキップ、そのまま.detect/にコピー                      ║
    ╚══════════════════════════════════════════════════════════════════╝

    処理フロー:
    1. .backupフォルダがなければ作成し、元画像をバックアップ
    2. .backupに既にファイルがあればスキップ（バックアップ済み）
    3. 処理実行（.backupから入力して検出精度を確保）
    4. branch_no=30,31,32は検出スキップ、コピーのみ

    Args:
        file_path: S3上のファイルパス (例: /upfile/1007/4856/20220824190333_1.jpg)
        is_first_image: 最初の画像かどうか (True=branch_no=1, バナー追加対象)
        force_overlay: 強制的にバナーを追加するか（branch_no=1のみ有効）
        force: .detect/が存在しても強制的に再処理（branch_no=1のみ有効）
        branch_no: 画像のbranch_no（検出スキップ判定用）

    Returns:
        dict: 処理結果
    """
    # フルパスを構築
    full_path = os.path.join(S3_MOUNT, file_path.lstrip("/"))

    logger.debug(f"処理開始: {full_path}")

    if not os.path.exists(full_path):
        logger.warn(f"ファイル未検出: {full_path}")
        return {"status": "skip", "reason": "file_not_found", "path": full_path}

    # バックアップパス設定
    file_name = os.path.basename(full_path)
    relative_path = file_path.lstrip("/")  # upfile/1041/8430/xxx.jpg

    # バックアップ処理（--forceモードでもバックアップがなければ作成）
    if force:
        # --force モード: バックアップがなければ作成、あればそれを使用
        if BACKUP_S3_BUCKET:
            dir_part = os.path.dirname(relative_path)
            s3_key = f"webroot/{dir_part}/.backup/{file_name}"
            try:
                if not s3_backup_exists(s3_key):
                    logger.debug(
                        f"--force: バックアップなし、作成: s3://{BACKUP_S3_BUCKET}/{s3_key}"
                    )
                    s3_upload_backup(full_path, s3_key)
            except Exception as e:
                logger.warn(f"--force: バックアップ作成失敗: {e}")
        elif BACKUP_DIR:
            backup_path = os.path.join(BACKUP_DIR, relative_path)
            if not os.path.exists(backup_path):
                try:
                    backup_dir = os.path.dirname(backup_path)
                    os.makedirs(backup_dir, exist_ok=True)
                    shutil.copy(full_path, backup_path)
                    logger.debug(f"--force: バックアップ作成: {backup_path}")
                except Exception as e:
                    logger.warn(f"--force: バックアップ作成失敗: {e}")
    else:
        # === 通常モード: 初回のみバックアップ作成（復元は手動restore_from_backup.pyで） ===
        # バックアップモード判定
        # 優先順位: BACKUP_S3_BUCKET > BACKUP_DIR
        if BACKUP_S3_BUCKET:
            # === boto3 S3バックアップ（推奨）===
            # S3 key: webroot/upfile/1041/8430/.backup/xxx.jpg
            dir_part = os.path.dirname(relative_path)  # upfile/1041/8430
            s3_key = f"webroot/{dir_part}/.backup/{file_name}"

            try:
                backup_exists = s3_backup_exists(s3_key)

                if not backup_exists:
                    # 初回: S3にバックアップ
                    logger.debug(
                        f"S3バックアップ作成: s3://{BACKUP_S3_BUCKET}/{s3_key}"
                    )
                    s3_upload_backup(full_path, s3_key)
                # backup_exists の場合は何もしない（手動復元用に保持）
            except Exception as e:
                logger.error(f"S3バックアップ失敗: {e}")
                return {
                    "status": "error",
                    "reason": f"s3_backup_failed: {e}",
                    "path": full_path,
                }
        elif BACKUP_DIR:
            # === ローカルバックアップ ===
            backup_path = os.path.join(BACKUP_DIR, relative_path)
            backup_dir = os.path.dirname(backup_path)
            backup_exists = os.path.exists(backup_path)

            if not backup_exists:
                # 初回: ローカルにバックアップ
                try:
                    if not os.path.exists(backup_dir):
                        os.makedirs(backup_dir, exist_ok=True)
                    logger.debug(f"ローカルバックアップ作成: {backup_path}")
                    shutil.copy(full_path, backup_path)
                except Exception as e:
                    logger.error(f"バックアップ失敗: {e}")
                    return {
                        "status": "error",
                        "reason": f"backup_failed: {e}",
                        "path": full_path,
                    }
            # backup_exists の場合は何もしない（手動復元用に保持）
        else:
            # どちらも未設定はエラー
            logger.error("BACKUP_S3_BUCKET または BACKUP_DIR を設定してください")
            return {
                "status": "error",
                "reason": "no_backup_config",
                "path": full_path,
            }

    # .detect/ フォルダパス
    dir_part = os.path.dirname(relative_path)  # upfile/1041/8430

    # ============================================================
    # 検出スキップ対象のbranch_no (30, 31, 32)
    # ============================================================
    # これらは車両画像ではないため、ナンバープレート検出をスキップ
    # - 30: メーターパネル
    # - 31: コーションプレート
    # - 32: 車検証
    # 処理: バックアップ作成 → そのまま.detect/にコピー（検出なし）
    # ============================================================
    skip_detection = branch_no in SKIP_DETECTION_BRANCH_NOS if branch_no else False

    if skip_detection:
        logger.debug(f"検出スキップ: branch_no={branch_no} (コピーのみ)")

        # バックアップ作成（必須）
        if BACKUP_S3_BUCKET:
            backup_s3_key = f"webroot/{dir_part}/.backup/{file_name}"
            try:
                if not s3_backup_exists(backup_s3_key):
                    s3_upload_backup(full_path, backup_s3_key)
                    logger.debug(f"バックアップ作成: s3://{BACKUP_S3_BUCKET}/{backup_s3_key}")
            except Exception as e:
                logger.warn(f"バックアップ作成失敗: {e}")

            # .detect/にコピー（検出なし、そのままコピー）
            detect_s3_key = f"webroot/{dir_part}/.detect/{file_name}"
            try:
                s3_upload_backup(full_path, detect_s3_key)
                logger.debug(f".detect/コピー完了: s3://{BACKUP_S3_BUCKET}/{detect_s3_key}")
            except Exception as e:
                logger.error(f".detect/コピー失敗: {e}")
                return {
                    "status": "error",
                    "reason": f"detect_copy_failed: {e}",
                    "path": full_path,
                }

            return {
                "status": "success",
                "path": full_path,
                "output_path": f"s3://{BACKUP_S3_BUCKET}/{detect_s3_key}",
                "detections": 0,
                "skip_detection": True,
                "branch_no": branch_no,
            }
        elif BACKUP_DIR:
            backup_path = os.path.join(BACKUP_DIR, relative_path)
            if not os.path.exists(backup_path):
                try:
                    backup_dir_path = os.path.dirname(backup_path)
                    os.makedirs(backup_dir_path, exist_ok=True)
                    shutil.copy(full_path, backup_path)
                    logger.debug(f"バックアップ作成: {backup_path}")
                except Exception as e:
                    logger.warn(f"バックアップ作成失敗: {e}")

            # .detect/にコピー（検出なし、そのままコピー）
            detect_dir = os.path.join(os.path.dirname(full_path), ".detect")
            detect_output_path = os.path.join(detect_dir, file_name)
            try:
                os.makedirs(detect_dir, exist_ok=True)
                shutil.copy(full_path, detect_output_path)
                logger.debug(f".detect/コピー完了: {detect_output_path}")
            except Exception as e:
                logger.error(f".detect/コピー失敗: {e}")
                return {
                    "status": "error",
                    "reason": f"detect_copy_failed: {e}",
                    "path": full_path,
                }

            return {
                "status": "success",
                "path": full_path,
                "output_path": detect_output_path,
                "detections": 0,
                "skip_detection": True,
                "branch_no": branch_no,
            }
        else:
            return {
                "status": "error",
                "reason": "no_backup_config",
                "path": full_path,
            }

    # バナーの判定（通常モード用）:
    # - First file (branch_no=1) のみバナー追加
    # - --force / --force-overlay は別処理
    add_banner_to_detect = is_first_image

    # 処理実行（Two-Stage: Seg + Pose）
    try:
        # ============================================================
        # --force-overlay モード: 元画像にバナーのみ上書き
        # ============================================================
        # 処理内容:
        #   - branch_no=1 のみ: 元画像にバナーのみ（マスクなし）で上書き
        #   - branch_no!=1: スキップ（処理しない）
        # 前提条件:
        #   - .backup が存在しない場合は先に作成（必須）
        # 入力:
        #   - 現在の元画像（full_path）をそのまま使用
        # 出力:
        #   - 元画像を直接上書き
        #   - .detect/ は作成しない、マスクなし
        # ============================================================
        if force_overlay:
            # branch_no=1 以外はスキップ
            if not is_first_image:
                logger.debug(f"--force-overlay: branch_no!=1 のためスキップ")
                return {
                    "status": "skip",
                    "reason": "force_overlay_not_first",
                    "path": full_path,
                }

            # バックアップがなければ作成（必須）
            if BACKUP_S3_BUCKET:
                dir_part = os.path.dirname(relative_path)
                backup_s3_key = f"webroot/{dir_part}/.backup/{file_name}"
                try:
                    if not s3_backup_exists(backup_s3_key):
                        logger.debug(
                            f"--force-overlay: バックアップ作成: s3://{BACKUP_S3_BUCKET}/{backup_s3_key}"
                        )
                        s3_upload_backup(full_path, backup_s3_key)
                except Exception as e:
                    logger.warn(f"--force-overlay: バックアップ作成失敗: {e}")
            elif BACKUP_DIR:
                backup_path = os.path.join(BACKUP_DIR, relative_path)
                if not os.path.exists(backup_path):
                    try:
                        backup_dir = os.path.dirname(backup_path)
                        os.makedirs(backup_dir, exist_ok=True)
                        shutil.copy(full_path, backup_path)
                        logger.debug(f"--force-overlay: バックアップ作成: {backup_path}")
                    except Exception as e:
                        logger.warn(f"--force-overlay: バックアップ作成失敗: {e}")

            logger.debug(
                f"--force-overlay: 元画像にバナーのみ上書き (masking=False, banner=True)"
            )
            result = process_image(
                input_path=full_path,
                output_path=full_path,
                seg_model=seg_model,
                pose_model=pose_model,
                mask_image=mask_image,
                is_masking=False,  # マスクなし
                add_banner=True,  # バナーあり
            )
            result["status"] = "success"
            result["output_path"] = full_path
            result["is_first"] = is_first_image
            result["force_overlay"] = True
            logger.debug(f"--force-overlay完了: {full_path}")
            return result

        # ============================================================
        # --force モード: .detect/ + original を強制再作成
        # ============================================================
        # 処理内容:
        #   - branch_no=1:
        #     - .detect/ にマスク+バナーで上書き
        #     - 元画像にバナーのみ（マスクなし）で上書き
        #   - branch_no!=1:
        #     - .detect/ にマスクのみ（バナー【絶対禁止】）で上書き
        #     - 元画像は変更しない
        # 入力:
        #   - .backup から元画像を取得（検出精度のため）
        #   - .backup がなければ先に作成してから使用
        # 出力:
        #   - .detect/ に処理済み画像を保存
        #   - branch_no=1: 元画像にバナーのみ版を上書き
        # 用途:
        #   - 間違ってバナーが付いた.detect/ファイルを修正
        #   - 元画像のバナーを再適用
        # 注意:
        #   - .detect/ は常にマスクあり
        #   - バナーは branch_no=1 のみ
        # ============================================================
        if force:
            # .detect/ は常にマスクあり
            # branch_no=1: マスク+バナー
            # branch_no!=1: マスクのみ（バナー【絶対禁止】）
            use_masking = True  # .detect/ は常にマスクあり
            use_banner = is_first_image  # branch_no=1 のみバナー

            logger.debug(
                f"--force: .detect/再作成 (masking={use_masking}, banner={use_banner})"
            )

            if BACKUP_S3_BUCKET:
                import tempfile

                # .backupから元画像をダウンロード（検出精度のため）
                backup_s3_key = f"webroot/{dir_part}/.backup/{file_name}"
                with tempfile.NamedTemporaryFile(
                    suffix=os.path.splitext(file_name)[1], delete=False
                ) as tmp:
                    temp_input_path = tmp.name

                if s3_backup_exists(backup_s3_key):
                    s3_download_backup(backup_s3_key, temp_input_path)
                    logger.debug(
                        f"バックアップから入力: s3://{BACKUP_S3_BUCKET}/{backup_s3_key}"
                    )
                else:
                    # バックアップがない場合は現在の画像を使用
                    logger.warn(f"バックアップなし、現在の画像を使用: {full_path}")
                    shutil.copy(full_path, temp_input_path)

                with tempfile.NamedTemporaryFile(
                    suffix=os.path.splitext(file_name)[1], delete=False
                ) as tmp:
                    temp_detect_path = tmp.name

                result = process_image(
                    input_path=temp_input_path,
                    output_path=temp_detect_path,
                    seg_model=seg_model,
                    pose_model=pose_model,
                    mask_image=mask_image,
                    is_masking=True,  # .detect/は常にマスクあり
                    add_banner=use_banner,  # branch_no=1のみバナー【それ以外は絶対禁止】
                )

                detect_s3_key = f"webroot/{dir_part}/.detect/{file_name}"
                s3_upload_backup(temp_detect_path, detect_s3_key)
                logger.debug(
                    f".detect/アップロード完了: s3://{BACKUP_S3_BUCKET}/{detect_s3_key}"
                )

                # 一時ファイル削除（.detect/処理完了）
                os.unlink(temp_input_path)
                os.unlink(temp_detect_path)

                # branch_no=1: 元ファイルにバナーのみ版を上書き（最後に実行）
                # ※ 現在のoriginal画像を使用（backupではない）
                if is_first_image:
                    logger.debug(
                        f"--force: First file - 元ファイル(現在)にバナーのみ上書き"
                    )
                    with tempfile.NamedTemporaryFile(
                        suffix=os.path.splitext(file_name)[1], delete=False
                    ) as tmp:
                        temp_banner_path = tmp.name

                    # 現在のoriginal画像をダウンロード
                    original_s3_key = f"webroot/{relative_path}"
                    s3_download_backup(original_s3_key, temp_banner_path)

                    # バナーを追加して上書き
                    process_image(
                        input_path=temp_banner_path,
                        output_path=temp_banner_path,
                        seg_model=seg_model,
                        pose_model=pose_model,
                        mask_image=mask_image,
                        is_masking=False,  # マスクなし
                        add_banner=True,  # バナーあり
                    )

                    # 元ファイルを上書き
                    s3_upload_backup(temp_banner_path, original_s3_key)
                    logger.debug(
                        f"元ファイル上書き完了: s3://{BACKUP_S3_BUCKET}/{original_s3_key}"
                    )
                    os.unlink(temp_banner_path)
                detect_output_path = f"s3://{BACKUP_S3_BUCKET}/{detect_s3_key}"
                if is_first_image:
                    original_output_path = f"s3://{BACKUP_S3_BUCKET}/{original_s3_key}"
            else:
                # ローカルバックアップから入力
                backup_path = (
                    os.path.join(BACKUP_DIR, relative_path) if BACKUP_DIR else None
                )
                if backup_path and os.path.exists(backup_path):
                    input_path = backup_path
                    logger.debug(f"バックアップから入力: {backup_path}")
                else:
                    input_path = full_path
                    logger.warn(f"バックアップなし、現在の画像を使用: {full_path}")

                detect_dir = os.path.join(os.path.dirname(full_path), ".detect")
                detect_output_path = os.path.join(detect_dir, file_name)
                os.makedirs(detect_dir, exist_ok=True)

                result = process_image(
                    input_path=input_path,
                    output_path=detect_output_path,
                    seg_model=seg_model,
                    pose_model=pose_model,
                    mask_image=mask_image,
                    is_masking=True,  # .detect/は常にマスクあり
                    add_banner=use_banner,  # branch_no=1のみバナー【それ以外は絶対禁止】
                )

                # branch_no=1: 元ファイルにバナーのみ版を上書き（最後に実行）
                # ※ 現在のoriginal画像を使用（backupではない）
                if is_first_image:
                    logger.debug(
                        f"--force: First file - 元ファイル(現在)にバナーのみ上書き"
                    )
                    process_image(
                        input_path=full_path,  # 現在のoriginalを使用
                        output_path=full_path,
                        seg_model=seg_model,
                        pose_model=pose_model,
                        mask_image=mask_image,
                        is_masking=False,  # マスクなし
                        add_banner=True,  # バナーあり
                    )
                    logger.debug(f"元ファイル上書き完了: {full_path}")

            result["status"] = "success"
            result["output_path"] = detect_output_path
            result["is_first"] = is_first_image
            result["force"] = True
            if is_first_image:
                result["original_output"] = full_path  # First fileは元ファイルも更新
            logger.debug(f"--force完了: {detect_output_path}")
            return result

        # ============================================================
        # 通常モード: 新規画像を処理
        # ============================================================
        # 処理内容:
        #   - branch_no=1: .detect/ にマスク+バナー、元画像にバナーのみ
        #   - branch_no!=1: .detect/ にマスクのみ（バナーなし）、元画像は変更しない
        # 入力:
        #   - .backup から元画像を取得（検出精度のため）
        # 出力:
        #   - .detect/ にマスク処理済み画像を保存
        #   - branch_no=1 のみ元画像にバナー追加
        # 条件:
        #   - .detect/ が既に存在する場合はスキップ
        # ============================================================
        # .detect/ファイルが既に存在するかチェック
        detect_check_path = os.path.join(
            os.path.dirname(full_path), ".detect", file_name
        )
        if os.path.exists(detect_check_path):
            logger.debug(f".detect/既存のためスキップ: {detect_check_path}")
            return {
                "status": "skip",
                "reason": "detect_exists",
                "path": full_path,
                "detect_path": detect_check_path,
            }

        logger.debug(
            f"Two-Stage推論開始: .detect/ 出力 (masking=True, banner={add_banner_to_detect})"
        )

        # S3の場合: tempファイルに出力後、boto3でアップロード
        # ローカルの場合: 直接.detect/に出力
        if BACKUP_S3_BUCKET:
            import tempfile

            # .backupから元画像をダウンロード（検出精度のため）
            backup_s3_key = f"webroot/{dir_part}/.backup/{file_name}"
            with tempfile.NamedTemporaryFile(
                suffix=os.path.splitext(file_name)[1], delete=False
            ) as tmp:
                temp_input_path = tmp.name

            if s3_backup_exists(backup_s3_key):
                s3_download_backup(backup_s3_key, temp_input_path)
                logger.debug(
                    f"バックアップから入力: s3://{BACKUP_S3_BUCKET}/{backup_s3_key}"
                )
            else:
                # バックアップがない場合は現在の画像を使用
                logger.warn(f"バックアップなし、現在の画像を使用: {full_path}")
                shutil.copy(full_path, temp_input_path)

            # 一時ファイルに出力
            with tempfile.NamedTemporaryFile(
                suffix=os.path.splitext(file_name)[1], delete=False
            ) as tmp:
                temp_detect_path = tmp.name

            result = process_image(
                input_path=temp_input_path,
                output_path=temp_detect_path,
                seg_model=seg_model,
                pose_model=pose_model,
                mask_image=mask_image,
                is_masking=True,  # マスクあり
                add_banner=add_banner_to_detect,  # First fileのみバナー
            )

            # S3にアップロード
            detect_s3_key = f"webroot/{dir_part}/.detect/{file_name}"
            s3_upload_backup(temp_detect_path, detect_s3_key)
            logger.debug(
                f".detect/アップロード完了: s3://{BACKUP_S3_BUCKET}/{detect_s3_key}"
            )

            # First fileのみ: 元ファイルにバナーのみ版を上書き
            # ※ 現在のoriginal画像を使用（--forceと同じロジック）
            if is_first_image:
                logger.debug(f"First file: 元ファイル(現在)にバナーのみ上書き")
                with tempfile.NamedTemporaryFile(
                    suffix=os.path.splitext(file_name)[1], delete=False
                ) as tmp:
                    temp_banner_path = tmp.name

                # 現在のoriginal画像をダウンロード（--forceと同じ）
                original_s3_key = f"webroot/{relative_path}"
                s3_download_backup(original_s3_key, temp_banner_path)

                # バナーを追加
                process_image(
                    input_path=temp_banner_path,
                    output_path=temp_banner_path,
                    seg_model=seg_model,
                    pose_model=pose_model,
                    mask_image=mask_image,
                    is_masking=False,  # マスクなし
                    add_banner=True,  # バナーあり
                )

                # boto3でS3にアップロード
                s3_upload_backup(temp_banner_path, original_s3_key)
                logger.debug(f"元ファイル上書き完了: s3://{BACKUP_S3_BUCKET}/{original_s3_key}")
                os.unlink(temp_banner_path)

            # 一時ファイル削除
            os.unlink(temp_input_path)
            os.unlink(temp_detect_path)

            detect_output_path = f"s3://{BACKUP_S3_BUCKET}/{detect_s3_key}"
        else:
            # ローカルバックアップから入力
            backup_path = (
                os.path.join(BACKUP_DIR, relative_path) if BACKUP_DIR else None
            )
            if backup_path and os.path.exists(backup_path):
                input_path = backup_path
                logger.debug(f"バックアップから入力: {backup_path}")
            else:
                input_path = full_path
                logger.warn(f"バックアップなし、現在の画像を使用: {full_path}")

            # ローカルの場合は直接出力
            detect_dir = os.path.join(os.path.dirname(full_path), ".detect")
            detect_output_path = os.path.join(detect_dir, file_name)
            os.makedirs(detect_dir, exist_ok=True)

            result = process_image(
                input_path=input_path,
                output_path=detect_output_path,
                seg_model=seg_model,
                pose_model=pose_model,
                mask_image=mask_image,
                is_masking=True,  # マスクあり
                add_banner=add_banner_to_detect,  # First fileのみバナー
            )

            # First fileのみ: 元ファイルにバナーのみ版を上書き
            # ※ 現在のoriginal画像を使用（--forceと同じロジック）
            if is_first_image:
                logger.debug(f"First file: 元ファイル(現在)にバナーのみ上書き")
                process_image(
                    input_path=full_path,  # 現在のoriginal画像
                    output_path=full_path,
                    seg_model=seg_model,
                    pose_model=pose_model,
                    mask_image=mask_image,
                    is_masking=False,  # マスクなし
                    add_banner=True,  # バナーあり
                )

        result["status"] = "success"
        result["output_path"] = detect_output_path
        result["is_first"] = is_first_image
        if is_first_image:
            result["original_output"] = full_path  # First fileは元ファイルも更新

        # バックアップパス情報
        if BACKUP_S3_BUCKET:
            dir_part = os.path.dirname(relative_path)
            result["backup_path"] = (
                f"s3://{BACKUP_S3_BUCKET}/webroot/{dir_part}/.backup/{file_name}"
            )
        elif BACKUP_DIR:
            result["backup_path"] = os.path.join(BACKUP_DIR, relative_path)

        logger.debug(
            f"処理完了: 検出数={result.get('detections', 0)}, "
            f".detect={detect_output_path}"
            + (f", original={full_path}" if is_first_image else "")
        )
        return result
    except Exception as e:
        logger.error(f"処理失敗: {e}")
        return {"status": "error", "reason": str(e), "path": full_path}


# ======================
# ログローテーション
# ======================
def cleanup_old_logs(log_dir: Path, retention_days: int = 60):
    """古いログファイルを削除"""
    if not log_dir.exists():
        return

    cutoff_date = datetime.now() - timedelta(days=retention_days)
    deleted_count = 0

    # process.log.YYYYMMDD 形式のローテーションログ
    for file_path in log_dir.glob("process.log.*"):
        try:
            if file_path.stat().st_mtime < cutoff_date.timestamp():
                file_path.unlink()
                deleted_count += 1
                logger.debug(f"ログファイル削除: {file_path.name}")
        except OSError:
            pass

    # 古いログファイル（*.log）
    for file_path in log_dir.glob("*.log"):
        if file_path.name == "process.log":
            continue  # 現在のログはスキップ
        try:
            if file_path.stat().st_mtime < cutoff_date.timestamp():
                file_path.unlink()
                deleted_count += 1
                logger.debug(f"ログファイル削除: {file_path.name}")
        except OSError:
            pass

    if deleted_count > 0:
        logger.info(
            f"ログローテーション: {deleted_count}件削除 ({retention_days}日以前)"
        )


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
  python fetch_today_images.py --date 2026-02-03 # 特定の日付を指定
  python fetch_today_images.py --limit 50        # 最大50件処理
  python fetch_today_images.py --days-ago 7 --limit 100
  python fetch_today_images.py --path /1554913G  # 特定フォルダを直接処理（DBバイパス）
  python fetch_today_images.py --path 1554913G   # 先頭/は省略可
        """,
    )
    parser.add_argument(
        "--days-ago",
        type=int,
        default=0,
        help="何日前の画像を処理するか (0=今日, 1=昨日, ...) [デフォルト: 0]",
    )
    parser.add_argument(
        "--date",
        type=str,
        default=None,
        help="処理対象日を指定 (YYYY-MM-DD形式、--days-agoより優先)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="1回の実行で処理する最大車両数 [デフォルト: 10]",
    )
    parser.add_argument(
        "--path",
        type=str,
        default=None,
        help="特定フォルダのみ処理 (例: /1554913G または 1554913G) - DBをバイパス",
    )
    parser.add_argument(
        "--force-overlay",
        action="store_true",
        help="全画像にバナーを強制適用 (デフォルト: 先頭画像のみ)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help=".detect/とoriginal(branch_no=1)を強制的に再処理",
    )
    parser.add_argument(
        "--order",
        type=str,
        choices=["newest", "oldest"],
        default="newest",
        help="処理順序: newest=新しい順, oldest=古い順 [デフォルト: newest]",
    )

    # 引数を解析（不正なオプションはエラー）
    args, unknown = parser.parse_known_args()
    if unknown:
        logger.error(f"不正なオプション: {unknown}")
        logger.error("有効なオプション: --days-ago, --date, --limit, --path, --force, --force-overlay, --order")
        return

    # 対象日を計算
    if args.date:
        try:
            target_date = datetime.strptime(args.date, "%Y-%m-%d").date()
        except ValueError:
            logger.error(f"日付形式が不正: {args.date} (YYYY-MM-DD形式で指定)")
            return
    else:
        target_date = datetime.now().date() - timedelta(days=args.days_ago)

    # バックアップ先
    if BACKUP_S3_BUCKET:
        backup_location = f"s3://{BACKUP_S3_BUCKET}/webroot/.../.backup/ (boto3)"
    elif BACKUP_DIR:
        backup_location = f"{BACKUP_DIR} (ローカル)"
    else:
        backup_location = "未設定（エラー）"

    order_display = "新しい順" if args.order == "newest" else "古い順"
    logger.info("=" * 60)
    logger.info(f"バッチ処理開始 (Two-Stage)")
    if args.path:
        logger.info(f"  対象フォルダ: {args.path}")
    else:
        logger.info(f"  対象日: {target_date}")
    logger.info(f"  最大処理数: {args.limit}件")
    logger.info(f"  処理順序: {order_display} (--order {args.order})")
    logger.info(f"  S3マウント: {S3_MOUNT}")
    logger.info(f"  バックアップ: {backup_location}")
    logger.info(f"  Segモデル: {SEG_MODEL_PATH}")
    logger.info(f"  Poseモデル: {POSE_MODEL_PATH}")
    logger.info("=" * 60)

    # 設定検証
    if not validate_config():
        return

    # モデル読み込み（初回のみ）
    try:
        load_models()
    except Exception as e:
        logger.error(f"モデル読み込み失敗: {e}")
        return

    # ログローテーション（60日以前を削除）
    tracker.cleanup_old_files(LOG_RETENTION_DAYS)
    cleanup_old_logs(LOG_DIR, LOG_RETENTION_DAYS)

    # --path モードか通常モードかで処理を分岐
    if args.path:
        # フォルダ直接指定モード（DBバイパス）
        logger.info(f"フォルダ直接モード: {args.path}")
        images = get_images_from_path(args.path)

        if not images:
            logger.info(f"フォルダに画像がありません: {args.path}")
            return
    else:
        # 通常モード（DB取得）
        # 既存のトラッキング統計
        existing_stats = tracker.get_stats(target_date)
        logger.info(
            f"トラッキング状況: 処理済み {existing_stats['total']}件 "
            f"(成功: {existing_stats['success']}, エラー: {existing_stats['error']})"
        )

        # 常に全件取得し、トラッキングで処理済みをスキップする
        # ※ 増分取得は未処理ファイルを見逃すため廃止
        last_fetch_time = None
        logger.info("全件取得モード（トラッキングで処理済みをスキップ）")

        # 画像を取得
        images = get_images_by_date(
            target_date=target_date,
            last_fetch_time=last_fetch_time,
            only_first=args.force_overlay,  # --force-overlayの場合はbranch_no=1のみ
            order=args.order,  # newest=新しい順, oldest=古い順
        )

        if not images:
            logger.info(f"{target_date} の画像はありません")
            return

        logger.info(f"DB取得: {len(images)}件")

    # 車両ごとにグループ化
    car_images = {}
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

        car_key = inspresultdata_cd if inspresultdata_cd else str(car_cd)

        if car_key not in car_images:
            car_images[car_key] = []

        car_images[car_key].append(
            {
                "id": file_id,
                "branch_no": branch_no,
                "path": save_file_name,
                "created": created,
                "modified": modified,
            }
        )

    total_cars = len(car_images)
    cars_to_process = min(total_cars, args.limit)
    logger.info(f"車両数: {total_cars}台 (処理予定: {cars_to_process}台)")

    # 処理カウンター
    stats = {
        "success": 0,
        "skip_tracked": 0,
        "skip_other": 0,
        "error": 0,
        "verified": 0,
        "pending": 0,
    }
    processed_cars = 0  # 処理した車両数（limitはこれで判定）

    # 車両ごとの結果（Chatwork通知用）
    car_results = []  # [(car_id, success, error, detections), ...]

    # トラッキングモードフラグ
    # --force-overlayのみトラッキングスキップ（.detect/を作成しないため）
    # --path, --limit, --days-ago, --date はフィルタのみ、ロジックに影響しない
    use_tracking = not args.force_overlay

    # 各車両を処理
    for car_key, car_files in car_images.items():
        # limit到達チェック（車両数で判定）
        if processed_cars >= args.limit:
            logger.info(f"処理上限到達: {args.limit}台")
            break

        # branch_noでソート
        car_files.sort(key=lambda x: x["branch_no"] or 999)

        # この車両が既に一部処理済みかチェック（パスパターンで判定）
        # 車両のフォルダパスを取得: /upfile/1041/8430/
        first_file_path = car_files[0]["path"]
        car_dir = os.path.dirname(first_file_path) + "/"  # /upfile/1041/8430/

        # --forceモードでない場合、全ファイルがdone状態ならスキップ
        if use_tracking and not args.force:
            if tracker.has_car_all_done(target_date, car_dir):
                stats["skip_tracked"] += len(car_files)
                logger.debug(f"車両スキップ（全完了）: {car_key} (フォルダ: {car_dir})")
                continue

        logger.debug(f"車両処理開始: {car_key} ({len(car_files)}枚)")

        # 車両ごとの統計
        car_success = 0
        car_error = 0
        car_detections = 0
        car_processed_files = []  # 処理成功した画像のリスト [(branch_no, path), ...]
        car_verified_count = 0  # verified状態のファイル数

        # ============================================================
        # Phase 1: 全ファイルをpending状態として登録
        # ============================================================
        if use_tracking:
            for file_info in car_files:
                is_first = file_info["branch_no"] == 1
                # 既にdone状態ならスキップ（--forceの場合は再処理）
                if not args.force and tracker.is_done(target_date, file_info["id"]):
                    continue
                tracker.mark_pending(
                    target_date=target_date,
                    file_id=file_info["id"],
                    path=file_info["path"],
                    branch_no=file_info["branch_no"],
                    car_id=car_key,
                    is_first=is_first,
                )

        # ============================================================
        # Phase 2: 各ファイルを処理
        # ============================================================
        for idx, file_info in enumerate(car_files):
            file_id = file_info["id"]

            # branch_no == 1 のみ first file として扱う
            is_first = file_info["branch_no"] == 1
            logger.debug(
                f"branch_no={file_info['branch_no']} (type={type(file_info['branch_no']).__name__}), is_first={is_first}"
            )

            # 既にdone状態ならスキップ（--forceの場合は再処理）
            if use_tracking and not args.force:
                if tracker.is_done(target_date, file_id):
                    logger.debug(f"スキップ（done状態）: {file_info['path']}")
                    stats["skip_tracked"] += 1
                    continue

            # processing状態にマーク
            if use_tracking:
                tracker.mark_processing(target_date, file_id)

            # 処理実行
            result = backup_and_process(
                file_path=file_info["path"],
                is_first_image=is_first,
                force_overlay=args.force_overlay,
                force=args.force,
                branch_no=file_info["branch_no"],
            )

            status = result.get("status", "error")

            if status == "success":
                stats["success"] += 1
                car_success += 1
                car_detections += result.get("detections", 0)

                # ============================================================
                # Phase 3: 出力ファイルを検証
                # ============================================================
                # --force-overlay は .detect/ を作成しないため、検証スキップ
                # skip_detection (branch_no 30,31,32) は直接コピーのため、検証スキップ
                if args.force_overlay:
                    car_verified_count += 1
                    stats["verified"] += 1
                    car_processed_files.append(
                        (file_info["branch_no"] or 999, file_info["path"])
                    )
                    logger.success(
                        f"{file_info['path']} "
                        f"(--force-overlay: バナー追加完了)"
                    )
                elif result.get("skip_detection"):
                    # skip_detection: 直接コピーしたので検証不要
                    car_verified_count += 1
                    stats["verified"] += 1
                    car_processed_files.append(
                        (file_info["branch_no"] or 999, file_info["path"])
                    )
                    if use_tracking:
                        tracker.mark_verified(
                            target_date=target_date,
                            file_id=file_id,
                            detections=0,
                            output_paths={"detect": result.get("output_path", "")},
                        )
                    logger.success(
                        f"{file_info['path']} "
                        f"(skip_detection: branch_no={result.get('branch_no')}, コピー完了)"
                    )
                else:
                    verify_result = verify_output_exists(
                        file_path=file_info["path"],
                        is_first_image=is_first,
                    )

                    if verify_result["verified"]:
                        car_verified_count += 1
                        stats["verified"] += 1

                        # verified状態にマーク
                        if use_tracking:
                            output_paths = {
                                "detect": verify_result["detect_path"],
                            }
                            if is_first:
                                output_paths["original"] = verify_result["original_path"]

                            tracker.mark_verified(
                                target_date=target_date,
                                file_id=file_id,
                                detections=result.get("detections", 0),
                                output_paths=output_paths,
                            )

                        # 処理成功した画像を記録
                        car_processed_files.append(
                            (file_info["branch_no"] or 999, file_info["path"])
                        )

                        logger.success(
                            f"{file_info['path']} "
                            f"(検出: {result.get('detections', 0)}, "
                            f"バナー: {'あり' if is_first else 'なし'}, "
                            f"verified: ✓)"
                        )
                    else:
                        # 処理成功したが出力ファイルがない（エラー扱い）
                        stats["error"] += 1
                        car_error += 1
                        if use_tracking:
                            tracker.mark_error(
                                target_date=target_date,
                                file_id=file_id,
                                error_reason=f"output_missing: {verify_result['missing']}",
                            )
                        logger.error(
                            f"{file_info['path']} - 出力ファイル未検出: {verify_result['missing']}"
                        )

            elif status == "skip":
                # --force-overlay で branch_no != 1 の場合などスキップ
                stats["skip_other"] += 1
                logger.debug(
                    f"スキップ: {file_info['path']} - {result.get('reason', 'skip')}"
                )

            elif status == "error":
                stats["error"] += 1
                car_error += 1

                # エラー状態にマーク
                if use_tracking:
                    tracker.mark_error(
                        target_date=target_date,
                        file_id=file_id,
                        error_reason=result.get("reason", "unknown"),
                    )

                logger.error(f"{file_info['path']} - {result.get('reason', 'unknown')}")

            else:  # skip
                stats["skip_other"] += 1
                logger.debug(
                    f"スキップ: {file_info['path']} - {result.get('reason', '')}"
                )

        # ============================================================
        # Phase 4: 車両の全ファイルがverifiedなら、doneに昇格
        # ============================================================
        if use_tracking and car_verified_count == len(car_files):
            tracker.mark_car_done(target_date, car_dir)
            logger.info(f"車両完了（全ファイルdone）: {car_key}")

        # 車両処理完了後、結果を記録（処理があった場合のみ）
        if car_success > 0 or car_error > 0:
            processed_cars += 1  # 車両カウント
            logger.info(
                f"[{processed_cars}/{cars_to_process}台] {car_key}: "
                f"{car_success}枚成功, {car_error}枚エラー, "
                f"検出{car_detections}件, verified: {car_verified_count}/{len(car_files)}"
            )
            # branch_noでソートしてから記録
            car_processed_files.sort(key=lambda x: x[0])
            car_results.append(
                (car_key, car_success, car_error, car_detections, car_processed_files)
            )

    # 最終統計
    logger.info("=" * 60)
    logger.info("処理完了")
    logger.info(f"  成功: {stats['success']}件")
    logger.info(f"  verified: {stats['verified']}件")
    logger.info(f"  エラー: {stats['error']}件")
    logger.info(f"  スキップ（処理済み）: {stats['skip_tracked']}件")
    logger.info(f"  スキップ（その他）: {stats['skip_other']}件")

    # --path モードではトラッキング更新をスキップ
    if use_tracking:
        # 最終トラッキング統計
        final_stats = tracker.get_stats(target_date)
        logger.info(
            f"トラッキング累計: {final_stats['total']}件 "
            f"(pending: {final_stats['pending']}, processing: {final_stats['processing']}, "
            f"verified: {final_stats['verified']}, done: {final_stats['done']}, "
            f"error: {final_stats['error']})"
        )

        # last_fetch_timeを更新（次回は新規ファイルのみ取得）
        tracker.set_last_processed_time(target_date, datetime.now())
        logger.info(f"次回増分取得: {datetime.now().strftime('%H:%M:%S')} 以降")
    logger.info("=" * 60)

    # Chatwork通知（処理があった場合のみ）
    if CHATWORK_API_KEY and CHATWORK_ROOM_ID and car_results:
        message = build_processing_summary(target_date, stats, car_results)
        if send_chatwork_notification(message):
            logger.info("Chatwork通知送信完了")
        else:
            logger.warn("Chatwork通知送信失敗")


if __name__ == "__main__":
    main()
