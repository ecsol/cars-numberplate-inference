#!/usr/bin/env python3
"""
バックアップから画像を復元するスクリプト

トラッキングファイルを読み込み、処理済み画像をバックアップから復元する。
S3バックアップまたはローカルバックアップから復元可能。

使用例:
  python restore_from_backup.py                      # 今日の処理済み画像を復元
  python restore_from_backup.py --date 2026-02-03   # 特定の日付
  python restore_from_backup.py --dry-run           # 実際に復元せず確認のみ
  python restore_from_backup.py --car-id 10418430   # 特定の車両のみ復元
"""

import argparse
import json
import os
import shutil
import sys
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

# boto3 deprecation warning を抑制
warnings.filterwarnings("ignore", message=".*Boto3 will no longer support Python 3.9.*")

import boto3
from botocore.exceptions import ClientError
from dotenv import load_dotenv

# 環境変数読み込み
env_file = Path(__file__).parent.parent / ".env.production"
if env_file.exists():
    load_dotenv(env_file)
else:
    load_dotenv()

# プロジェクトルート
PROJECT_DIR = Path(__file__).parent.parent

# ログディレクトリ
_log_dir_env = os.getenv("LOG_DIR", "")
if _log_dir_env:
    if os.path.isabs(_log_dir_env):
        LOG_DIR = Path(_log_dir_env)
    else:
        LOG_DIR = PROJECT_DIR / _log_dir_env
else:
    LOG_DIR = PROJECT_DIR / "logs"

# トラッキングディレクトリ（logs/tracking/）
TRACKING_DIR = LOG_DIR / "tracking"

# S3マウントポイント
S3_MOUNT = os.getenv("S3_MOUNT", "")

# バックアップ設定
BACKUP_S3_BUCKET = os.getenv("BACKUP_S3_BUCKET", "")
BACKUP_S3_PREFIX = os.getenv("BACKUP_S3_PREFIX", ".backup")
BACKUP_DIR = os.getenv("BACKUP_DIR", "")

# boto3 S3 client (lazy init)
_s3_client = None


class Logger:
    """シンプルなロガー"""

    def info(self, msg):
        print(f"[INFO] {msg}")

    def error(self, msg):
        print(f"[ERROR] {msg}", file=sys.stderr)

    def warn(self, msg):
        print(f"[WARN] {msg}")

    def ok(self, msg):
        print(f"[OK] {msg}")

    def debug(self, msg):
        print(f"[DEBUG] {msg}")


logger = Logger()


def get_s3_client():
    """S3クライアントを取得"""
    global _s3_client
    if _s3_client is None:
        _s3_client = boto3.client("s3")
    return _s3_client


def get_tracking_file(target_date: datetime.date) -> Path:
    """トラッキングファイルパスを取得"""
    return TRACKING_DIR / f"processed_{target_date.strftime('%Y%m%d')}.json"


def load_tracking_data(target_date: datetime.date) -> dict:
    """トラッキングデータを読み込み"""
    tracking_file = get_tracking_file(target_date)

    if not tracking_file.exists():
        logger.error(f"トラッキングファイルが見つかりません: {tracking_file}")
        return {}

    try:
        with open(tracking_file, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"トラッキングファイル読み込み失敗: {e}")
        return {}


def get_backup_s3_key(relative_path: str) -> str:
    """
    相対パスからS3バックアップキーを生成
    例: /upfile/1041/8430/xxx.jpg -> webroot/upfile/1041/8430/.backup/xxx.jpg
    """
    # 先頭の/を削除
    clean_path = relative_path.lstrip("/")
    # ディレクトリとファイル名を分離
    dir_path = os.path.dirname(clean_path)
    filename = os.path.basename(clean_path)
    # .backupをディレクトリ内に配置
    return f"webroot/{dir_path}/{BACKUP_S3_PREFIX}/{filename}"


def get_local_backup_path(relative_path: str) -> str:
    """相対パスからローカルバックアップパスを生成"""
    clean_path = relative_path.lstrip("/")
    return os.path.join(BACKUP_DIR, clean_path)


def restore_from_s3(relative_path: str, dry_run: bool = False) -> bool:
    """S3バックアップから復元"""
    s3_key = get_backup_s3_key(relative_path)
    target_path = os.path.join(S3_MOUNT, relative_path.lstrip("/"))

    if dry_run:
        logger.info(f"[DRY-RUN] 復元: s3://{BACKUP_S3_BUCKET}/{s3_key} -> {target_path}")
        return True

    try:
        # S3からバックアップを取得
        response = get_s3_client().get_object(Bucket=BACKUP_S3_BUCKET, Key=s3_key)
        data = response["Body"].read()

        # 元の場所に書き込み
        with open(target_path, "wb") as f:
            f.write(data)

        logger.ok(f"復元完了: {relative_path}")
        return True

    except ClientError as e:
        if e.response["Error"]["Code"] == "NoSuchKey":
            logger.error(f"バックアップが見つかりません: s3://{BACKUP_S3_BUCKET}/{s3_key}")
        else:
            logger.error(f"S3エラー: {e}")
        return False
    except Exception as e:
        logger.error(f"復元失敗: {relative_path} - {e}")
        return False


def restore_from_local(relative_path: str, dry_run: bool = False) -> bool:
    """ローカルバックアップから復元"""
    backup_path = get_local_backup_path(relative_path)
    target_path = os.path.join(S3_MOUNT, relative_path.lstrip("/"))

    if not os.path.exists(backup_path):
        logger.error(f"バックアップが見つかりません: {backup_path}")
        return False

    if dry_run:
        logger.info(f"[DRY-RUN] 復元: {backup_path} -> {target_path}")
        return True

    try:
        shutil.copy(backup_path, target_path)
        logger.ok(f"復元完了: {relative_path}")
        return True
    except Exception as e:
        logger.error(f"復元失敗: {relative_path} - {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="バックアップから画像を復元",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  python restore_from_backup.py                      # 今日の処理済み画像を復元
  python restore_from_backup.py --date 2026-02-03   # 特定の日付
  python restore_from_backup.py --dry-run           # 実際に復元せず確認のみ
  python restore_from_backup.py --car-id 10418430   # 特定の車両のみ復元
  python restore_from_backup.py --status success    # 成功したファイルのみ復元
        """,
    )
    parser.add_argument(
        "--date",
        type=str,
        default=None,
        help="対象日 (YYYY-MM-DD形式)",
    )
    parser.add_argument(
        "--days-ago",
        type=int,
        default=0,
        help="何日前のデータを復元するか [デフォルト: 0=今日]",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="実際に復元せず、対象ファイルを表示するのみ",
    )
    parser.add_argument(
        "--car-id",
        type=str,
        default=None,
        help="特定の車両IDのみ復元 (例: 10418430)",
    )
    parser.add_argument(
        "--status",
        type=str,
        choices=["success", "error", "all"],
        default="all",
        help="復元対象のステータス [デフォルト: all]",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="復元する最大ファイル数",
    )

    args = parser.parse_args()

    # 対象日を決定
    if args.date:
        try:
            target_date = datetime.strptime(args.date, "%Y-%m-%d").date()
        except ValueError:
            logger.error(f"日付形式が不正: {args.date} (YYYY-MM-DD形式で指定)")
            return
    else:
        target_date = datetime.now().date() - timedelta(days=args.days_ago)

    logger.info("=" * 60)
    logger.info("バックアップ復元")
    logger.info(f"対象日: {target_date}")
    logger.info(f"S3マウント: {S3_MOUNT}")

    if BACKUP_S3_BUCKET:
        logger.info(f"バックアップ元: s3://{BACKUP_S3_BUCKET} (boto3)")
        restore_func = restore_from_s3
    elif BACKUP_DIR:
        logger.info(f"バックアップ元: {BACKUP_DIR} (ローカル)")
        restore_func = restore_from_local
    else:
        logger.error("BACKUP_S3_BUCKET または BACKUP_DIR を設定してください")
        return

    if args.dry_run:
        logger.warn("DRY-RUNモード: 実際の復元は行いません")
    if args.car_id:
        logger.info(f"車両ID指定: {args.car_id}")
    if args.status != "all":
        logger.info(f"ステータス指定: {args.status}")
    logger.info("=" * 60)

    # トラッキングデータを読み込み
    data = load_tracking_data(target_date)
    if not data:
        return

    processed = data.get("processed", {})
    if not processed:
        logger.info("処理済みファイルがありません")
        return

    logger.info(f"トラッキング: {len(processed)}件の処理済みファイル")

    # フィルタリング
    files_to_restore = []
    for file_id, record in processed.items():
        path = record.get("path", "")
        status = record.get("status", "")

        # ステータスフィルタ
        if args.status != "all" and status != args.status:
            continue

        # 車両IDフィルタ
        if args.car_id:
            # パスから車両IDを抽出 (例: /upfile/1041/8430/xxx.jpg -> 10418430)
            parts = path.split("/")
            if len(parts) >= 4:
                car_id_from_path = parts[2] + parts[3]
                if car_id_from_path != args.car_id:
                    continue

        files_to_restore.append({
            "file_id": file_id,
            "path": path,
            "status": status,
        })

    if not files_to_restore:
        logger.info("復元対象のファイルがありません")
        return

    # リミット適用
    if args.limit and len(files_to_restore) > args.limit:
        logger.info(f"リミット適用: {len(files_to_restore)} -> {args.limit}件")
        files_to_restore = files_to_restore[:args.limit]

    logger.info(f"復元対象: {len(files_to_restore)}件")
    logger.info("-" * 40)

    # 復元実行
    stats = {"success": 0, "error": 0}

    for i, file_info in enumerate(files_to_restore, 1):
        path = file_info["path"]
        logger.debug(f"[{i}/{len(files_to_restore)}] {path}")

        if restore_func(path, dry_run=args.dry_run):
            stats["success"] += 1
        else:
            stats["error"] += 1

    # 結果
    logger.info("=" * 60)
    logger.info("復元完了")
    logger.info(f"  成功: {stats['success']}件")
    logger.info(f"  エラー: {stats['error']}件")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
