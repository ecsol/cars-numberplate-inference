"""
Configuration management for the plate detection system.

設定管理モジュール
"""

from pathlib import Path
from typing import Optional
import os


# プロジェクトルートディレクトリ
PROJECT_ROOT = Path(__file__).parent.parent.parent


class Settings:
    """
    アプリケーション設定クラス
    
    環境変数から設定を読み込み、デフォルト値を提供する。
    """
    
    def __init__(self):
        # API設定
        self.api_host: str = os.getenv("API_HOST", "0.0.0.0")
        self.api_port: int = int(os.getenv("API_PORT", "8000"))
        self.debug: bool = os.getenv("DEBUG", "false").lower() == "true"
        
        # モデル設定
        self.model_path: Path = PROJECT_ROOT / os.getenv(
            "MODEL_PATH", "models/best.pt"
        )
        self.confidence_threshold: float = float(
            os.getenv("CONFIDENCE_THRESHOLD", "0.1")
        )
        self.device: str = os.getenv("DEVICE", "mps")
        
        # アップロード制限
        self.max_file_size_mb: int = int(os.getenv("MAX_FILE_SIZE_MB", "10"))
    
    @property
    def max_file_size_bytes(self) -> int:
        """最大ファイルサイズをバイト単位で返す"""
        return self.max_file_size_mb * 1024 * 1024


# シングルトンインスタンス
settings = Settings()
