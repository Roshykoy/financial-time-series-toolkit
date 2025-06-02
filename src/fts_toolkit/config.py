"""Simple configuration for Financial Time Series Toolkit"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Config:
    """Application configuration loaded from environment variables"""

    # Project paths
    PROJECT_ROOT = Path(__file__).parent.parent.parent
    DATA_DIR = Path(os.getenv('DATA_DIR', PROJECT_ROOT / 'data'))
    RAW_DATA_DIR = DATA_DIR / 'raw'
    PROCESSED_DATA_DIR = DATA_DIR / 'processed'

    # Data settings
    DEFAULT_SYMBOL = os.getenv('DEFAULT_SYMBOL', 'EURUSD=X')
    LOOKBACK_DAYS = int(os.getenv('LOOKBACK_DAYS', '730'))  # Keep this at 730 or your desired longer period
    WINDOW_SIZE = int(os.getenv('WINDOW_SIZE', '60'))

    # Scraping settings
    REQUEST_DELAY = float(os.getenv('REQUEST_DELAY', '2'))
    MAX_RETRIES = int(os.getenv('MAX_RETRIES', '3'))

    # Model settings
    RANDOM_SEED = int(os.getenv('RANDOM_SEED', '42'))

    # Logging
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    DEBUG = os.getenv('DEBUG', 'False').lower() == 'true'

    @classmethod
    def setup_directories(cls):
        """Create necessary directories if they don't exist"""
        cls.RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
        cls.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
        print(f"✅ Directories ready: {cls.DATA_DIR}")

    @classmethod
    def print_config(cls):
        """Print current configuration"""
        print("🔧 Current Configuration:")
        print(f"   Symbol: {cls.DEFAULT_SYMBOL}")
        print(f"   Data Directory: {cls.DATA_DIR}")
        print(f"   Lookback Days: {cls.LOOKBACK_DAYS}")
        print(f"   Window Size: {cls.WINDOW_SIZE}")
        print(f"   Random Seed: {cls.RANDOM_SEED}")


# Create global config instance
config = Config()
