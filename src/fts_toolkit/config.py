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
    ML_TEST_SIZE = float(os.getenv('ML_TEST_SIZE', '0.3'))  # For train/test spl

    #PCA Sttings
    PCA_N_COMPONENTS_CONFIG = os.getenv('PCA_N_COMPONENTS', '0.95')

    # Logging
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    DEBUG = os.getenv('DEBUG', 'False').lower() == 'true'

    # Random Forest Hyperparameters
    RF_N_ESTIMATORS = int(os.getenv('RF_N_ESTIMATORS', '100'))
    RF_MAX_DEPTH = int(os.getenv('RF_MAX_DEPTH', '10'))  # You already use 10 in main.py

    # SVR Hyperparameters
    SVR_KERNEL = os.getenv('SVR_KERNEL', 'rbf')
    SVR_C = float(os.getenv('SVR_C', '1.0'))
    SVR_EPSILON = float(os.getenv('SVR_EPSILON', '0.1'))
    SVR_GAMMA = os.getenv('SVR_GAMMA', 'scale')  # 'scale', 'auto', or a float

    # XGBoost Hyperparameters
    XGB_N_ESTIMATORS = int(os.getenv('XGB_N_ESTIMATORS', '100'))
    XGB_LEARNING_RATE = float(os.getenv('XGB_LEARNING_RATE', '0.1'))
    XGB_MAX_DEPTH = int(os.getenv('XGB_MAX_DEPTH', '3'))  # You use 3 in main.py
    XGB_EARLY_STOPPING_ROUNDS = int(os.getenv('XGB_EARLY_STOPPING_ROUNDS', '10'))

    # LSTM Hyperparameters
    LSTM_UNITS = int(os.getenv('LSTM_UNITS', '50'))
    LSTM_DROPOUT_RATE = float(os.getenv('LSTM_DROPOUT_RATE', '0.2'))
    LSTM_EPOCHS = int(os.getenv('LSTM_EPOCHS', '50'))
    LSTM_BATCH_SIZE = int(os.getenv('LSTM_BATCH_SIZE', '32'))
    LSTM_VALIDATION_SPLIT = float(os.getenv('LSTM_VALIDATION_SPLIT', '0.1'))
    LSTM_EARLY_STOPPING_PATIENCE = int(os.getenv('LSTM_EARLY_STOPPING_PATIENCE', '10'))

    # KerasTuner Specific Settings for LSTM
    LSTM_TUNER_MAX_TRIALS = int(os.getenv('LSTM_TUNER_MAX_TRIALS', '10')) # Number of hyperparameter combinations to test
    LSTM_TUNER_EXECUTIONS_PER_TRIAL = int(os.getenv('LSTM_TUNER_EXECUTIONS_PER_TRIAL', '2')) # Number of models to train per trial for robustness
    LSTM_TUNER_EPOCHS = int(os.getenv('LSTM_TUNER_EPOCHS', '20')) # Max epochs to train *each model variation* during the search
    LSTM_TUNER_DIR = os.getenv('LSTM_TUNER_DIR', 'keras_tuner_lstm') # Directory to store tuning results
    LSTM_TUNER_PROJECT_NAME = os.getenv('LSTM_TUNER_PROJECT_NAME', 'lstm_hyperparameter_tuning')

    # --- NEW: GRU Hyperparameters ---
    GRU_UNITS = int(os.getenv('GRU_UNITS', '50')) # Similar default to LSTM
    GRU_DROPOUT_RATE = float(os.getenv('GRU_DROPOUT_RATE', '0.2'))
    GRU_EPOCHS = int(os.getenv('GRU_EPOCHS', '50'))
    GRU_BATCH_SIZE = int(os.getenv('GRU_BATCH_SIZE', '32'))
    GRU_VALIDATION_SPLIT = float(os.getenv('GRU_VALIDATION_SPLIT', '0.1'))
    GRU_EARLY_STOPPING_PATIENCE = int(os.getenv('GRU_EARLY_STOPPING_PATIENCE', '10'))

    # KerasTuner Specific Settings for GRU
    GRU_TUNER_MAX_TRIALS = int(os.getenv('GRU_TUNER_MAX_TRIALS', '10'))
    GRU_TUNER_EXECUTIONS_PER_TRIAL = int(os.getenv('GRU_TUNER_EXECUTIONS_PER_TRIAL', '2'))
    GRU_TUNER_EPOCHS = int(os.getenv('GRU_TUNER_EPOCHS', '20'))  # Epochs per trial during search
    GRU_TUNER_DIR = os.getenv('GRU_TUNER_DIR', 'keras_tuner_gru')
    GRU_TUNER_PROJECT_NAME = os.getenv('GRU_TUNER_PROJECT_NAME', 'gru_hyperparameter_tuning')

    LSTM_L1_UNITS = int(os.getenv('LSTM_L1_UNITS', '64'))
    LSTM_L2_UNITS = int(os.getenv('LSTM_L2_UNITS', '32'))

    # VAR Model Settings
    VAR_SYMBOLS_STR = os.getenv('VAR_SYMBOLS', 'HKDKRW=X,HKDJPY=X,HKDCNY=X') # Comma-separated string
    VAR_MAX_LAGS = int(os.getenv('VAR_MAX_LAGS', '12')) # Max lags to check for order selection
    VAR_FORECAST_STEPS = int(os.getenv('VAR_FORECAST_STEPS', '5'))
    VAR_IRF_STEPS = int(os.getenv('VAR_IRF_STEPS', '10'))
    VAR_STATIONARITY_METHOD = os.getenv('VAR_STATIONARITY_METHOD', 'diff') # 'diff' or 'log_returns'

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
        print(f"   ML Test Size: {cls.ML_TEST_SIZE}")
        print(f"   PCA Components/Variance: {cls.PCA_N_COMPONENTS_CONFIG}")
        print(f"   VAR Symbols: {cls.VAR_SYMBOLS_STR}")
# Create global config instance
config = Config()
