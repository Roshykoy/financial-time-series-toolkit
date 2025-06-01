"""
Data Processing for Financial Time Series Toolkit
Processes and features engineering for FX data
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from .config import config
import logging

logger = logging.getLogger(__name__)

class FXProcessor:
    """Data processor for FX time series"""
    
    def __init__(self):
        self.scaler = MinMaxScaler()
        self.feature_columns = []
    
    def clean_data(self, df):
        """Clean and prepare data for processing"""
        df = df.copy()
        
        # Convert Volume to numeric, removing commas
        if 'Volume' in df.columns:
            df['Volume'] = pd.to_numeric(df['Volume'].astype(str).str.replace(',', ''), errors='coerce')
            df['Volume'] = df['Volume'].fillna(0)  # Fill NaN volumes with 0
        
        # Ensure all price columns are numeric
        for col in ['Open', 'High', 'Low', 'Close']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Drop rows with NaN in essential columns
        df = df.dropna(subset=['Open', 'High', 'Low', 'Close'])
        
        return df
    
    def add_technical_indicators(self, df):
        """
        Add technical indicators to the DataFrame
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with additional technical indicators
        """
        df = self.clean_data(df)
        
        # Simple Moving Averages
        df['SMA_5'] = df['Close'].rolling(window=5).mean()
        df['SMA_10'] = df['Close'].rolling(window=10).mean()
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        
        # Exponential Moving Averages
        df['EMA_5'] = df['Close'].ewm(span=5).mean()
        df['EMA_10'] = df['Close'].ewm(span=10).mean()
        
        # RSI (Relative Strength Index)
        df['RSI'] = self._calculate_rsi(df['Close'], 14)
        
        # MACD
        ema_12 = df['Close'].ewm(span=12).mean()
        ema_26 = df['Close'].ewm(span=26).mean()
        df['MACD'] = ema_12 - ema_26
        df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
        df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
        
        # Bollinger Bands
        sma_20 = df['Close'].rolling(window=20).mean()
        std_20 = df['Close'].rolling(window=20).std()
        df['BB_Upper'] = sma_20 + (2 * std_20)
        df['BB_Lower'] = sma_20 - (2 * std_20)
        df['BB_Width'] = df['BB_Upper'] - df['BB_Lower']
        
        # Price change features
        df['Price_Change'] = df['Close'].pct_change()
        df['Price_Change_1d'] = df['Close'].pct_change(1)
        df['Price_Change_5d'] = df['Close'].pct_change(5)
        
        # Volatility
        df['Volatility_5d'] = df['Price_Change'].rolling(window=5).std()
        df['Volatility_10d'] = df['Price_Change'].rolling(window=10).std()
        
        # High-Low spread
        df['HL_Spread'] = (df['High'] - df['Low']) / df['Close']
        
        logger.info(f"Added technical indicators. DataFrame shape: {df.shape}")
        return df
    
    def _calculate_rsi(self, prices, window=14):
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def create_sequences(self, df, target_col='Close', window_size=None):
        """
        Create sequences for time series prediction
        
        Args:
            df: DataFrame with features
            target_col: Column to predict
            window_size: Size of the input sequence
            
        Returns:
            X, y arrays for training
        """
        if window_size is None:
            window_size = config.WINDOW_SIZE
        
        # Select numeric feature columns only (exclude date, string columns)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = [col for col in numeric_cols if col != target_col]
        
        # Remove rows with NaN values
        df_clean = df.dropna()
        
        if len(df_clean) < window_size + 1:
            raise ValueError(f"Not enough data. Need at least {window_size + 1} rows, got {len(df_clean)}")
        
        # Prepare features and target
        features = df_clean[feature_cols].values
        target = df_clean[target_col].values
        
        # Scale features
        features_scaled = self.scaler.fit_transform(features)
        
        # Create sequences
        X, y = [], []
        for i in range(len(features_scaled) - window_size):
            X.append(features_scaled[i:i + window_size])
            y.append(target[i + window_size])
        
        X = np.array(X)
        y = np.array(y)
        
        self.feature_columns = feature_cols
        
        logger.info(f"Created sequences: X shape {X.shape}, y shape {y.shape}")
        logger.info(f"Using {len(feature_cols)} features: {feature_cols[:5]}...")
        return X, y
    
    def prepare_training_data(self, df, test_size=0.2):
        """
        Prepare data for training with train/test split
        
        Args:
            df: DataFrame with features
            test_size: Fraction of data for testing
            
        Returns:
            X_train, X_test, y_train, y_test
        """
        X, y = self.create_sequences(df)
        
        # Time series split (no shuffling)
        split_idx = int(len(X) * (1 - test_size))
        
        X_train = X[:split_idx]
        X_test = X[split_idx:]
        y_train = y[:split_idx]
        y_test = y[split_idx:]
        
        logger.info(f"Training data: {X_train.shape}, Test data: {X_test.shape}")
        return X_train, X_test, y_train, y_test
    
    def save_processed_data(self, df, filename):
        """Save processed data to CSV"""
        filepath = config.PROCESSED_DATA_DIR / filename
        df.to_csv(filepath, index=False)
        logger.info(f"Processed data saved to {filepath}")
        return filepath

# Create global processor instance
processor = FXProcessor()
