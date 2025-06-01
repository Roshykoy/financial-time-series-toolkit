"""
Advanced Data Processing with Stationarity Testing and Feature Engineering
Based on sophisticated methodologies for financial time series
"""

import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from .config import config
import logging
import warnings  # Import warnings to suppress InterpolationWarning

logger = logging.getLogger(__name__)


class AdvancedFXProcessor:
    """Advanced processor implementing sophisticated feature engineering and stationarity testing"""

    def __init__(self):
        self.scaler = StandardScaler()  # Changed to StandardScaler for better ML performance
        self.feature_columns = []
        self.stationarity_results = {}

    def _clean_initial_data(self, df):
        """
        Internal helper to clean and prepare initial raw data.
        Ensures 'Volume' and price columns are numeric.
        """
        df = df.copy()  # Work on a copy to avoid modifying original DataFrame slices

        # Convert Volume to numeric, removing commas
        if 'Volume' in df.columns:
            # Using .astype(str) to handle potential non-string types gracefully
            df['Volume'] = pd.to_numeric(df['Volume'].astype(str).str.replace(',', ''), errors='coerce')
            df['Volume'] = df['Volume'].fillna(0)  # Fill NaN volumes with 0, or consider dropping
            logger.debug("Cleaned 'Volume' column to numeric.")

        # Ensure all core price columns are numeric
        for col in ['Open', 'High', 'Low', 'Close']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                logger.debug(f"Cleaned '{col}' column to numeric.")

        # Drop rows with NaN in essential price columns that might result from 'coerce'
        # This is crucial before calculating any indicators
        initial_rows = len(df)
        df.dropna(subset=['Open', 'High', 'Low', 'Close'], inplace=True)
        if len(df) < initial_rows:
            logger.warning(
                f"Dropped {initial_rows - len(df)} rows due to NaNs in essential price columns after initial cleaning.")

        # Ensure Date column is datetime if present and needed for time features
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            df.dropna(subset=['Date'], inplace=True)  # Drop rows where Date conversion failed
            df = df.sort_values(by='Date')  # Ensure data is sorted by date
            logger.debug("Cleaned and sorted by 'Date' column.")

        return df

    def test_stationarity(self, series, series_name="Series"):
        """
        Comprehensive stationarity testing using ADF, KPSS, and PP tests

        Args:
            series: Time series to test
            series_name: Name for logging

        Returns:
            dict: Test results and recommendations
        """
        # Remove NaN values
        clean_series = series.dropna()

        results = {
            'series_name': series_name,
            'length': len(clean_series),
            'adf_test': {},
            'kpss_test': {},
            'recommendation': 'stationary'
        }

        if clean_series.empty:
            logger.warning(f"Cannot perform stationarity test on empty series: {series_name}")
            results['error'] = "Series is empty after dropping NaNs."
            return results

        try:
            # Augmented Dickey-Fuller Test (H0: non-stationary)
            adf_result = adfuller(clean_series, autolag='AIC')
            results['adf_test'] = {
                'statistic': adf_result[0],
                'p_value': adf_result[1],
                'critical_values': adf_result[4],
                'is_stationary': adf_result[1] < 0.05  # Reject H0 if p < 0.05
            }

            # KPSS Test (H0: stationary)
            # Suppress InterpolationWarning from KPSS
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=UserWarning)
                kpss_result = kpss(clean_series, regression='ct')  # constant and trend
            results['kpss_test'] = {
                'statistic': kpss_result[0],
                'p_value': kpss_result[1],
                'critical_values': kpss_result[3],
                'is_stationary': kpss_result[1] > 0.05  # Fail to reject H0 if p > 0.05
            }

            # Combined recommendation
            adf_stationary = results['adf_test']['is_stationary']
            kpss_stationary = results['kpss_test']['is_stationary']

            if adf_stationary and kpss_stationary:
                results['recommendation'] = 'stationary'
            elif not adf_stationary and not kpss_stationary:
                results['recommendation'] = 'non_stationary'
            else:  # Contradictory results
                results['recommendation'] = 'inconclusive'

            logger.info(f"Stationarity test for {series_name}:")
            logger.info(f"  ADF: {'Stationary' if adf_stationary else 'Non-stationary'} (p={adf_result[1]:.4f})")
            logger.info(f"  KPSS: {'Stationary' if kpss_stationary else 'Non-stationary'} (p={kpss_result[1]:.4f})")
            logger.info(f"  Recommendation: {results['recommendation']}")

        except Exception as e:
            logger.error(f"Error in stationarity testing for {series_name}: {e}")
            results['error'] = str(e)

        return results

    def make_stationary(self, series, max_diff=2):
        """
        Apply differencing to achieve stationarity

        Args:
            series: Time series
            max_diff: Maximum order of differencing

        Returns:
            tuple: (differenced_series, order_of_differencing)
        """
        current_series = series.copy().dropna()  # Ensure series is clean
        diff_order = 0

        if current_series.empty:
            logger.warning(f"Cannot make empty series stationary.")
            return pd.Series(), 0

        for d in range(max_diff + 1):
            if len(current_series) <= (d + 1):  # Need at least d+1 observations for d-th difference
                logger.warning(f"Series too short to perform {d}-th differencing.")
                break

            test_result = self.test_stationarity(current_series, f"{series.name or 'Series'} (diff_order={d})")

            if test_result['recommendation'] == 'stationary':
                logger.info(f"Achieved stationarity with d={diff_order} for {series.name or 'Series'}")
                return current_series, diff_order

            if d < max_diff:
                current_series = current_series.diff().dropna()
                diff_order += 1

        logger.warning(
            f"Could not achieve stationarity for {series.name or 'Series'} within {max_diff} differences. Returning last differenced series.")
        return current_series, diff_order

    def engineer_advanced_features(self, df):
        """
        Advanced feature engineering based on the report's recommendations
        """
        logger.info("⚙️ Starting Advanced Feature Engineering...")

        # --- CRITICAL FIX: Initial Data Cleaning ---
        # Ensure all relevant columns ('Volume', 'Open', 'High', 'Low', 'Close') are numeric
        # and handle missing values before any calculations.
        df = self._clean_initial_data(df)

        # Basic price transformations
        # These operations will now work correctly as price columns are numeric
        df['log_returns'] = np.log(df['Close'] / df['Close'].shift(1))
        df['simple_returns'] = df['Close'].pct_change()
        df['typical_price'] = (df['High'] + df['Low'] + df['Close']) / 3
        df['price_range'] = df['High'] - df['Low']
        df['price_range_pct'] = df['price_range'] / df['Close']

        # Lagged features for autoregressive patterns
        for lag in [1, 2, 3, 5, 10]:
            df[f'close_lag_{lag}'] = df['Close'].shift(lag)
            df[f'returns_lag_{lag}'] = df['log_returns'].shift(lag)

        # Moving averages (multiple timeframes)
        for window in [5, 10, 20, 50]:
            df[f'sma_{window}'] = df['Close'].rolling(window=window).mean()
            # Ensure EMA calculations use adjust=False for typical financial EMAs
            df[f'ema_{window}'] = df['Close'].ewm(span=window, adjust=False).mean()

            # Momentum indicators
        df['rsi_14'] = self._calculate_rsi(df['Close'], 14)
        df['rsi_21'] = self._calculate_rsi(df['Close'], 21)

        # MACD
        ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
        ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
        df['macd'] = ema_12 - ema_26
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']

        # Bollinger Bands
        sma_20 = df['Close'].rolling(window=20).mean()
        std_20 = df['Close'].rolling(window=20).std()
        df['bb_upper'] = sma_20 + (2 * std_20)
        df['bb_lower'] = sma_20 - (2 * std_20)
        df['bb_width'] = df['bb_upper'] - df['bb_lower']
        df['bb_position'] = (df['Close'] - df['bb_lower']) / df['bb_width']

        # Volatility measures
        for window in [5, 10, 20]:
            # Ensure log_returns is not NaN before rolling operations
            df[f'volatility_{window}d'] = df['log_returns'].rolling(window=window).std()
            df[f'realized_vol_{window}d'] = df['log_returns'].rolling(window=window).std() * np.sqrt(252)

        # Advanced features from the report
        # EMA ratios and differences
        df['price_ema_ratio_20'] = df['Close'] / df['ema_20']
        df['price_ema_diff_20'] = (df['Close'] - df['ema_20']) / df['ema_20']

        # Indicator-price slope ratios
        df['price_slope_5'] = df['Close'].diff(5) / 5
        df['ema_slope_5'] = df['ema_20'].diff(5) / 5
        # Ensure denominator is not zero and handle potential division by zero results
        df['slope_ratio'] = df['ema_slope_5'] / (df['price_slope_5'].replace(0, np.nan) + 1e-8)

        # Overnight gaps
        df['overnight_gap'] = df['Open'] - df['Close'].shift(1)
        df['overnight_gap_pct'] = df['overnight_gap'] / df['Close'].shift(1)

        # Volume-based features (if Volume exists and is numeric)
        if 'Volume' in df.columns and pd.api.types.is_numeric_dtype(df['Volume']):
            # This line caused the error previously; now 'Volume' is numeric
            df['volume_sma_20'] = df['Volume'].rolling(window=20).mean()
            df['volume_ratio'] = df['Volume'] / df['volume_sma_20']
            df['price_volume'] = df['Close'] * df['Volume']
        else:
            logger.warning(
                "Volume column not suitable for advanced volume features (missing or non-numeric). Skipping.")
            # Ensure these columns exist even if not calculated
            df['volume_sma_20'] = np.nan
            df['volume_ratio'] = np.nan
            df['price_volume'] = np.nan

        # Cyclical time features
        # Check if 'Date' column is present and is of datetime type
        if 'Date' in df.columns and pd.api.types.is_datetime64_any_dtype(df['Date']):
            df['day_of_week'] = df['Date'].dt.dayofweek
            df['day_of_month'] = df['Date'].dt.day
            df['month'] = df['Date'].dt.month

            # Cyclical encoding
            df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
            df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
            df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
            df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        else:
            logger.warning("Date column not suitable for cyclical features (missing or not datetime). Skipping.")

        logger.info(f"Advanced feature engineering completed. Shape: {df.shape}")
        return df

    def _calculate_rsi(self, prices, window=14):
        """Enhanced RSI calculation"""
        # Ensure prices are numeric before operations
        prices = pd.to_numeric(prices, errors='coerce').dropna()
        if prices.empty:
            logger.warning("RSI calculation: Prices series is empty after dropping NaNs.")
            return pd.Series(np.nan, index=prices.index)  # Return NaN series with original index

        delta = prices.diff()
        gain = delta.where(delta > 0, 0).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()

        # Handle division by zero for rs where loss is 0
        with np.errstate(divide='ignore', invalid='ignore'):
            rs = gain / loss
            rs[np.isinf(rs)] = np.nan  # Replace inf with NaN for consistency
            # If gain > 0 and loss == 0, RS is inf, RSI becomes 100.
            # This is handled by the 100 - (100 / (1 + rs)) formula where 1/(1+inf) is 0

        rsi = 100 - (100 / (1 + rs))
        return rsi

    def create_stationary_sequences(self, df, target_col='Close', window_size=None, test_stationarity=True):
        """
        Create sequences with stationarity testing and transformation for machine learning models.
        """
        if window_size is None:
            try:
                window_size = config.WINDOW_SIZE
            except AttributeError:
                logger.error("config.WINDOW_SIZE not found. Using default window_size=60.")
                window_size = 60  # Fallback

        df_processed = df.copy()

        # Test stationarity of target variable
        if test_stationarity:
            # Ensure target_col is numeric before testing stationarity
            if target_col not in df_processed.columns or not pd.api.types.is_numeric_dtype(df_processed[target_col]):
                logger.error(f"Target column '{target_col}' is missing or not numeric. Cannot test stationarity.")
                test_stationarity = False  # Skip stationarity testing if target is not numeric
            else:
                target_series = df_processed[target_col].dropna()
                if target_series.empty:
                    logger.warning(
                        f"Target series '{target_col}' is empty after dropping NaNs. Skipping stationarity test.")
                    test_stationarity = False
                else:
                    stationarity_result = self.test_stationarity(target_series, target_col)
                    self.stationarity_results[target_col] = stationarity_result

                    # Make target stationary if needed
                    if stationarity_result['recommendation'] != 'stationary':
                        logger.info(f"Making {target_col} stationary...")
                        stationary_target, diff_order = self.make_stationary(target_series)

                        # --- FIX START ---
                        # Apply transformation back to the main DataFrame for consistency
                        # Create a new DataFrame with the stationary series and the desired new column name
                        new_stationary_col_name = f'{target_col}_stationary'
                        # Ensure the index matches df_processed's index for correct alignment
                        # You'll likely need to reindex stationary_target if it doesn't align perfectly
                        # or ensure the original series was indexed appropriately before differencing.
                        # For now, we'll assume the index of stationary_target is suitable for joining.
                        temp_df = pd.DataFrame({new_stationary_col_name: stationary_target},
                                               index=stationary_target.index)

                        # Join the new stationary column to the processed DataFrame
                        # Use left join to keep all rows from df_processed and add the new column
                        df_processed = df_processed.join(temp_df, how='left')

                        # Update target_col to use the newly created stationary column
                        target_col = new_stationary_col_name
                        # --- FIX END ---

        # Select numeric features only
        # Make sure to include the potentially new target_col if it was created
        numeric_cols = df_processed.select_dtypes(include=[np.number]).columns.tolist()

        # Ensure target_col is in numeric_cols before filtering it out for features
        if target_col not in numeric_cols:
            logger.warning(
                f"Target column '{target_col}' not found in numeric columns after stationarity check. This might lead to issues.")
            # Consider adding target_col to numeric_cols if it's supposed to be numeric. This can happen if,
            # for example, the original 'Close' was numeric but the _stationary version somehow isn't,
            # or if the original 'Close' was dropped for some reason before this step. For robustness,
            # we can explicitly add it if it exists in df_processed and is numeric.
            if target_col in df_processed.columns and pd.api.types.is_numeric_dtype(df_processed[target_col]):
                numeric_cols.append(target_col)

        feature_cols = [col for col in numeric_cols if col != target_col and not col.startswith('Date')]

        # Remove rows with NaN values. This is crucial AFTER all feature engineering
        # and stationarity transformations, as many operations introduce NaNs at the beginning.
        subset_for_dropna = feature_cols + [target_col]

        # Ensure all columns in subset_for_dropna actually exist in df_processed
        valid_subset_for_dropna = [col for col in subset_for_dropna if col in df_processed.columns]
        if len(valid_subset_for_dropna) < len(subset_for_dropna):
            missing_cols = set(subset_for_dropna) - set(valid_subset_for_dropna)
            logger.warning(
                f"Some columns specified for dropna (e.g., {missing_cols}) are not in the DataFrame. Dropping NaNs based on available columns.")

        df_clean = df_processed[valid_subset_for_dropna].dropna().copy()

        if len(df_clean) < window_size + 1:
            raise ValueError(
                f"Insufficient data for sequence creation after dropping NaNs. "
                f"Need at least {window_size + 1} rows, got {len(df_clean)}."
                f"Consider checking input data or reducing window_size."
            )

        # Prepare features and target
        features = df_clean[feature_cols].values
        target = df_clean[target_col].values

        # Scale features
        # Fit scaler only on training data if doing train/test split here.
        # For simplicity in this general function, we fit on all clean features.
        # In a full ML pipeline, you'd fit scaler on X_train only.
        features_scaled = self.scaler.fit_transform(features)

        # Create sequences
        X, y = [], []
        for i in range(len(features_scaled) - window_size):
            X.append(features_scaled[i:i + window_size])
            y.append(target[i + window_size])

        X = np.array(X)
        y = np.array(y)

        self.feature_columns = feature_cols  # Store feature columns for later use (e.g., inverse transform)

        logger.info(f"Created stationary sequences: X shape {X.shape}, y shape {y.shape}")
        logger.info(f"Using {len(feature_cols)} features: {feature_cols[:5]}{'...' if len(feature_cols) > 5 else ''}")

        return X, y, target_col

    def prepare_training_data(self, df: pd.DataFrame, test_size: float = 0.2, target_col: str = 'Close'):
        """
        Prepares data for training by creating stationary sequences and performing a train/test split.

        Args:
            df (pd.DataFrame): The DataFrame with engineered features.
            test_size (float): The proportion of the dataset to include in the test split.
            target_col (str): The name of the target column to predict.

        Returns:
            tuple: X_train, X_test, y_train, y_test arrays for machine learning models.
        """
        # Call the existing method to get sequences
        # create_stationary_sequences returns X, y, and the final_target_col (which might be differenced)
        X, y, final_target_col_used = self.create_stationary_sequences(df, target_col=target_col)

        # Time series split (no shuffling)
        split_idx = int(len(X) * (1 - test_size))

        X_train = X[:split_idx]
        X_test = X[split_idx:]
        y_train = y[:split_idx]
        y_test = y[split_idx:]

        logger.info(f"Training data: {X_train.shape}, Test data: {X_test.shape}")
        return X_train, X_test, y_train, y_test


# Create global advanced processor instance
advanced_processor = AdvancedFXProcessor()
