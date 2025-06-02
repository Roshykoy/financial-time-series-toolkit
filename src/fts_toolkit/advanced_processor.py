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
from sklearn.decomposition import PCA
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

    def handle_outliers_iqr(self, df: pd.DataFrame, columns: list, threshold: float = 1.5) -> pd.DataFrame:
        """
                Detects and handles outliers in specified columns using the IQR method by capping.

                Args:
                    df (pd.DataFrame): Input DataFrame.
                    columns (list): List of column names to process for outliers.
                    threshold (float): The IQR threshold multiplier (e.g., 1.5).

                Returns:
                    pd.DataFrame: DataFrame with outliers capped in the specified columns.
        """
        df_processed = df.copy()  # Work on a copy
        logger.info(f"🔍 Starting outlier handling for columns: {columns}")

        for col in columns:
            if col not in df_processed.columns:
                logger.warning(f"  Column '{col}' not found in DataFrame. Skipping outlier handling for it.")
                continue

            if not pd.api.types.is_numeric_dtype(df_processed[col]):
                logger.warning(f"  Column '{col}' is not numeric. Skipping outlier handling for it.")
                continue

            logger.debug(f"  Processing column: {col}")
            Q1 = df_processed[col].quantile(0.25)
            Q3 = df_processed[col].quantile(0.75)
            IQR = Q3 - Q1

            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR

            # Detect outliers
            outliers_lower = df_processed[col] < lower_bound
            outliers_upper = df_processed[col] > upper_bound
            num_outliers_lower = outliers_lower.sum()
            num_outliers_upper = outliers_upper.sum()

            if num_outliers_lower > 0 or num_outliers_upper > 0:
                logger.info(
                    f"    Found {num_outliers_lower} lower-bound and {num_outliers_upper} upper-bound outliers in '{col}'.")
                logger.info(
                    f"    '{col}': Q1={Q1:.4f}, Q3={Q3:.4f}, IQR={IQR:.4f}, LowerBound={lower_bound:.4f}, UpperBound={upper_bound:.4f}")

                # Cap outliers
                df_processed[col] = np.where(outliers_lower, lower_bound, df_processed[col])
                df_processed[col] = np.where(outliers_upper, upper_bound, df_processed[col])
                logger.info(f"    Capped outliers in '{col}'.")
            else:
                logger.debug(f"    No outliers detected in '{col}' with threshold {threshold}.")

        logger.info("✅ Outlier handling completed.")
        return df_processed

    def engineer_advanced_features(self, df):
        """
        Advanced feature engineering based on the report's recommendations
        """
        logger.info("⚙️ Starting Advanced Feature Engineering...")

        # --- CRITICAL FIX: Initial Data Cleaning ---
        # Ensure all relevant columns ('Volume', 'Open', 'High', 'Low', 'Close') are numeric
        # and handle missing values before any calculations.
        df = self._clean_initial_data(df)

        ohlcv_columns = ['Open', 'High', 'Low', 'Close']
        if 'Volume' in df.columns and pd.api.types.is_numeric_dtype(df['Volume']):  # Check if Volume is numeric
            ohlcv_columns.append('Volume')

        # Filter out columns that might not exist or are not numeric yet from ohlcv_columns
        columns_to_check_for_outliers = [col for col in ohlcv_columns if
                                         col in df.columns and pd.api.types.is_numeric_dtype(df[col])]

        if columns_to_check_for_outliers:
            df = self.handle_outliers_iqr(df, columns=columns_to_check_for_outliers)
        else:
            logger.info("No suitable OHLCV columns found for outlier handling or Volume is not numeric.")

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
            logger.debug("Calculating volume-based features...")
            # Use min_periods=1 for rolling mean to get a value as soon as possible,
            # though for a 20-period SMA, it's more meaningful after 20 periods.
            df['volume_sma_20'] = df['Volume'].rolling(window=20, min_periods=1).mean()

            # Calculate volume_ratio carefully to avoid infinity and handle NaNs
            # 1. Divide, which might produce NaN (0/0) or inf (x/0)
            # 2. Replace inf/-inf with NaN
            # 3. Fill remaining NaNs (e.g., with 0, assuming 0 ratio if volume or avg volume is 0/NaN)
            df['volume_ratio'] = df['Volume'].divide(df['volume_sma_20']).replace([np.inf, -np.inf], np.nan)

            # What to fill NaNs in volume_ratio with is debatable.
            # If volume_sma_20 is 0, the ratio is undefined.
            # If we are keeping the column, we need to fill these NaNs.
            # Filling with 0 is one option, implying no significant volume activity relative to average.
            # The initial NaNs (first 19 from rolling) will also be filled.
            df['volume_ratio'] = df['volume_ratio'].fillna(0)
            logger.debug(f"NaNs in 'volume_ratio' after processing: {df['volume_ratio'].isnull().sum()}")

            df['price_volume'] = df['Close'] * df['Volume']  # This is fine; will be 0 if Volume is 0
        else:
            logger.warning(
                "Volume column not suitable for advanced volume features (missing or non-numeric). Skipping and "
                "filling with NaN.")
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

    def create_stationary_sequences(self, df, target_col='Close', window_size=None,
                                    test_stationarity_bool=True,  # Renamed for clarity
                                    n_pca_components=None):  # New parameter for PCA
        """
        Create sequences with stationarity testing, optional PCA, and transformation for ML models.
        """
        if window_size is None:
            try:
                window_size = config.WINDOW_SIZE
            except AttributeError:
                logger.error("config.WINDOW_SIZE not found. Using default window_size=60.")
                window_size = 60

        df_processed = df.copy()

        # Test stationarity of target variable
        if test_stationarity_bool:
            if target_col not in df_processed.columns or \
                    not pd.api.types.is_numeric_dtype(df_processed[target_col]):
                logger.error(
                    f"Target column '{target_col}' for stationarity testing is missing or not numeric. Skipping stationarity processing.")
                # test_stationarity_bool = False # Flag is already being checked; no need to modify here
            else:
                target_series = df_processed[target_col].dropna()
                if target_series.empty:
                    logger.warning(
                        f"Target series '{target_col}' is empty after dropping NaNs. Skipping stationarity test and differencing.")
                else:
                    # Perform the stationarity test
                    stationarity_test_output = self.test_stationarity(target_series, target_col)
                    self.stationarity_results[
                        target_col] = stationarity_test_output  # Store original target_col results

                    # Check if the test itself had an error
                    if stationarity_test_output.get('error'):
                        logger.warning(
                            f"Stationarity test for {target_col} resulted in an error: {stationarity_test_output['error']}. Not attempting to make stationary.")
                    # Access 'recommendation' only if no error from the test
                    elif stationarity_test_output.get('recommendation') != 'stationary':
                        logger.info(
                            f"Target '{target_col}' is {stationarity_test_output.get('recommendation', 'unknown')}. Making it stationary...")
                        # Pass the original target_series to make_stationary
                        stationary_target, diff_order = self.make_stationary(target_series)

                        if not stationary_target.empty:
                            # Use original target_col for naming to avoid nesting _stationary_stationary
                            original_target_base_name = target_col.replace(f'_stationary_d{diff_order - 1}',
                                                                           '') if diff_order > 0 else target_col  # Get base name
                            new_stationary_col_name = f'{original_target_base_name}_stationary_d{diff_order}'

                            temp_df = pd.DataFrame({new_stationary_col_name: stationary_target},
                                                   index=stationary_target.index)
                            df_processed = df_processed.join(temp_df, how='left')
                            target_col = new_stationary_col_name  # Update the target column name FOR THIS FUNCTION'S SCOPE
                            logger.info(f"Target column for sequences is now '{target_col}'.")
                        elif len(target_series) > 0:  # stationary_target is empty but original series was not
                            logger.warning(
                                f"Failed to make '{target_col}' stationary or result is empty (e.g., series too short for differencing). Using original (or last successfully differenced) series as target.")
                        # If stationary_target is empty and original series was also empty, it's already handled.
                    else:
                        logger.info(
                            f"Target '{target_col}' is already {stationarity_test_output.get('recommendation', 'stationary')}.")
        # Ensure current_target_col_name reflects the final target_col after potential stationarization
        current_target_col_name = target_col

        numeric_cols = df_processed.select_dtypes(include=[np.number]).columns.tolist()

        if target_col not in numeric_cols:
            if target_col in df_processed.columns and pd.api.types.is_numeric_dtype(df_processed[target_col]):
                numeric_cols.append(target_col)  # Ensure target_col is considered numeric
            else:  # If target_col is still not numeric (e.g. after differencing it was removed or became all NaN)
                logger.error(
                    f"Target column '{target_col}' is not available as a numeric column. Cannot proceed with sequence creation.")
                # You might want to raise an error or return empty arrays here
                return np.array([]), np.array([]), target_col

        feature_cols = [col for col in numeric_cols if col != target_col and not col.startswith('Date')]

        # Ensure all columns in subset_for_dropna actually exist
        # Original code used target_col which could be string 'Close_stationary'
        # Ensure the actual target_col (potentially modified by stationarity) is used for subset
        current_target_col_name = target_col
        subset_for_dropna = feature_cols + [current_target_col_name]

        valid_subset_for_dropna = [col for col in subset_for_dropna if col in df_processed.columns]

        # ---- START OF DEBUGGING BLOCK ----
        if not valid_subset_for_dropna:
            logger.error("CRITICAL DEBUG: No valid columns found for dropna. Check feature_cols and target_col.")
        else:
            logger.info(
                f"DEBUG: About to call dropna. Inspecting df_processed with {len(valid_subset_for_dropna)} columns for dropna.")

            # Create the slice that will be used in dropna
            df_slice_for_dropna = df_processed[valid_subset_for_dropna]

            # Print how many NaNs each column in this slice has
            nan_counts = df_slice_for_dropna.isnull().sum().sort_values(ascending=False)
            logger.info(
                f"DEBUG: NaN counts per column (before dropna):\n{nan_counts[nan_counts > 0]}")  # Show only columns with NaNs

            # Save this slice to a CSV for detailed inspection
            # You can open this CSV in Excel or a data analysis tool
            debug_csv_path = "debug_df_slice_before_dropna.csv"
            try:
                df_slice_for_dropna.to_csv(debug_csv_path)
                logger.info(f"DEBUG: Saved df_slice_for_dropna to {debug_csv_path} for inspection.")
            except Exception as e_csv:
                logger.error(f"DEBUG: Failed to save debug CSV: {e_csv}")

            # Check how many rows remain if we drop NaNs
            rows_after_dropna_test = df_slice_for_dropna.dropna().shape[0]
            logger.info(f"DEBUG: Test - number of rows that would remain after dropna: {rows_after_dropna_test}")

        # ---- END OF DEBUGGING BLOCK ----

        df_clean = df_processed[valid_subset_for_dropna].dropna().copy()

        if len(df_clean) < window_size + 1:
            raise ValueError(
                f"Insufficient data for sequence creation after dropping NaNs. "
                f"Need {window_size + 1} rows, got {len(df_clean)}. "
                f"Target: '{current_target_col_name}', Features: {feature_cols[:3]}..."
            )

        features_data = df_clean[feature_cols].values
        target_data = df_clean[current_target_col_name].values

        self.feature_columns = feature_cols  # Store original feature names BEFORE PCA

        # --- NEW: Optional PCA step ---
        if n_pca_components is not None and features_data.shape[
            1] > 1:  # Apply PCA if requested and more than 1 feature
            logger.info(f"🔩 Applying PCA with n_components={n_pca_components} to {features_data.shape[1]} features.")
            pca = PCA(n_components=n_pca_components)
            features_data = pca.fit_transform(features_data)  # Fit PCA and transform features
            logger.info(f"  Features transformed to {features_data.shape[1]} principal components.")
            # Note: After PCA, self.feature_columns (original names) won't directly map to the PCA components.
            # For interpretability, you might store pca.explained_variance_ratio_
        else:
            logger.info("PCA step skipped.")
        # --- End of PCA step ---

        # Scale features (either original or PCA-transformed)
        # Important: self.scaler should be fit only on training data in a true pipeline.
        # For now, it's fit on all data passed to this function.
        # This is generally handled correctly in your walk_forward_validation.
        if features_data.shape[0] > 0:  # Check if features_data is not empty
            features_scaled = self.scaler.fit_transform(features_data)
        else:
            logger.warning("No feature data to scale after cleaning and PCA (if applied).")
            return np.array([]), np.array([]), current_target_col_name

        X, y = [], []
        for i in range(len(features_scaled) - window_size):
            X.append(features_scaled[i:i + window_size])
            y.append(target_data[i + window_size])  # Use target_data

        X = np.array(X)
        y = np.array(y)

        # Log the number of features actually used for X after PCA and scaling
        num_features_in_x = X.shape[2] if X.ndim == 3 else 0  # X is (samples, window, features)
        logger.info(f"Created sequences: X shape {X.shape}, y shape {y.shape}")
        if num_features_in_x > 0:
            logger.info(
                f"Using {num_features_in_x} features in sequences (after PCA if applied). Original features (before PCA): {self.feature_columns[:5]}...")
        else:
            logger.info("No features in the generated sequences X.")

        return X, y, current_target_col_name  # Return the actual target column name used

    def prepare_training_data(self, df: pd.DataFrame, test_size: float = 0.2, target_col: str = 'Close',
                              n_pca_components=None):  # Pass PCA components here too
        """
        Prepares data for training by creating stationary sequences and performing a train/test split.
        """
        # Pass n_pca_components to create_stationary_sequences
        X, y, final_target_col_used = self.create_stationary_sequences(
            df, target_col=target_col, n_pca_components=n_pca_components
        )  # Note: I removed test_stationarity_bool for now, assuming it's usually True.
        # You can add it back if you need to control it from here.

        if X.shape[0] == 0:  # Check if X is empty
            logger.error("No data returned from create_stationary_sequences. Cannot split into train/test.")
            return np.array([]), np.array([]), np.array([]), np.array([])

        split_idx = int(len(X) * (1 - test_size))

        X_train = X[:split_idx]
        X_test = X[split_idx:]
        y_train = y[:split_idx]
        y_test = y[split_idx:]

        logger.info(f"Training data: {X_train.shape}, Test data: {X_test.shape}")
        return X_train, X_test, y_train, y_test


# Create global advanced processor instance
advanced_processor = AdvancedFXProcessor()
