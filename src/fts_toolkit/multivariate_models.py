# src/fts_toolkit/multivariate_models.py
import pandas as pd
import numpy as np
import traceback
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller  # For a quick check if needed, though processor handles it
from .config import config  # For VAR related configs if needed directly here
import logging

logger = logging.getLogger(__name__)


class VARAnalyzer:
    def __init__(self, advanced_processor_instance):
        """
        Initialize VARAnalyzer with an instance of AdvancedFXProcessor.
        Args:
            advanced_processor_instance: An instance of your AdvancedFXProcessor class.
        """
        self.fitted_model = None
        self.data_columns_stationary = None  # Will store names of stationary columns
        self.stationary_data = None
        self.optimal_lag = None
        self.processor = advanced_processor_instance
        self.raw_data_columns = None  # To store original column names before differencing

    def prepare_data_for_var(self, df_multi_series_raw):
        """
        Takes a DataFrame of multiple raw time series (e.g., 'Close' prices),
        makes each series stationary using the provided processor, and aligns them.

        Args:
            df_multi_series_raw (pd.DataFrame): DataFrame with multiple time series as columns,
                                                index as datetime.
        Returns:
            pd.DataFrame: DataFrame with stationary series, or None if preparation fails.
        """
        logger.info("Preparing data for VAR model: Ensuring stationarity for each series...")
        if df_multi_series_raw.empty:
            logger.error("Input DataFrame for VAR preparation is empty.")
            return None

        self.raw_data_columns = df_multi_series_raw.columns.tolist()
        stationary_series_list = []
        new_column_names = []

        for col in self.raw_data_columns:
            series = df_multi_series_raw[col].dropna()
            if series.empty:
                logger.warning(f"Series {col} is empty after initial dropna. Skipping.")
                continue

            logger.debug(f"Processing stationarity for series: {col}")
            # make_stationary returns (stationary_series, order_of_differencing)
            stat_series, d_order = self.processor.make_stationary(series.copy())

            if stat_series.empty:
                logger.warning(f"Series {col} became empty after attempting to make it stationary. Skipping.")
                continue

            new_name = f"{col}_stationary_d{d_order}" if d_order > 0 else col
            stat_series.name = new_name
            stationary_series_list.append(stat_series)
            new_column_names.append(new_name)

        if not stationary_series_list:
            logger.error("No series could be made stationary or all were empty.")
            return None

        # Concatenate all stationary series. Inner join handles different start/end dates after differencing.
        stationary_df = pd.concat(stationary_series_list, axis=1, join='inner')

        if stationary_df.empty:
            logger.error(
                "Stationary DataFrame is empty after concatenation and join. Check data alignment and stationarity process.")
            return None

        self.stationary_data = stationary_df
        self.data_columns_stationary = stationary_df.columns.tolist()  # Final columns used for VAR
        logger.info(f"Final stationary VAR input data shape: {self.stationary_data.shape}")
        logger.info(f"Stationary columns for VAR: {self.data_columns_stationary}")
        return self.stationary_data

    def fit(self, maxlags=None, ic='aic'):
        """Fits the VAR model to the prepared stationary data."""
        if self.stationary_data is None or self.stationary_data.empty:
            logger.error("Stationary data not prepared or is empty. Cannot fit VAR model.")
            return None

        if len(self.stationary_data) < (maxlags or config.VAR_MAX_LAGS) * self.stationary_data.shape[
            1] * 2:  # Heuristic for enough data
            logger.warning(
                f"Potentially insufficient data for VAR lag selection and fitting (data length: {len(self.stationary_data)}). Proceeding with caution.")

        if maxlags is None:
            maxlags = config.VAR_MAX_LAGS

        logger.info(f"Finding optimal lag order for VAR (maxlags={maxlags}, ic={ic})...")
        var_model_for_lag_selection = VAR(self.stationary_data)
        try:
            # Note: select_order can be computationally intensive
            selected_orders = var_model_for_lag_selection.select_order(maxlags=maxlags)
            self.optimal_lag = selected_orders.selected_orders.get(ic.lower())
            if self.optimal_lag is None:
                self.optimal_lag = selected_orders.aic  # Default to AIC's lag
                logger.warning(f"{ic.upper()} not found in lag selection results, using AIC lag: {self.optimal_lag}")
            logger.info(f"Selected optimal lag order: {self.optimal_lag}")
            # print(selected_orders.summary()) # Optional: for detailed view
        except Exception as e:
            logger.error(f"Error during VAR lag selection: {e}. Defaulting to lag 1.")
            self.optimal_lag = 1

        logger.info(f"Fitting VAR({self.optimal_lag}) model...")
        model = VAR(self.stationary_data)
        self.fitted_model = model.fit(self.optimal_lag)
        logger.info("VAR model fitted successfully.")
        # print(self.fitted_model.summary()) # For detailed summary
        return self.fitted_model

    def forecast(self, steps=None):
        """Generates forecasts from the fitted VAR model."""
        if self.fitted_model is None:
            logger.error("Model not fitted. Cannot forecast.")
            return None
        if steps is None:
            steps = config.VAR_FORECAST_STEPS

        lag_order = self.fitted_model.k_ar
        if len(self.stationary_data) < lag_order:
            logger.error(
                f"Not enough data points ({len(self.stationary_data)}) to make a forecast with lag order {lag_order}.")
            return None

        y_input = self.stationary_data.values[-lag_order:]

        forecast_result = self.fitted_model.forecast(y=y_input, steps=steps)
        forecast_df = pd.DataFrame(forecast_result, columns=self.data_columns_stationary)

        logger.info(f"Generated {steps}-step VAR forecast.")
        return forecast_df

    def get_impulse_response(self, steps=None): # Removed signif from method signature for now, or keep and use for plotting later
        """
        Calculate Impulse Response Functions.
        Args:
            steps (int, optional): Number of steps for IRF. Defaults to config.VAR_IRF_STEPS.
            orth (bool, optional): Whether to compute orthogonalized IRFs (Cholesky). Defaults to True.
        Returns:
            statsmodels.tsa.vector_ar.irf.IRAnalysis or None
        """
        if self.fitted_model is None:
            logger.error("Model not fitted. Cannot get IRF.")
            return None
        if steps is None:
            steps = config.VAR_IRF_STEPS

        # MODIFIED LOGGING LINE:
        logger.info(f"Calculating Impulse Response Functions for {steps} steps...")
        try:
            irf = self.fitted_model.irf(periods=steps)  # Correct: orth is not a direct param here
            logger.info("Impulse Response Functions calculated successfully.")
            return irf
        except Exception as e:
            logger.error(f"Error calculating Impulse Response Functions: {e}")
            traceback.print_exc()
            return None

    def get_fevd(self, steps=None):
        """
        Calculate Forecast Error Variance Decomposition.
        Args:
            steps (int, optional): Number of steps for FEVD. Defaults to config.VAR_IRF_STEPS.
        Returns:
            statsmodels.tsa.vector_ar.var_model.FEVD or None
        """
        if self.fitted_model is None:
            logger.error("Model not fitted. Cannot get FEVD.")
            return None
        if steps is None:
            steps = config.VAR_IRF_STEPS  # Can use the same horizon as IRF

        logger.info(f"Calculating Forecast Error Variance Decomposition for {steps} steps...")
        try:
            fevd = self.fitted_model.fevd(periods=steps)
            logger.info("Forecast Error Variance Decomposition calculated successfully.")
            return fevd
        except Exception as e:
            logger.error(f"Error calculating Forecast Error Variance Decomposition: {e}")
            return None