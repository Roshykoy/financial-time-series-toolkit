"""
Forecasting Models for Financial Time Series Toolkit
Simple models for FX prediction
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.svm import SVR
import xgboost as xgb
from .config import config
import logging

logger = logging.getLogger(__name__)


class FXForecaster:
    """Simple forecasting models for FX data"""

    def __init__(self):
        self.models = {}
        self.metrics = {}
        self.predictions = {}  # NEW: Dictionary to store predictions separately

    def train_linear_model(self, X_train, y_train, X_test, y_test):
        """Train a linear regression model"""
        model = LinearRegression()

        # Reshape data for sklearn (flatten sequences)
        X_train_flat = X_train.reshape(X_train.shape[0], -1)
        X_test_flat = X_test.reshape(X_test.shape[0], -1)

        # Ensure y_train is 1D for sklearn models if it somehow became 2D
        if y_train.ndim > 1:
            y_train = y_train.ravel()

        model.fit(X_train_flat, y_train)

        # Predictions
        y_pred_train = model.predict(X_train_flat)
        y_pred_test = model.predict(X_test_flat)  # Keep this raw prediction

        # --- CRITICAL FIX START (re-emphasized) ---
        # Ensure y_test and y_pred_test are aligned and free of NaNs before calculating metrics
        temp_df = pd.DataFrame({'y_true': y_test, 'y_pred': y_pred_test})
        temp_df.dropna(inplace=True)

        y_test_clean = temp_df['y_true'].values
        y_pred_test_clean = temp_df['y_pred'].values  # This is the clean prediction for metrics

        # Also, clean y_train and y_pred_train for training metrics
        temp_train_df = pd.DataFrame({'y_true': y_train, 'y_pred': y_pred_train})
        temp_train_df.dropna(inplace=True)

        y_train_clean = temp_train_df['y_true'].values
        y_pred_train_clean = temp_train_df['y_pred'].values
        # --- CRITICAL FIX END ---

        # Metrics - use the cleaned arrays
        train_mse = mean_squared_error(y_train_clean, y_pred_train_clean)
        test_mse = mean_squared_error(y_test_clean, y_pred_test_clean)
        train_mae = mean_absolute_error(y_train_clean, y_pred_train_clean)
        test_mae = mean_absolute_error(y_test_clean, y_pred_test_clean)

        self.models['linear'] = model
        self.metrics['linear'] = {
            'train_mse': train_mse,
            'test_mse': test_mse,
            'train_mae': train_mae,
            'test_mae': test_mae,
            # DO NOT STORE y_pred_test directly in self.metrics for comparison DF
        }
        self.predictions['linear'] = y_pred_test_clean  # Store cleaned test predictions here

        logger.info(f"Linear Model - Test MSE: {test_mse:.6f}, Test MAE: {test_mae:.6f}")
        return model

    def train_random_forest(self, X_train, y_train, X_test, y_test, n_estimators=50):
        """Train a random forest model"""
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            random_state=config.RANDOM_SEED,
            max_depth=10
        )

        # Reshape data for sklearn
        X_train_flat = X_train.reshape(X_train.shape[0], -1)
        X_test_flat = X_test.reshape(X_test.shape[0], -1)

        # Ensure y_train is 1D for sklearn models if it somehow became 2D
        if y_train.ndim > 1:
            y_train = y_train.ravel()

        model.fit(X_train_flat, y_train)

        # Predictions
        y_pred_train = model.predict(X_train_flat)
        y_pred_test = model.predict(X_test_flat)  # Keep this raw prediction

        # --- CRITICAL FIX START (re-emphasized) ---
        # Ensure y_test and y_pred_test are aligned and free of NaNs before calculating metrics
        temp_df = pd.DataFrame({'y_true': y_test, 'y_pred': y_pred_test})
        temp_df.dropna(inplace=True)

        y_test_clean = temp_df['y_true'].values
        y_pred_test_clean = temp_df['y_pred'].values  # This is the clean prediction for metrics

        # Also, clean y_train and y_pred_train for training metrics
        temp_train_df = pd.DataFrame({'y_true': y_train, 'y_pred': y_pred_train})
        temp_train_df.dropna(inplace=True)

        y_train_clean = temp_train_df['y_true'].values
        y_pred_train_clean = temp_train_df['y_pred'].values
        # --- CRITICAL FIX END ---

        # Metrics - use the cleaned arrays
        train_mse = mean_squared_error(y_train_clean, y_pred_train_clean)
        test_mse = mean_squared_error(y_test_clean, y_pred_test_clean)
        train_mae = mean_absolute_error(y_train_clean, y_pred_train_clean)
        test_mae = mean_absolute_error(y_test_clean, y_pred_test_clean)

        self.models['random_forest'] = model
        self.metrics['random_forest'] = {
            'train_mse': train_mse,
            'test_mse': test_mse,
            'train_mae': train_mae,
            'test_mae': test_mae,
            # DO NOT STORE y_pred_test directly in self.metrics for comparison DF
        }
        self.predictions['random_forest'] = y_pred_test_clean  # Store cleaned test predictions here

        logger.info(f"Random Forest - Test MSE: {test_mse:.6f}, Test MAE: {test_mae:.6f}")
        return model

    def train_svr_model(self, X_train, y_train, X_test, y_test,
                        kernel='rbf', C=1.0, epsilon=0.1, gamma='scale'):
        """
        Train a Support Vector Regression (SVR) model.

        Args:
            X_train, y_train: Training data and labels.
            X_test, y_test: Testing data and labels.
            kernel (str): Specifies the kernel type to be used in the algorithm.
                          Common values: 'linear', 'poly', 'rbf', 'sigmoid'.
            C (float): Regularization parameter. The strength of the regularization is
                       inversely proportional to C. Must be strictly positive.
            epsilon (float): Epsilon in the epsilon-SVR model. It specifies the epsilon-tube
                             within which no penalty is associated in the training loss function
                             with points predicted within a distance epsilon from the actual value.
            gamma (str or float): Kernel coefficient for 'rbf', 'poly' and 'sigmoid'.
                                  'scale' means 1 / (n_features * X.var())
                                  'auto' means 1 / n_features
        """
        logger.info(f"Training SVR model with kernel={kernel}, C={C}, epsilon={epsilon}, gamma={gamma}...")
        model = SVR(kernel=kernel, C=C, epsilon=epsilon, gamma=gamma)

        # Reshape data for sklearn (flatten sequences)
        # X_train is (samples, window_size, features_per_step)
        # X_train_flat should be (samples, window_size * features_per_step)
        X_train_flat = X_train.reshape(X_train.shape[0], -1)
        X_test_flat = X_test.reshape(X_test.shape[0], -1)

        # Ensure y_train is 1D for sklearn models
        if y_train.ndim > 1:
            y_train = y_train.ravel()

        model.fit(X_train_flat, y_train)

        # Predictions
        y_pred_train = model.predict(X_train_flat)
        y_pred_test = model.predict(X_test_flat)

        # --- CRITICAL FIX for aligning y_true and y_pred and handling potential NaNs ---
        # (Copied from your other model methods for consistency)
        temp_df_test = pd.DataFrame({'y_true': y_test, 'y_pred': y_pred_test})
        temp_df_test.dropna(inplace=True)
        y_test_clean = temp_df_test['y_true'].values
        y_pred_test_clean = temp_df_test['y_pred'].values

        temp_df_train = pd.DataFrame({'y_true': y_train, 'y_pred': y_pred_train})
        temp_df_train.dropna(inplace=True)
        y_train_clean = temp_df_train['y_true'].values
        y_pred_train_clean = temp_df_train['y_pred'].values
        # --- END OF CRITICAL FIX ---

        # Calculate metrics - use the cleaned arrays
        # Ensure there's data to calculate metrics on after cleaning
        if len(y_train_clean) > 0 and len(y_test_clean) > 0:
            train_mse = mean_squared_error(y_train_clean, y_pred_train_clean)
            test_mse = mean_squared_error(y_test_clean, y_pred_test_clean)
            train_mae = mean_absolute_error(y_train_clean, y_pred_train_clean)
            test_mae = mean_absolute_error(y_test_clean, y_pred_test_clean)

            logger.info(f"SVR Model - Test MSE: {test_mse:.6f}, Test MAE: {test_mae:.6f}")
        else:
            logger.warning(
                "SVR Model - Not enough data to calculate metrics after cleaning NaNs from predictions/actuals.")
            train_mse, test_mse, train_mae, test_mae = np.nan, np.nan, np.nan, np.nan

        self.models['svr'] = model
        self.metrics['svr'] = {
            'train_mse': train_mse,
            'test_mse': test_mse,
            'train_mae': train_mae,
            'test_mae': test_mae,
        }
        # Store cleaned test predictions
        self.predictions['svr'] = y_pred_test_clean if len(y_test_clean) > 0 else np.array([])

        return model

    def train_xgboost_model(self, X_train, y_train, X_test, y_test,
                            n_estimators=100, learning_rate=0.1, max_depth=5,
                            early_stopping_rounds=10, verbosity=0, **kwargs):  # Added verbosity
        """
        Train an XGBoost regression model.

        Args:
            X_train, y_train: Training data and labels.
            X_test, y_test: Testing data and labels.
            n_estimators (int): Number of gradient boosted trees. Equivalent to number of boosting rounds.
            learning_rate (float): Boosting learning rate (xgb's "eta")
            max_depth (int): Maximum depth of a tree.
            early_stopping_rounds (int): Activates early stopping. Validation error needs to decrease at least
                                         every <early_stopping_rounds> round(s) to continue training.
                                         Requires an eval_set.
            verbosity (int): Verbosity of printing messages. 0 (silent) - 3 (debug).
            **kwargs: Other XGBoost specific parameters.
        """
        logger.info(
            f"Training XGBoost model with n_estimators={n_estimators}, learning_rate={learning_rate}, max_depth={max_depth}...")

        model = xgb.XGBRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=config.RANDOM_SEED,  # For reproducibility
            objective='reg:squarederror',  # Common objective for regression
            early_stopping_rounds=early_stopping_rounds,
            verbosity=verbosity,  # Set verbosity
            **kwargs  # Pass any other xgb specific params
        )

        # Reshape data for XGBoost (flatten sequences)
        X_train_flat = X_train.reshape(X_train.shape[0], -1)
        X_test_flat = X_test.reshape(X_test.shape[0], -1)

        # Ensure y_train is 1D
        if y_train.ndim > 1:
            y_train = y_train.ravel()

        # XGBoost can use an evaluation set for early stopping
        eval_set = [(X_test_flat, y_test)]

        model.fit(X_train_flat, y_train, eval_set=eval_set,
                  verbose=False)  # verbose=False here to use xgb's verbosity param

        # Predictions
        y_pred_train = model.predict(X_train_flat)
        y_pred_test = model.predict(X_test_flat)

        # --- CRITICAL FIX for aligning y_true and y_pred and handling potential NaNs ---
        temp_df_test = pd.DataFrame({'y_true': y_test, 'y_pred': y_pred_test})
        temp_df_test.dropna(inplace=True)
        y_test_clean = temp_df_test['y_true'].values
        y_pred_test_clean = temp_df_test['y_pred'].values

        temp_df_train = pd.DataFrame({'y_true': y_train, 'y_pred': y_pred_train})
        temp_df_train.dropna(inplace=True)
        y_train_clean = temp_df_train['y_true'].values
        y_pred_train_clean = temp_df_train['y_pred'].values
        # --- END OF CRITICAL FIX ---

        # Calculate metrics
        if len(y_train_clean) > 0 and len(y_test_clean) > 0:
            train_mse = mean_squared_error(y_train_clean, y_pred_train_clean)
            test_mse = mean_squared_error(y_test_clean, y_pred_test_clean)
            train_mae = mean_absolute_error(y_train_clean, y_pred_train_clean)
            test_mae = mean_absolute_error(y_test_clean, y_pred_test_clean)
            logger.info(f"XGBoost Model - Test MSE: {test_mse:.6f}, Test MAE: {test_mae:.6f}")
            if model.best_iteration is not None:  # If early stopping was used
                logger.info(f"XGBoost Model - Best iteration: {model.best_iteration}")
        else:
            logger.warning("XGBoost Model - Not enough data to calculate metrics after cleaning.")
            train_mse, test_mse, train_mae, test_mae = np.nan, np.nan, np.nan, np.nan

        self.models['xgboost'] = model
        self.metrics['xgboost'] = {
            'train_mse': train_mse,
            'test_mse': test_mse,
            'train_mae': train_mae,
            'test_mae': test_mae,
        }
        self.predictions['xgboost'] = y_pred_test_clean if len(y_test_clean) > 0 else np.array([])

        return model

    def simple_moving_average_forecast(self, data, window=5):
        """Simple moving average forecast"""
        if len(data) < window:
            # Handle cases where data is too short for the window
            if len(data) > 0:
                return np.mean(data)
            return np.nan  # Return NaN if data is empty
        return np.mean(data[-window:])

    def predict_next(self, model_name, X_last):
        """Predict next value using trained model"""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not trained yet")

        model = self.models[model_name]

        # Ensure X_last is correctly reshaped for prediction
        # X_last might be a single sequence (e.g., X_test[0]), so reshape it to (1, -1)
        if X_last.ndim == 2:  # if X_last is (window_size, num_features)
            X_flat = X_last.reshape(1, -1)
        elif X_last.ndim == 1:  # if X_last is (num_features,) already flattened
            X_flat = X_last.reshape(1, -1)
        else:  # assuming X_last is (num_features,) from an already flattened context
            X_flat = X_last.reshape(1, -1)

        prediction = model.predict(X_flat)[0]

        return prediction

    def get_model_comparison(self):
        """Get comparison of all trained models"""
        if not self.metrics:
            logger.info("No models trained yet for comparison.")
            return pd.DataFrame()

            # --- START OF DEBUGGING SECTION ---
        logger.debug("--- DEBUGGING self.metrics content in get_model_comparison ---")
        for model_name, metrics_dict in self.metrics.items():
            logger.debug(f"  Model: {model_name}")
            for metric_name, value in metrics_dict.items():
                logger.debug(f"    Metric: {metric_name}, Type: {type(value)}, Value: {value}")
                if isinstance(value, (list, np.ndarray)):
                    logger.debug(f"    !!! WARNING: Array/List detected for {metric_name}. Length: {len(value)}")
        logger.debug("--- END OF DEBUGGING SECTION ---")

        try:
            comparison = pd.DataFrame(self.metrics).T
            logger.debug("Successfully created DataFrame in get_model_comparison.")
        except Exception as e:
            logger.error(f"Error creating DataFrame in get_model_comparison: {e}", exc_info=True)
            # Re-raise the exception to ensure it's propagated and the program stops
            raise

        comparison = comparison.round(6)

        return comparison

    def backtest_simple_strategy(self, y_true, model_name, initial_balance=10000):  # Updated signature
        """
        Simple backtesting: buy if predicted price > current, sell otherwise

        Args:
            y_true: Actual prices
            model_name: Name of the model whose predictions to use for backtesting
            initial_balance: Starting balance

        Returns:
            Dictionary with backtest results
        """
        if model_name not in self.predictions:
            logger.error(f"Predictions for model '{model_name}' not found for backtesting.")
            return {'initial_balance': initial_balance, 'final_balance': initial_balance, 'total_return': 0,
                    'num_trades': 0, 'trades': []}

        y_pred = self.predictions[model_name]  # Retrieve predictions from the new 'predictions' dict

        # Ensure y_true and y_pred are clean and aligned for backtesting
        temp_df = pd.DataFrame({'y_true': y_true, 'y_pred': y_pred})
        temp_df.dropna(inplace=True)  # Drop any NaNs

        y_true_clean = temp_df['y_true'].values
        y_pred_clean = temp_df['y_pred'].values

        if len(y_true_clean) < 2:  # Need at least 2 points for a "current price" and "actual price" transition
            logger.warning("Not enough clean data points for backtesting after dropping NaNs. Skipping backtest.")
            return {'initial_balance': initial_balance, 'final_balance': initial_balance, 'total_return': 0,
                    'num_trades': 0, 'trades': []}

        balance = initial_balance
        position = 0  # 0 = no position, 1 = long
        trades = []

        for i in range(1, len(y_pred_clean)):  # Iterate from 1 as we need y_true[i-1] for current price
            current_price = y_true_clean[i - 1]
            predicted_price = y_pred_clean[i - 1]  # Prediction made for the next step (i) based on data up to i-1
            actual_price = y_true_clean[i]  # The actual price at step i, which we act upon

            # Simple strategy: go long if prediction for *next step* > current price
            if predicted_price > current_price and position == 0:
                # Buy signal
                position = 1
                buy_price = actual_price  # We buy at the actual price of the current time step (i)
                trades.append({'action': 'buy', 'price': buy_price, 'balance_before_trade': balance})

            elif predicted_price <= current_price and position == 1:
                # Sell signal
                position = 0
                sell_price = actual_price  # We sell at the actual price of the current time step (i)

                # Find the corresponding buy trade to calculate profit
                # This simple logic assumes one buy/sell cycle at a time
                last_buy_trade_idx = -1
                for j in range(len(trades) - 1, -1, -1):
                    if trades[j]['action'] == 'buy':
                        last_buy_trade_idx = j
                        break

                if last_buy_trade_idx != -1:
                    buy_price = trades[last_buy_trade_idx]['price']
                    profit = (sell_price - buy_price) / buy_price
                    balance *= (1 + profit)
                    trades[last_buy_trade_idx]['balance_after_trade'] = balance  # Update balance in buy trade
                    trades.append(
                        {'action': 'sell', 'price': sell_price, 'balance_after_trade': balance, 'profit': profit})
                else:
                    logger.warning(
                        f"Sell signal at index {i} but no matching buy trade found. Skipping profit calculation for this sell.")

        # If still in a position at the end, close it at the last available price
        if position == 1 and len(y_true_clean) > 0:
            last_price = y_true_clean[-1]
            last_buy_trade_idx = -1
            for j in range(len(trades) - 1, -1, -1):
                if trades[j]['action'] == 'buy':
                    last_buy_trade_idx = j
                    break
            if last_buy_trade_idx != -1:
                buy_price = trades[last_buy_trade_idx]['price']
                profit = (last_price - buy_price) / buy_price
                balance *= (1 + profit)
                trades[last_buy_trade_idx]['balance_after_trade'] = balance
                trades.append(
                    {'action': 'sell_at_end', 'price': last_price, 'balance_after_trade': balance, 'profit': profit})
            else:
                logger.warning("Still in a long position at the end but no buy trade found to close.")

        total_return = (balance - initial_balance) / initial_balance
        num_trades = len([t for t in trades if t['action'].startswith('sell')])  # Count all sell trades

        results = {
            'initial_balance': initial_balance,
            'final_balance': balance,
            'total_return': total_return,
            'num_trades': num_trades,
            'trades': trades
        }

        logger.info(f"Backtest results: {total_return:.2%} return, {num_trades} trades")
        return results


# Create global forecaster instance
forecaster = FXForecaster()
