"""
Financial Time Series Toolkit - Advanced Implementation Test
Testing sophisticated methodologies for financial prediction
"""
import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew, jarque_bera
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import traceback
import logging


# It's good practice to have imports at the top,
# but the original structure places them inside functions/try blocks.
# For this combination, I will keep the original structure of imports within main().

def main():
    # --- ADD THIS LINE HERE ---
    logging.basicConfig(level=logging.DEBUG, format='%(levelname)s:%(name)s:%(message)s')
    # --- END OF ADDITION ---

    print("🚀 Advanced Financial Time Series Toolkit - Comprehensive Test")
    print("=" * 80)

    # Initialize acf_squared_returns and kurt to default values
    # in case their computation is skipped due to data size.
    acf_squared_returns = 0.0
    kurt = 0.0

    try:
        # Initialize configuration
        from src.fts_toolkit.config import config
        config.setup_directories()
        config.print_config()

        # Get data
        print("\n📊 Testing Advanced Data Pipeline...")
        from src.fts_toolkit.scraper import scraper

        # Get more data for robust testing
        df = scraper.get_fx_data_yahoo(symbol=config.DEFAULT_SYMBOL,
                                       days=config.LOOKBACK_DAYS)  # Fetch ~2 years of data
        print(f"    Fetching data for {config.LOOKBACK_DAYS} days...")  # Optional: log what you're fetching
        print(f"    ✅ Got {len(df)} data points")
        print(f"    📅 Date range: {df['Date'].min()} to {df['Date'].max()}")

        # Advanced data processing with stationarity testing
        print("\n⚙️ Testing Advanced Data Processing...")
        from src.fts_toolkit.advanced_processor import advanced_processor

        # Test stationarity of close prices
        close_stationarity = advanced_processor.test_stationarity(df['Close'], 'Close Prices')

        # Test stationarity of returns
        df['log_returns'] = np.log(df['Close'] / df['Close'].shift(1))
        returns_stationarity = advanced_processor.test_stationarity(
            df['log_returns'].dropna(), 'Log Returns'
        )

        # Advanced feature engineering
        df_processed = advanced_processor.engineer_advanced_features(df)
        print(f"    ✅ Advanced features: {df_processed.shape[1]} columns")

        # Create stationary sequences for ML
        X_train = X_test = y_train = y_test = None  # Initialize
        n_components_for_pca = 0.95  # Initialize to None (PCA skipped by default)

        try:
            if n_components_for_pca is not None:
                print(
                    f"    🧪 Attempting ML sequence creation with PCA, aiming for n_components/variance: {n_components_for_pca}")
            else:
                print(f"    🧪 Attempting ML sequence creation WITHOUT PCA.")

            X_train, X_test, y_train, y_test = advanced_processor.prepare_training_data(
                df_processed,
                test_size=0.3,
                n_pca_components=n_components_for_pca  # Pass the PCA parameter here
            )

            # Check if sequence creation was successful (X_train would not be empty or None)
            if X_train is not None and X_train.shape[0] > 0:
                print(f"    🎯 ML sequences created: Train {X_train.shape}, Test {X_test.shape}")
                if n_components_for_pca is not None:
                    # X_train.shape[2] will be the actual number of components PCA selected if n_components_for_pca
                    # was a float (e.g. 0.95) or it will be the integer you specified.
                    print(f"    ✨ PCA resulted in {X_train.shape[2]} features.")
            else:
                # This case might be redundant if prepare_training_data raises an error or returns empty arrays
                # that are caught by the except block, but it's a safeguard.
                print(f"    ⚠️ ML sequence creation did not return data, even if no exception was raised.")
                X_train = None  # Ensure X_train is None to skip subsequent ML model training

        except Exception as e:
            print(f"    ⚠️ ML sequence creation failed: {e}")
            X_train = None  # Ensure X_train is None if an exception occurs

        # Test Econometric Models
        print("\n📈 Testing Econometric Models...")
        from src.fts_toolkit.econometric_models import arima_model, garch_model, arima_garch_model

        # Prepare returns series for econometric modeling
        returns_series = df['log_returns'].dropna()
        price_series = df['Close'].dropna()

        if len(returns_series) > 50:  # Minimum data requirement

            # Test ARIMA model
            print("\n    🔍 ARIMA Model Testing:")
            try:
                arima_fitted = arima_model.fit(price_series)
                arima_diagnostics = arima_model.diagnostic_check()
                arima_forecast = arima_model.forecast(steps=5)

                print(f"    ✅ ARIMA{arima_model.order} fitted successfully")
                print(f"    📊 AIC: {arima_fitted.aic:.4f}")
                print(f"    🔮 5-step forecast: {arima_forecast['forecast'].iloc[-1]:.5f}")

            except Exception as e:
                print(f"    ❌ ARIMA fitting failed: {e}")

            # Test GARCH model
            print("\n    📊 GARCH Model Testing:")
            try:
                garch_fitted = garch_model.fit(returns_series, order=(1, 1))
                garch_diagnostics = garch_model.diagnostic_check()
                garch_vol_forecast = garch_model.forecast_volatility(horizon=5)

                print(f"    ✅ GARCH(1,1) fitted successfully")
                print(f"    📊 Log-likelihood: {garch_fitted.loglikelihood:.4f}")
                print(f"    🔮 Volatility forecast: {garch_vol_forecast['volatility_forecast'][0]:.6f}")

            except Exception as e:
                print(f"    ❌ GARCH fitting failed: {e}")

            # Test Hybrid ARIMA-GARCH
            print("\n    🤝 Hybrid ARIMA-GARCH Testing:")
            try:
                hybrid_fitted = arima_garch_model.fit(price_series)
                hybrid_forecast = arima_garch_model.forecast(steps=5)

                print(f"    ✅ Hybrid model fitted successfully")
                print(f"    📊 Model type: {hybrid_forecast['model_type']}")
                print(f"    🔮 Mean forecast: {hybrid_forecast['mean_forecast'].iloc[-1]:.5f}")

                if hybrid_forecast['volatility_forecast'] is not None:
                    print(f"    📈 Volatility forecast: {hybrid_forecast['volatility_forecast'][0]:.6f}")

            except Exception as e:
                print(f"    ❌ Hybrid modeling failed: {e}")

        # Enhanced ML Models (if sequences were created)
        if X_train is not None and y_train is not None and X_train.size > 0 and X_test.size > 0:
            print("\n🤖 Testing Enhanced ML Models...")
            from src.fts_toolkit.models import forecaster

            try:
                # Train models
                linear_model = forecaster.train_linear_model(X_train, y_train, X_test, y_test)
                rf_model = forecaster.train_random_forest(X_train, y_train, X_test, y_test, n_estimators=100)
                svr_model = forecaster.train_svr_model(X_train, y_train, X_test, y_test)
                xgboost_model = forecaster.train_xgboost_model(
                    X_train, y_train, X_test, y_test,
                    n_estimators=100, learning_rate=0.1, max_depth=3, early_stopping_rounds=10
                )

                # Model comparison
                print("\n📊 Enhanced Model Comparison:")
                comparison = forecaster.get_model_comparison()
                print(comparison[['test_mse', 'test_mae']].round(6))

                # --- GENERATE FLAT FEATURE NAMES ONCE (after X_train is confirmed valid) ---
                flat_feature_names = []  # Initialize
                num_features_per_step = X_train.shape[2]
                base_feature_names = []

                if n_components_for_pca is not None:  # Check if PCA was applied
                    base_feature_names = [f"PC_{i}" for i in range(num_features_per_step)]
                    print(
                        f"    📊 Generating feature importance names for {num_features_per_step} Principal Components.")
                elif hasattr(advanced_processor, 'feature_columns') and advanced_processor.feature_columns:
                    base_feature_names = advanced_processor.feature_columns
                    print(
                        f"    📊 Generating feature importance names for {len(base_feature_names)} original engineered features.")
                else:
                    base_feature_names = [f"feature_{i}" for i in range(num_features_per_step)]  # Fallback
                    print(f"    📊 Using generic base feature names for importance.")

                window_steps = X_train.shape[1]  # This is your window_size
                for i in range(window_steps):
                    for col_name in base_feature_names:
                        flat_feature_names.append(f"{col_name}_t-{window_steps - 1 - i}")
                # --- END OF FLAT FEATURE NAMES GENERATION ---

                # Feature importance for Random Forest
                if hasattr(rf_model, 'feature_importances_'):
                    if len(flat_feature_names) == len(rf_model.feature_importances_):
                        feature_importance_rf = pd.DataFrame({
                            'feature': flat_feature_names,
                            'importance': rf_model.feature_importances_
                        }).sort_values('importance', ascending=False).head(10)
                        print("\n🎯 Top 10 Most Important Features for Random Forest (Flattened):")
                        for _, row_rf in feature_importance_rf.iterrows():
                            print(f"    {row_rf['feature']}: {row_rf['importance']:.4f}")
                    else:
                        print(
                            f"    ⚠️ Could not display RF feature importance: Mismatch in names length ({len(flat_feature_names)}) vs importances length ({len(rf_model.feature_importances_)}). Using generic.")
                        # Fallback for RF if needed
                        generic_rf_flat_names = [f"rf_flat_feature_{k}" for k in
                                                 range(len(rf_model.feature_importances_))]
                        feature_importance_rf_generic = pd.DataFrame({
                            'feature': generic_rf_flat_names,
                            'importance': rf_model.feature_importances_
                        }).sort_values('importance', ascending=False).head(10)
                        print("\n🎯 Top 10 Most Important Features for Random Forest (Generic Flattened, Debug):")
                        for _, row_rf_gen in feature_importance_rf_generic.iterrows():
                            print(f"    {row_rf_gen['feature']}: {row_rf_gen['importance']:.4f}")

                # Feature importance for XGBoost
                if hasattr(xgboost_model, 'feature_importances_'):
                    if len(flat_feature_names) == len(xgboost_model.feature_importances_):
                        feature_importance_xgb = pd.DataFrame({
                            'feature': flat_feature_names,  # Reusing the same flat_feature_names
                            'importance': xgboost_model.feature_importances_
                        }).sort_values('importance', ascending=False).head(10)
                        print("\n🎯 Top 10 Most Important Features for XGBoost (Flattened):")
                        for _, row_xgb in feature_importance_xgb.iterrows():
                            print(f"    {row_xgb['feature']}: {row_xgb['importance']:.4f}")
                    else:
                        print(
                            f"    ⚠️ Could not display XGBoost feature importance: Mismatch in names length ({len(flat_feature_names)}) vs importances length ({len(xgboost_model.feature_importances_)}). Using generic.")
                        # Fallback for XGBoost if needed
                        generic_xgb_flat_names = [f"xgb_flat_feature_{k}" for k in
                                                  range(len(xgboost_model.feature_importances_))]
                        feature_importance_xgb_generic = pd.DataFrame({
                            'feature': generic_xgb_flat_names,
                            'importance': xgboost_model.feature_importances_
                        }).sort_values('importance', ascending=False).head(10)
                        print("\n🎯 Top 10 Most Important Features for XGBoost (Generic Flattened, Debug):")
                        for _, row_xgb_gen in feature_importance_xgb_generic.iterrows():
                            print(f"    {row_xgb_gen['feature']}: {row_xgb_gen['importance']:.4f}")

            except Exception as e:
                print(f"    ❌ Enhanced ML modeling failed: {e}")
                traceback.print_exc()  # Print full traceback for ML errors
        else:
            print("\n🤖 Skipping Enhanced ML Models training as no valid sequence data is available.")

        # Advanced Performance Metrics
        print("\n📊 Advanced Performance Analysis...")

        # Market regime analysis
        print("\n    📈 Market Regime Analysis:")
        df['volatility_regime'] = np.where(
            df['log_returns'].rolling(20).std() > df['log_returns'].rolling(60).std().mean(),
            'High Volatility', 'Low Volatility'
        )

        regime_counts = df['volatility_regime'].value_counts()
        print(f"    High Volatility periods: {regime_counts.get('High Volatility', 0)} days")
        print(f"    Low Volatility periods: {regime_counts.get('Low Volatility', 0)} days")

        # Stylized facts analysis
        print("\n    📊 Stylized Facts Analysis:")
        returns_clean = returns_series.dropna()

        if len(returns_clean) > 30:
            # Volatility clustering test
            acf_returns = np.abs(returns_clean).autocorr(lag=1)  # Original variable name
            acf_squared_returns = (returns_clean ** 2).autocorr(lag=1)  # Will overwrite default if computed

            print(f"    Volatility clustering (|returns| ACF): {acf_returns:.4f}")
            print(f"    Volatility persistence (returns² ACF): {acf_squared_returns:.4f}")

            # Excess kurtosis (fat tails)
            kurt = kurtosis(returns_clean, fisher=True)  # Excess kurtosis, will overwrite default
            skewness = skew(returns_clean)

            print(f"    Excess kurtosis (fat tails): {kurt:.4f}")
            print(f"    Skewness: {skewness:.4f}")

            # Jarque-Bera normality test
            jb_stat, jb_pvalue = jarque_bera(returns_clean)
            print(f"    Jarque-Bera test p-value: {jb_pvalue:.4f} ({'Non-normal' if jb_pvalue < 0.05 else 'Normal'})")

        # Walk-forward validation simulation
        print("\n🔄 Walk-Forward Validation Simulation...")
        if len(df_processed) > 120:  # Need sufficient data in df_processed
            try:
                results = simulate_walk_forward_validation(df_processed, window_size=60, test_size=10)
                if results['rmse_scores']:  # Check if simulation ran successfully
                    print(f"    ✅ Walk-forward validation completed")
                    print(f"    📊 Average RMSE: {np.mean(results['rmse_scores']):.6f}")
                    print(f"    📊 RMSE std: {np.std(results['rmse_scores']):.6f}")
                    print(f"    🔄 Number of validation windows: {len(results['rmse_scores'])}")
                else:
                    print(f"    ⚠️ Walk-forward validation did not produce results. Check data and parameters.")
            except Exception as e:
                print(f"    ❌ Walk-forward validation failed: {e}")
        else:
            print(f"    ℹ️ Insufficient data for walk-forward validation (need > 120, got {len(df_processed)})")

        # Risk metrics
        print("\n⚠️ Risk Analysis...")
        if len(returns_clean) > 30:
            # Value at Risk (VaR)
            var_95 = np.percentile(returns_clean, 5)  # 5% VaR
            var_99 = np.percentile(returns_clean, 1)  # 1% VaR

            print(f"    VaR (95%): {var_95:.4f} ({var_95 * 100:.2f}%)")
            print(f"    VaR (99%): {var_99:.4f} ({var_99 * 100:.2f}%)")

            # Maximum Drawdown
            cumulative_returns = (1 + returns_clean).cumprod()
            rolling_max = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - rolling_max) / rolling_max
            max_drawdown = drawdown.min()

            print(f"    Maximum Drawdown: {max_drawdown:.4f} ({max_drawdown * 100:.2f}%)")

        # Model interpretation and insights
        print("\n🔍 Model Insights and Recommendations...")

        print("    📝 Stationarity Analysis:")
        print(f"    • Price series: {close_stationarity['recommendation']}")
        print(f"    • Return series: {returns_stationarity['recommendation']}")

        print("\n    📝 Model Recommendations:")
        if close_stationarity['recommendation'] != 'stationary':
            print("    • Use differenced prices or returns for ARIMA modeling")

        if acf_squared_returns > 0.1:  # Uses value computed (or default 0.0)
            print("    • Strong volatility clustering detected → GARCH modeling recommended")
        if kurt > 3:  # Uses value computed (or default 0.0)
            print("    • Fat tails detected → Consider t-distribution for GARCH")

        print("\n    📝 Practical Considerations:")
        print("    • Short-term forecasts (1-5 days): ARIMA-GARCH hybrid recommended")
        print("    • Medium-term forecasts (1-4 weeks): ML models with regime detection")
        print("    • Long-term forecasts (>1 month): Fundamental analysis required")

        print("\n✅ Advanced Financial Time Series Analysis Completed!")
        print("\n🎯 Next Steps for Production Implementation:")
        print("    1. Implement Vector Autoregression (VAR) for multi-currency analysis")
        print("    2. Add LSTM/GRU models with proper sequence handling")
        print("    3. Develop regime-switching models")
        print("    4. Implement real-time data feeds")
        print("    5. Add portfolio optimization and risk management")
        print("    6. Create interactive dashboard with plotly/dash")

    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("    Install required packages: pip install statsmodels arch scipy pandas numpy scikit-learn")
    except Exception as e:
        print(f"❌ Error in main: {e}")
        traceback.print_exc()


def simulate_walk_forward_validation(df_input, window_size=60, test_size=10):
    """Simulate walk-forward validation using a simple Linear Regression model."""
    # Module imports for this function are kept local as in the original snippet
    # from src.fts_toolkit.advanced_processor import advanced_processor # Not directly used

    results = {'rmse_scores': [], 'predictions': [], 'actuals': []}

    total_needed_for_one_fold = window_size + test_size
    # The original prompt had total_needed * 2 for df length check,
    # let's assume it meant at least two folds possible.
    # A single fold requires at least `total_needed_for_one_fold` rows in df_clean.
    if len(df_input) < total_needed_for_one_fold:
        raise ValueError(
            f"Insufficient data for even one walk-forward validation fold. "
            f"Need at least {total_needed_for_one_fold} rows, got {len(df_input)}"
        )

    numeric_cols = df_input.select_dtypes(include=[np.number]).columns.tolist()

    if 'Close' not in numeric_cols:
        raise ValueError("'Close' column not found in the numeric columns of the DataFrame.")

    feature_cols = [col for col in numeric_cols if col not in ['Close', 'Date']]

    if not feature_cols:
        raise ValueError("No feature columns found after excluding 'Close' and 'Date'. Ensure features are numeric.")

    # Ensure 'Close' is present for target, and features are present
    df_clean = df_input[feature_cols + ['Close']].copy()  # Use .copy() to avoid SettingWithCopyWarning
    df_clean.dropna(inplace=True)  # Drop rows with NaNs in selected features or target

    if len(df_clean) < total_needed_for_one_fold:
        raise ValueError(
            f"Insufficient data after dropping NaNs for walk-forward validation. "
            f"Need {total_needed_for_one_fold} rows, got {len(df_clean)}"
        )

    start_idx = 0
    while start_idx + total_needed_for_one_fold <= len(df_clean):
        train_end = start_idx + window_size
        test_end = train_end + test_size

        train_data = df_clean.iloc[start_idx:train_end]
        test_data = df_clean.iloc[train_end:test_end]

        if train_data.empty or test_data.empty:
            print(f"⚠️ Skipping fold: Empty train or test data at start_idx {start_idx}")
            start_idx += test_size
            continue

        X_train = train_data[feature_cols].values
        y_train = train_data['Close'].values
        X_test = test_data[feature_cols].values
        y_test = test_data['Close'].values

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model = LinearRegression()
        model.fit(X_train_scaled, y_train)

        y_pred = model.predict(X_test_scaled)

        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        results['rmse_scores'].append(rmse)
        results['predictions'].extend(y_pred.tolist())  # Ensure predictions are lists
        results['actuals'].extend(y_test.tolist())  # Ensure actuals are lists

        start_idx += test_size

    return results


if __name__ == "__main__":
    main()
