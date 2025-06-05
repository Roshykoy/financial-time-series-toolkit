"""
Financial Time Series Toolkit - Advanced Implementation Test
Testing sophisticated methodologies for financial prediction
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
        from src.fts_toolkit.config import config  # Ensure this is how you import your config
        config.setup_directories()
        config.print_config()  # This will show your loaded configurations

        # Get data
        print("\n📊 Testing Advanced Data Pipeline...")
        from src.fts_toolkit.scraper import scraper

        df = scraper.get_fx_data_yahoo(symbol=config.DEFAULT_SYMBOL,
                                       days=config.LOOKBACK_DAYS)
        print(f"    Fetching data for {config.LOOKBACK_DAYS} days...")
        if df.empty:
            print(f"    ❌ Could not fetch data for {config.DEFAULT_SYMBOL}. Exiting.")
            return
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

        # --- Get PCA config from config.py ---
        pca_config_value = str(config.PCA_N_COMPONENTS_CONFIG)  # Ensure it's a string for .lower()
        n_components_for_pca = None  # Default to skipping PCA

        if pca_config_value.lower() != 'none' and pca_config_value:
            try:
                # Try to convert to float (for variance e.g., 0.95)
                n_components_for_pca_float = float(pca_config_value)
                if 0 < n_components_for_pca_float <= 1.0:
                    n_components_for_pca = n_components_for_pca_float
                elif n_components_for_pca_float >= 1.0:  # Treat as integer number of components
                    n_components_for_pca = int(n_components_for_pca_float)
                else:  # Invalid float (e.g., negative)
                    logger.warning(
                        f"    ⚠️ Invalid float value for PCA_N_COMPONENTS: '{pca_config_value}'. Skipping PCA.")
                    n_components_for_pca = None
            except ValueError:
                try:
                    # Try to convert to int (for specific number of components e.g., 10)
                    n_components_for_pca = int(pca_config_value)
                    if n_components_for_pca < 1:  # Number of components must be at least 1
                        logger.warning(
                            f"    ⚠️ PCA_N_COMPONENTS must be >= 1 if integer: '{pca_config_value}'. Skipping PCA.")
                        n_components_for_pca = None
                except ValueError:
                    logger.warning(
                        f"    ⚠️ Invalid value for PCA_N_COMPONENTS in config: '{pca_config_value}'. Skipping PCA.")
                    n_components_for_pca = None
        else:
            logger.info("    PCA_N_COMPONENTS is 'None' or empty in config. Skipping PCA.")
        # --- End of PCA config handling ---

        # Create stationary sequences for ML
        X_train = X_test = y_train = y_test = None  # Initialize

        try:
            if n_components_for_pca is not None:
                print(
                    f"    🧪 Attempting ML sequence creation with PCA, aiming for n_components/variance: {n_components_for_pca}")
            else:
                print(f"    🧪 Attempting ML sequence creation WITHOUT PCA.")

            X_train, X_test, y_train, y_test = advanced_processor.prepare_training_data(
                df_processed,
                test_size=config.ML_TEST_SIZE,  # Use from config
                n_pca_components=n_components_for_pca  # Use parsed value from config
            )

            if X_train is not None and X_train.shape[0] > 0:
                print(f"    🎯 ML sequences created: Train {X_train.shape}, Test {X_test.shape}")
                if n_components_for_pca is not None:
                    print(f"    ✨ PCA resulted in {X_train.shape[2]} features.")
            else:
                print(f"    ⚠️ ML sequence creation did not return data, even if no exception was raised.")
                X_train = None
        except Exception as e:
            print(f"    ⚠️ ML sequence creation failed: {e}")
            traceback.print_exc()  # Good to have traceback here too
            X_train = None

        # Test Econometric Models
        print("\n📈 Testing Econometric Models...")
        from src.fts_toolkit.econometric_models import arima_model, garch_model, arima_garch_model

        returns_series = df['log_returns'].dropna()
        price_series = df['Close'].dropna()

        if len(price_series) > 50:  # ARIMA works on price series
            print("\n    🔍 ARIMA Model Testing:")
            try:
                arima_fitted = arima_model.fit(price_series)  # Uses internal auto_arima
                arima_diagnostics = arima_model.diagnostic_check()
                arima_forecast = arima_model.forecast(steps=5)
                print(f"    ✅ ARIMA{arima_model.order} fitted successfully")
                print(f"    📊 AIC: {arima_fitted.aic:.4f}")
                print(f"    🔮 5-step forecast: {arima_forecast['forecast'].iloc[-1]:.5f}")
            except Exception as e:
                print(f"    ❌ ARIMA fitting failed: {e}")
                traceback.print_exc()
        else:
            print("    ℹ️ Insufficient data for ARIMA model testing.")

        if len(returns_series) > 50:  # GARCH works on returns series
            print("\n    📊 GARCH Model Testing:")
            try:
                garch_fitted = garch_model.fit(returns_series, order=(1, 1))  # order can be configured
                garch_diagnostics = garch_model.diagnostic_check()
                garch_vol_forecast = garch_model.forecast_volatility(horizon=5)
                print(f"    ✅ GARCH(1,1) fitted successfully")
                print(f"    📊 Log-likelihood: {garch_fitted.loglikelihood:.4f}")
                print(f"    🔮 Volatility forecast: {garch_vol_forecast['volatility_forecast'][0]:.6f}")
            except Exception as e:
                print(f"    ❌ GARCH fitting failed: {e}")
                traceback.print_exc()

            print("\n    🤝 Hybrid ARIMA-GARCH Testing:")
            try:
                # Assuming arima_model was fitted above. Hybrid can use that instance or fit its own.
                # The current hybrid_arima_garch.fit() re-fits ARIMA.
                hybrid_fitted = arima_garch_model.fit(price_series)  # Pass price_series
                hybrid_forecast = arima_garch_model.forecast(steps=5)
                print(f"    ✅ Hybrid model fitted successfully")
                print(f"    📊 Model type: {hybrid_forecast['model_type']}")
                print(f"    🔮 Mean forecast: {hybrid_forecast['mean_forecast'].iloc[-1]:.5f}")
                if hybrid_forecast['volatility_forecast'] is not None:
                    print(f"    📈 Volatility forecast: {hybrid_forecast['volatility_forecast'][0]:.6f}")
            except Exception as e:
                print(f"    ❌ Hybrid modeling failed: {e}")
                traceback.print_exc()
        else:
            print("    ℹ️ Insufficient data for GARCH/Hybrid model testing.")

        # Enhanced ML Models (if sequences were created)
        if X_train is not None and y_train is not None and X_train.size > 0 and X_test.size > 0:
            print("\n🤖 Testing Enhanced ML Models...")
            from src.fts_toolkit.models import forecaster

            try:
                # Train models using hyperparameters from config
                linear_model = forecaster.train_linear_model(X_train, y_train, X_test, y_test)

                rf_model = forecaster.train_random_forest(
                    X_train, y_train, X_test, y_test,
                    n_estimators=config.RF_N_ESTIMATORS,
                    max_depth=config.RF_MAX_DEPTH
                )

                svr_model = forecaster.train_svr_model(
                    X_train, y_train, X_test, y_test,
                    kernel=config.SVR_KERNEL, C=config.SVR_C,
                    epsilon=config.SVR_EPSILON, gamma=config.SVR_GAMMA
                )

                xgboost_model = forecaster.train_xgboost_model(
                    X_train, y_train, X_test, y_test,
                    n_estimators=config.XGB_N_ESTIMATORS, learning_rate=config.XGB_LEARNING_RATE,
                    max_depth=config.XGB_MAX_DEPTH, early_stopping_rounds=config.XGB_EARLY_STOPPING_ROUNDS
                )

                lstm_model = forecaster.train_lstm_model(
                    X_train, y_train, X_test, y_test,
                    epochs=config.LSTM_TUNER_EPOCHS, batch_size=config.LSTM_BATCH_SIZE,
                    validation_split=config.LSTM_VALIDATION_SPLIT,
                    early_stopping_patience=config.LSTM_EARLY_STOPPING_PATIENCE,
                    tuner_max_trials=config.LSTM_TUNER_MAX_TRIALS,
                    tuner_executions_per_trial=config.LSTM_TUNER_EXECUTIONS_PER_TRIAL,
                    tuner_directory=config.LSTM_TUNER_DIR,
                    tuner_project_name=config.LSTM_TUNER_PROJECT_NAME
                )

                gru_model = forecaster.train_gru_model(
                    X_train, y_train, X_test, y_test,
                    epochs=config.GRU_TUNER_EPOCHS, batch_size=config.GRU_BATCH_SIZE,
                    validation_split=config.GRU_VALIDATION_SPLIT,
                    early_stopping_patience=config.GRU_EARLY_STOPPING_PATIENCE,
                    tuner_max_trials=config.GRU_TUNER_MAX_TRIALS,
                    tuner_executions_per_trial=config.GRU_TUNER_EXECUTIONS_PER_TRIAL,
                    tuner_directory=config.GRU_TUNER_DIR,
                    tuner_project_name=config.GRU_TUNER_PROJECT_NAME
                )

                # Model comparison
                print("\n📊 Enhanced Model Comparison:")
                comparison = forecaster.get_model_comparison()
                print(comparison[['test_mse', 'test_mae']].round(6))

                # --- GENERATE FLAT FEATURE NAMES ONCE (after X_train is confirmed valid) ---
                flat_feature_names = []
                num_features_per_step = X_train.shape[2]
                base_feature_names = []

                if n_components_for_pca is not None:
                    base_feature_names = [f"PC_{i}" for i in range(num_features_per_step)]
                    print(
                        f"    📊 Generating feature importance names for {num_features_per_step} Principal Components.")
                elif hasattr(advanced_processor, 'feature_columns') and advanced_processor.feature_columns:
                    base_feature_names = advanced_processor.feature_columns
                    print(
                        f"    📊 Generating feature importance names for {len(base_feature_names)} original engineered features.")
                else:
                    base_feature_names = [f"feature_{i}" for i in range(num_features_per_step)]
                    print(f"    📊 Using generic base feature names for importance.")

                window_steps = X_train.shape[1]
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
                        generic_rf_flat_names = [f"rf_flat_feature_{k}" for k in
                                                 range(len(rf_model.feature_importances_))]
                        feature_importance_rf_generic = pd.DataFrame({
                            'feature': generic_rf_flat_names, 'importance': rf_model.feature_importances_
                        }).sort_values('importance', ascending=False).head(10)
                        print("\n🎯 Top 10 Most Important Features for Random Forest (Generic Flattened, Debug):")
                        for _, row_rf_gen in feature_importance_rf_generic.iterrows():
                            print(f"    {row_rf_gen['feature']}: {row_rf_gen['importance']:.4f}")

                # Feature importance for XGBoost
                if hasattr(xgboost_model, 'feature_importances_'):
                    if len(flat_feature_names) == len(xgboost_model.feature_importances_):
                        feature_importance_xgb = pd.DataFrame({
                            'feature': flat_feature_names,
                            'importance': xgboost_model.feature_importances_
                        }).sort_values('importance', ascending=False).head(10)
                        print("\n🎯 Top 10 Most Important Features for XGBoost (Flattened):")
                        for _, row_xgb in feature_importance_xgb.iterrows():
                            print(f"    {row_xgb['feature']}: {row_xgb['importance']:.4f}")
                    else:
                        print(
                            f"    ⚠️ Could not display XGBoost feature importance: Mismatch in names length ({len(flat_feature_names)}) vs importances length ({len(xgboost_model.feature_importances_)}). Using generic.")
                        generic_xgb_flat_names = [f"xgb_flat_feature_{k}" for k in
                                                  range(len(xgboost_model.feature_importances_))]
                        feature_importance_xgb_generic = pd.DataFrame({
                            'feature': generic_xgb_flat_names, 'importance': xgboost_model.feature_importances_
                        }).sort_values('importance', ascending=False).head(10)
                        print("\n🎯 Top 10 Most Important Features for XGBoost (Generic Flattened, Debug):")
                        for _, row_xgb_gen in feature_importance_xgb_generic.iterrows():
                            print(f"    {row_xgb_gen['feature']}: {row_xgb_gen['importance']:.4f}")

            except Exception as e:
                print(f"    ❌ Enhanced ML modeling failed: {e}")
                traceback.print_exc()
        else:
            print("\n🤖 Skipping Enhanced ML Models training as no valid sequence data is available.")

        # --- 📈 Testing Multivariate Models (VAR) ---
        from src.fts_toolkit.multivariate_models import VARAnalyzer
        print("\n📈 Testing Multivariate Models (VAR)...")
        var_symbols_str = config.VAR_SYMBOLS_STR
        var_symbols_list = [s.strip() for s in var_symbols_str.split(',') if s.strip()]

        if not var_symbols_list or len(var_symbols_list) < 2:
            print("   ⚠️ Insufficient symbols defined for VAR analysis in config.VAR_SYMBOLS_STR. Skipping VAR.")
        else:
            print(f"   Fetching data for VAR symbols: {var_symbols_list}")
            all_series_data_var = {}  # Use a different name to avoid conflict with 'df' if it's used above
            for symbol_var in var_symbols_list:
                df_symbol_var = scraper.get_fx_data_yahoo(symbol=symbol_var, days=config.LOOKBACK_DAYS)
                if df_symbol_var is not None and not df_symbol_var.empty:
                    df_symbol_var['Date'] = pd.to_datetime(df_symbol_var['Date'])
                    df_symbol_var.set_index('Date', inplace=True)
                    all_series_data_var[symbol_var] = df_symbol_var['Close']
                    print(f"    ✅ Got {len(df_symbol_var)} data points for {symbol_var}")
                else:
                    print(f"    ⚠️ Could not fetch data for {symbol_var} for VAR.")

            if len(all_series_data_var) >= 2:  # Need at least 2 series
                var_input_df_raw = pd.concat(all_series_data_var.values(), axis=1, keys=all_series_data_var.keys())
                var_input_df_raw.sort_index(inplace=True)
                var_input_df_raw.dropna(how='all', inplace=True)  # Drop rows where ALL values are NaN

                if len(var_input_df_raw) < config.VAR_MAX_LAGS + 10:  # Heuristic for enough data
                    print(
                        f"    ⚠️ Insufficient aligned data for VAR ({len(var_input_df_raw)} rows after initial NaN handling). Skipping VAR.")
                else:
                    print(f"    📊 Combined VAR input data shape (raw prices): {var_input_df_raw.shape}")

                    # Instantiate VARAnalyzer with your existing advanced_processor
                    var_analyzer = VARAnalyzer(advanced_processor_instance=advanced_processor)

                    # Prepare data (make stationary)
                    stationary_var_data = var_analyzer.prepare_data_for_var(
                        var_input_df_raw.copy())  # Pass a copy

                    if stationary_var_data is not None and not stationary_var_data.empty and stationary_var_data.shape[
                        1] >= 2:
                        # Fit VAR model
                        var_fitted_model = var_analyzer.fit(ic=config.AIC_BIC_FOR_VAR_LAG.lower() if hasattr(config,
                                                                                                             'AIC_BIC_FOR_VAR_LAG') else 'aic')  # Example: make IC configurable

                        if var_fitted_model:
                            # Forecast (already there)
                            var_forecast = var_analyzer.forecast(steps=config.VAR_FORECAST_STEPS)
                            print(f"\n    🔮 VAR Model Forecast ({config.VAR_FORECAST_STEPS}-step ahead):")
                            print(var_forecast)

                            # --- NEW: Impulse Response Functions (IRF) ---
                            print("\n    📊 Analyzing Impulse Response Functions...")
                            # Call get_impulse_response without orth. Orthogonalization is a plotting choice.
                            irf = var_analyzer.get_impulse_response(steps=config.VAR_IRF_STEPS)
                            if irf:
                                print(f"    IRF object calculated for {config.VAR_IRF_STEPS} steps.")
                                try:
                                    # When plotting, you specify orth=True for Cholesky orthogonalized IRFs
                                    fig_irf = irf.plot(orth=True, signif=0.05)
                                    if fig_irf:
                                        figures = [plt.figure(i) for i in plt.get_fignums() if
                                                   plt.figure(i) is fig_irf or (hasattr(fig_irf, 'fig') and plt.figure(
                                                       i) is fig_irf.fig)]
                                        # If irf.plot returns a single figure object directly
                                        if not figures and hasattr(fig_irf, 'tight_layout'):
                                            figures = [fig_irf]

                                        if not figures:  # If still no figures, it might be a multi-figure plot
                                            figures = [plt.figure(n) for n in
                                                       plt.get_fignums()]  # Get all current figures
                                            # This might get more than just the IRF plot if other plots were open.
                                            # A cleaner way would be to create a new figure before irf.plot if statsmodels doesn't return it well.

                                        plot_found_and_saved = False
                                        for i, fig in enumerate(figures):
                                            # Heuristic to check if it's likely the IRF plot (it might have multiple axes)
                                            if len(fig.axes) >= stationary_var_data.shape[1]:  # Number of variables
                                                fig.suptitle(f"Impulse Response Functions (VAR) - Plot {i + 1}",
                                                             fontsize=14)
                                                fig.savefig(f"var_irf_plot_{symbol_var}_{i + 1}.png")
                                                print(f"    Saved IRF plot var_irf_plot_{symbol_var}_{i + 1}.png")
                                                plot_found_and_saved = True

                                        if plot_found_and_saved:
                                            plt.close(
                                                'all')  # Close all figures after saving to prevent display in non-GUI
                                        else:
                                            print(
                                                "    IRF plot was generated by statsmodels but not identified for saving/titling.")

                                    print("    IRF plots generated (attempted to save as PNG).")
                                except Exception as e_plot_irf:
                                    print(f"    Could not plot/save IRF: {e_plot_irf}")
                                    traceback.print_exc()
                                    print(
                                        "    IRF object is available but plotting/saving failed. You can access irf.irfs, irf.cum_effects etc. for manual analysis.")
                            else:
                                print("    ❌ IRF calculation failed.")

                            # --- NEW: Forecast Error Variance Decomposition (FEVD) ---
                            print("\n    📊 Analyzing Forecast Error Variance Decomposition...")
                            fevd = var_analyzer.get_fevd(steps=config.VAR_IRF_STEPS)
                            if fevd:
                                print(f"    FEVD calculated for {config.VAR_IRF_STEPS} steps. Summary:")
                                try:
                                    print(fevd.summary())
                                    # You can also plot FEVD:
                                    # fig_fevd = fevd.plot()
                                    # if fig_fevd:
                                    #    fig_fevd.suptitle("Forecast Error Variance Decomposition (VAR)", fontsize=14)
                                    #    # fig_fevd.savefig("var_fevd_plot.png")
                                    #    # print("    Saved FEVD plot var_fevd_plot.png")
                                    # print("    (FEVD plotting can be added similarly to IRF)")
                                except Exception as e_fevd_summary:
                                    print(f"    Could not print FEVD summary: {e_fevd_summary}")
                            else:
                                print("    ❌ VAR model fitting failed.")
                        else:
                            print("    ❌ VAR data preparation resulted in no usable data.")
                    else:
                        print("    ❌ Not enough series data successfully fetched/aligned for VAR analysis.")

        # Advanced Performance Metrics
        print("\n📊 Advanced Performance Analysis...")

        # Market regime analysis
        print("\n    📈 Market Regime Analysis:")
        # Ensure log_returns exists and has enough non-NaN values for rolling operations
        if 'log_returns' in df and df['log_returns'].count() >= 60:
            df['volatility_regime'] = np.where(
                df['log_returns'].rolling(20).std() > df['log_returns'].rolling(60).std().mean(),
                'High Volatility', 'Low Volatility'
            )
            regime_counts = df['volatility_regime'].value_counts()
            print(f"    High Volatility periods: {regime_counts.get('High Volatility', 0)} days")
            print(f"    Low Volatility periods: {regime_counts.get('Low Volatility', 0)} days")
        else:
            print("    ℹ️ Insufficient data or 'log_returns' missing for Market Regime Analysis.")

        # Stylized facts analysis
        print("\n    📊 Stylized Facts Analysis:")
        returns_clean = returns_series.dropna()  # returns_series was df['log_returns'].dropna()

        if len(returns_clean) > 30:
            acf_returns = np.abs(returns_clean).autocorr(lag=1)
            acf_squared_returns = (returns_clean ** 2).autocorr(lag=1)
            print(f"    Volatility clustering (|returns| ACF): {acf_returns:.4f}")
            print(f"    Volatility persistence (returns² ACF): {acf_squared_returns:.4f}")

            kurt = kurtosis(returns_clean, fisher=True)
            skewness = skew(returns_clean)
            print(f"    Excess kurtosis (fat tails): {kurt:.4f}")
            print(f"    Skewness: {skewness:.4f}")

            jb_stat, jb_pvalue = jarque_bera(returns_clean)
            print(f"    Jarque-Bera test p-value: {jb_pvalue:.4f} ({'Non-normal' if jb_pvalue < 0.05 else 'Normal'})")
        else:
            print("    ℹ️ Insufficient data for Stylized Facts Analysis.")

        # Walk-forward validation simulation
        print("\n🔄 Walk-Forward Validation Simulation...")
        if len(df_processed) > 120:
            try:
                # Ensure simulate_walk_forward_validation is defined in the global scope or imported
                results_wfv = simulate_walk_forward_validation(df_processed, window_size=60, test_size=10)
                if results_wfv['rmse_scores']:
                    print(f"    ✅ Walk-forward validation completed")
                    print(f"    📊 Average RMSE: {np.mean(results_wfv['rmse_scores']):.6f}")
                    print(f"    📊 RMSE std: {np.std(results_wfv['rmse_scores']):.6f}")
                    print(f"    🔄 Number of validation windows: {len(results_wfv['rmse_scores'])}")
                else:
                    print(f"    ⚠️ Walk-forward validation did not produce results. Check data and parameters.")
            except Exception as e:
                print(f"    ❌ Walk-forward validation failed: {e}")
                traceback.print_exc()
        else:
            print(f"    ℹ️ Insufficient data for walk-forward validation (need > 120, got {len(df_processed)})")

        # Risk metrics
        print("\n⚠️ Risk Analysis...")
        if len(returns_clean) > 30:
            var_95 = np.percentile(returns_clean, 5)
            var_99 = np.percentile(returns_clean, 1)
            print(f"    VaR (95%): {var_95:.4f} ({var_95 * 100:.2f}%)")
            print(f"    VaR (99%): {var_99:.4f} ({var_99 * 100:.2f}%)")

            cumulative_returns = (1 + returns_clean).cumprod()
            rolling_max = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - rolling_max) / rolling_max
            max_drawdown = drawdown.min()
            print(f"    Maximum Drawdown: {max_drawdown:.4f} ({max_drawdown * 100:.2f}%)")
        else:
            print("    ℹ️ Insufficient data for Risk Analysis.")

        # Model interpretation and insights
        print("\n🔍 Model Insights and Recommendations...")
        print("    📝 Stationarity Analysis:")
        print(f"    • Price series: {close_stationarity.get('recommendation', 'N/A')}")  # Use .get for safety
        print(f"    • Return series: {returns_stationarity.get('recommendation', 'N/A')}")  # Use .get for safety

        print("\n    📝 Model Recommendations:")
        if close_stationarity.get('recommendation') != 'stationary':
            print("    • Use differenced prices or returns for ARIMA modeling")
        if acf_squared_returns > 0.1:
            print("    • Strong volatility clustering detected → GARCH modeling recommended")
        if kurt > 1:  # Standard kurtosis of normal dist is 3, excess is 0. Fat tails usually > 0 for excess.
            print("    • Fat tails detected → Consider t-distribution for GARCH or models robust to outliers.")

        print("\n    📝 Practical Considerations:")
        print("    • Short-term forecasts (1-5 days): ARIMA-GARCH hybrid recommended")
        print("    • Medium-term forecasts (1-4 weeks): ML models with regime detection")
        print("    • Long-term forecasts (>1 month): Fundamental analysis required")

        print("\n✅ Advanced Financial Time Series Analysis Completed!")
        # ... (Next Steps printout) ...

    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("    Ensure all required packages from requirements.txt are installed in your environment.")
        print(
            "    Relevant packages include: statsmodels, arch, scipy, pandas, numpy, scikit-learn, tensorflow, keras-tuner, xgboost")
    except Exception as e:
        print(f"❌ An unexpected error occurred in main: {e}")
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
