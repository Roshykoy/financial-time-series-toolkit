"""
Econometric Models: ARIMA, GARCH, and Hybrid Models
Implementation of classical financial econometric techniques
"""

import pandas as pd
import numpy as np
import warnings
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from arch import arch_model
from .config import config
import logging

logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

class ARIMAModel:
    """ARIMA model implementation with automatic order selection"""
    
    def __init__(self):
        self.model = None
        self.fitted_model = None
        self.order = None
        self.aic_scores = {}
        
    def auto_arima(self, series, max_p=5, max_d=2, max_q=5, information_criterion='aic'):
        """
        Automatic ARIMA order selection using information criteria
        """
        best_aic = np.inf
        best_order = None
        
        logger.info("Starting automatic ARIMA order selection...")
        
        for p in range(max_p + 1):
            for d in range(max_d + 1):
                for q in range(max_q + 1):
                    try:
                        model = ARIMA(series, order=(p, d, q))
                        fitted = model.fit()
                        
                        if information_criterion == 'aic':
                            criterion_value = fitted.aic
                        elif information_criterion == 'bic':
                            criterion_value = fitted.bic
                        else:
                            criterion_value = fitted.aic
                            
                        self.aic_scores[(p, d, q)] = criterion_value
                        
                        if criterion_value < best_aic:
                            best_aic = criterion_value
                            best_order = (p, d, q)
                            
                    except Exception as e:
                        logger.debug(f"Failed to fit ARIMA({p},{d},{q}): {e}")
                        continue
        
        self.order = best_order
        logger.info(f"Selected ARIMA{best_order} with {information_criterion.upper()}={best_aic:.4f}")
        return best_order
    
    def fit(self, series, order=None):
        """Fit ARIMA model"""
        if order is None:
            order = self.auto_arima(series)
        else:
            self.order = order
            
        self.model = ARIMA(series, order=order)
        self.fitted_model = self.model.fit()
        
        logger.info(f"ARIMA{order} fitted successfully")
        logger.info(f"AIC: {self.fitted_model.aic:.4f}, BIC: {self.fitted_model.bic:.4f}")
        
        return self.fitted_model
    
    def diagnostic_check(self):
        """Comprehensive diagnostic checking"""
        if self.fitted_model is None:
            raise ValueError("Model must be fitted first")
            
        residuals = self.fitted_model.resid
        
        # Ljung-Box test for residual autocorrelation
        lb_test = acorr_ljungbox(residuals, lags=10, return_df=True)
        
        # Normality test (Jarque-Bera)
        from scipy.stats import jarque_bera
        jb_stat, jb_pvalue = jarque_bera(residuals.dropna())
        
        diagnostics = {
            'ljung_box': lb_test,
            'jarque_bera': {'statistic': jb_stat, 'p_value': jb_pvalue},
            'residual_mean': residuals.mean(),
            'residual_std': residuals.std()
        }
        
        logger.info("ARIMA Diagnostic Results:")
        logger.info(f"  Ljung-Box p-value (lag 10): {lb_test['lb_pvalue'].iloc[-1]:.4f}")
        logger.info(f"  Jarque-Bera p-value: {jb_pvalue:.4f}")
        logger.info(f"  Residual mean: {residuals.mean():.6f}")
        
        return diagnostics
    
    def forecast(self, steps=1):
        """Generate forecasts"""
        if self.fitted_model is None:
            raise ValueError("Model must be fitted first")
            
        forecast = self.fitted_model.forecast(steps=steps)
        conf_int = self.fitted_model.get_forecast(steps=steps).conf_int()
        
        return {
            'forecast': forecast,
            'conf_int': conf_int,
            'model_order': self.order
        }

class GARCHModel:
    """GARCH model for volatility forecasting"""
    
    def __init__(self):
        self.model = None
        self.fitted_model = None
        self.order = None
        
    def fit(self, returns, order=(1, 1), mean_model='Constant', distribution='normal'):
        """
        Fit GARCH model to returns series
        
        Args:
            returns: Return series
            order: (p, q) for GARCH(p,q)
            mean_model: 'Constant', 'Zero', 'ARX'
            distribution: 'normal', 't', 'skewt'
        """
        self.order = order
        
        # Remove NaN values
        clean_returns = returns.dropna()
        
        # Create GARCH model
        self.model = arch_model(
            clean_returns, 
            mean=mean_model, 
            vol='GARCH', 
            p=order[0], 
            q=order[1],
            dist=distribution
        )
        
        # Fit model
        self.fitted_model = self.model.fit(disp='off')
        
        logger.info(f"GARCH({order[0]},{order[1]}) fitted successfully")
        logger.info(f"Log-likelihood: {self.fitted_model.loglikelihood:.4f}")
        
        return self.fitted_model
    
    def diagnostic_check(self):
        """GARCH diagnostic tests"""
        if self.fitted_model is None:
            raise ValueError("Model must be fitted first")
            
        # Standardized residuals
        std_resid = self.fitted_model.std_resid
        
        # Ljung-Box test on standardized residuals
        lb_test_resid = acorr_ljungbox(std_resid.dropna(), lags=10, return_df=True)
        
        # Ljung-Box test on squared standardized residuals
        lb_test_resid_sq = acorr_ljungbox(std_resid.dropna()**2, lags=10, return_df=True)
        
        diagnostics = {
            'ljung_box_resid': lb_test_resid,
            'ljung_box_resid_squared': lb_test_resid_sq,
            'std_resid_mean': std_resid.mean(),
            'std_resid_std': std_resid.std()
        }
        
        logger.info("GARCH Diagnostic Results:")
        logger.info(f"  LB test (std. resid) p-value: {lb_test_resid['lb_pvalue'].iloc[-1]:.4f}")
        logger.info(f"  LB test (std. resid²) p-value: {lb_test_resid_sq['lb_pvalue'].iloc[-1]:.4f}")
        
        return diagnostics
    
    def forecast_volatility(self, horizon=1):
        """Forecast conditional volatility"""
        if self.fitted_model is None:
            raise ValueError("Model must be fitted first")
            
        vol_forecast = self.fitted_model.forecast(horizon=horizon, method='simulation')
        
        return {
            'volatility_forecast': vol_forecast.variance.iloc[-1].values,
            'residual_variance': vol_forecast.residual_variance.iloc[-1].values
        }

class ARIMAGARCHHybrid:
    """Hybrid ARIMA-GARCH model"""
    
    def __init__(self):
        self.arima_model = ARIMAModel()
        self.garch_model = GARCHModel()
        self.fitted = False
        
    def fit(self, series, arima_order=None, garch_order=(1, 1)):
        """
        Fit hybrid ARIMA-GARCH model
        
        Process:
        1. Fit ARIMA to the series
        2. Test ARIMA residuals for ARCH effects
        3. Fit GARCH to ARIMA residuals if ARCH effects present
        """
        logger.info("Fitting ARIMA-GARCH hybrid model...")
        
        # Step 1: Fit ARIMA
        self.arima_fitted = self.arima_model.fit(series, order=arima_order)
        arima_residuals = self.arima_fitted.resid
        
        # Step 2: Test for ARCH effects in ARIMA residuals
        from arch.unitroot import DFGLS
        arch_test = self._test_arch_effects(arima_residuals)
        
        if arch_test['has_arch_effects']:
            logger.info("ARCH effects detected in ARIMA residuals. Fitting GARCH...")
            # Step 3: Fit GARCH to residuals
            self.garch_fitted = self.garch_model.fit(arima_residuals, order=garch_order)
            self.fitted = True
        else:
            logger.info("No significant ARCH effects detected. ARIMA model sufficient.")
            self.garch_fitted = None
            self.fitted = True
            
        return self
    
    def _test_arch_effects(self, residuals, lags=5):
        """Test for ARCH effects using Ljung-Box test on squared residuals"""
        from statsmodels.stats.diagnostic import het_arch
        
        # ARCH-LM test
        try:
            lm_stat, lm_pvalue, f_stat, f_pvalue = het_arch(residuals.dropna(), nlags=lags)
            has_arch = lm_pvalue < 0.05
            
            logger.info(f"ARCH-LM test p-value: {lm_pvalue:.4f}")
            
            return {
                'has_arch_effects': has_arch,
                'lm_statistic': lm_stat,
                'lm_pvalue': lm_pvalue,
                'f_statistic': f_stat,
                'f_pvalue': f_pvalue
            }
        except Exception as e:
            logger.warning(f"ARCH test failed: {e}")
            return {'has_arch_effects': False, 'error': str(e)}
    
    def forecast(self, steps=1):
        """Generate hybrid forecasts"""
        if not self.fitted:
            raise ValueError("Model must be fitted first")
            
        # ARIMA forecast for mean
        arima_forecast = self.arima_model.forecast(steps=steps)
        
        # GARCH forecast for volatility (if fitted)
        if self.garch_fitted is not None:
            garch_forecast = self.garch_model.forecast_volatility(horizon=steps)
            
            return {
                'mean_forecast': arima_forecast['forecast'],
                'volatility_forecast': garch_forecast['volatility_forecast'],
                'conf_int': arima_forecast['conf_int'],
                'model_type': 'ARIMA-GARCH'
            }
        else:
            return {
                'mean_forecast': arima_forecast['forecast'],
                'volatility_forecast': None,
                'conf_int': arima_forecast['conf_int'],
                'model_type': 'ARIMA-only'
            }

# Create global instances
arima_model = ARIMAModel()
garch_model = GARCHModel()
arima_garch_model = ARIMAGARCHHybrid()
