"""
FX Data Scraper for Financial Time Series Toolkit
Scrapes historical FX data from free sources
"""

import requests
import pandas as pd
import time
from datetime import datetime, timedelta
from .config import config
import logging

# Set up logging
logging.basicConfig(level=getattr(logging, config.LOG_LEVEL))
logger = logging.getLogger(__name__)


class FXScraper:
    """Web scraper for FX historical data"""

    def __init__(self):
        self.session = requests.Session()
        # It's good to have a user-agent, Yahoo might block default ones or be suspicious
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) '
                          'Chrome/91.0.4472.124 Safari/537.36'
        })

    def get_fx_data_yahoo(self, symbol='EURUSD=X', days=365):
        """
        Scrape FX data from Yahoo Finance using the V8 Chart API (JSON).

        Args:
            symbol: FX pair symbol (e.g., 'EURUSD=X')
            days: Number of days to look back

        Returns:
            pandas.DataFrame with OHLCV data
        """
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)

            # Convert dates to Unix timestamps
            period1 = int(start_date.timestamp())
            period2 = int(end_date.timestamp())
            interval = '1d'  # For daily data, as per your original scraper

            # Use the V8 chart API endpoint
            url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
            params = {
                'period1': period1,
                'period2': period2,
                'interval': interval,
                'events': 'history'  # Often includes splits/dividends, may not be critical for FX
                # 'includePrePost': 'false' # Optional: to exclude pre/post market data if not needed
            }

            logger.info(f"Scraping {symbol} data for {days} days using V8 API...")
            response = self.session.get(url, params=params)
            response.raise_for_status()  # Will raise an HTTPError if the HTTP request returned an unsuccessful status code

            json_data = response.json()

            # --- Navigate the JSON structure ---
            # Based on typical Yahoo Finance V8 API response
            if not json_data.get('chart') or not json_data['chart'].get('result'):
                logger.warning(f"Yahoo V8 API: 'chart' or 'chart.result' not found in JSON response for {symbol}.")
                return self._get_mock_data(symbol, days)

            result = json_data['chart']['result'][0]  # Assuming first result is the one we want

            if not result.get('timestamp') or not result.get('indicators') or \
                    not result['indicators'].get('quote') or not result['indicators']['quote']:
                logger.warning(
                    f"Yahoo V8 API: Essential data (timestamp/indicators/quote) missing in JSON for {symbol}.")
                return self._get_mock_data(symbol, days)

            timestamps = result['timestamp']
            ohlcv_data = result['indicators']['quote'][0]  # Assuming quote data is in the first element

            if not all(k in ohlcv_data for k in ['open', 'high', 'low', 'close', 'volume']):
                logger.warning(f"Yahoo V8 API: OHLCV keys missing in indicators.quote for {symbol}.")
                return self._get_mock_data(symbol, days)

            # Create DataFrame
            df_data = {
                'Date': pd.to_datetime(timestamps, unit='s').normalize(),
                # Normalize to remove time part, keep only date
                'Open': ohlcv_data['open'],
                'High': ohlcv_data['high'],
                'Low': ohlcv_data['low'],
                'Close': ohlcv_data['close'],
                'Volume': ohlcv_data['volume']
            }

            df = pd.DataFrame(df_data)

            # Data cleaning: Yahoo can sometimes return rows with all NaNs for prices if there was no trading.
            # Also, filter out any days where 'Close' might be None or NaN before sorting,
            # as this can happen for non-trading days that still get a timestamp.
            df.dropna(subset=['Open', 'High', 'Low', 'Close'], inplace=True)

            if df.empty:
                logger.warning(
                    f"No valid data rows extracted from Yahoo V8 API for {symbol} after NaN drop. Using mock data.")
                return self._get_mock_data(symbol, days)

            # Ensure correct types for OHLCV (Yahoo sometimes returns None for missing values)
            for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')

            df.dropna(subset=['Open', 'High', 'Low', 'Close'], inplace=True)  # Drop again if coerce created NaNs

            df = df.sort_values('Date').reset_index(drop=True)

            # The 'Adj Close' is sometimes in result['indicators']['adjclose'][0]['adjclose']
            # You can add it if needed, similar to how OHLCV is extracted.
            # For FX, Close and Adj Close are often the same.

            logger.info(f"Successfully scraped {len(df)} data points for {symbol} using V8 API")
            return df

        except requests.RequestException as e:
            logger.error(f"Network error scraping {symbol} with V8 API: {e}")
            return self._get_mock_data(symbol, days)
        except Exception as e:
            logger.error(f"Unexpected error scraping {symbol} with V8 API: {e}")
            # To help debug JSON structure issues if they arise:
            # if 'json_data' in locals():
            #     logger.error(f"Problematic JSON data (first 500 chars): {str(json_data)[:500]}")
            # else:
            #     logger.error(f"Response content (if not JSON parsable): {response.text[:500] if 'response' in locals() else 'No response object'}")
            return self._get_mock_data(symbol, days)

    def _get_mock_data(self, symbol='EURUSD=X', days=365):  # Your existing mock data function
        # ... (keep your existing mock data implementation)
        import numpy as np  # Ensure numpy is imported if used here
        logger.info(f"Generating mock data for {symbol}")
        np.random.seed(config.RANDOM_SEED)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        base_price = 1.1000
        price_changes = np.random.normal(0, 0.002, len(dates))
        prices = base_price * np.exp(np.cumsum(price_changes))
        data = []
        for i, date in enumerate(dates):
            if i == 0:
                open_price = base_price
            else:
                open_price = data[i - 1]['Close']
            close_price = prices[i]
            high_price = max(open_price, close_price) * (1 + abs(np.random.normal(0, 0.001)))
            low_price = min(open_price, close_price) * (1 - abs(np.random.normal(0, 0.001)))
            data.append({
                'Date': date,
                'Open': round(open_price, 5),
                'High': round(high_price, 5),
                'Low': round(low_price, 5),
                'Close': round(close_price, 5),
                'Volume': f"{np.random.randint(1000000, 10000000):,}"
            })
        df = pd.DataFrame(data)
        logger.info(f"Generated {len(df)} mock data points for {symbol}")
        return df

    def save_data(self, df, symbol, filename=None):  # Your existing save_data function
        # ... (keep your existing save_data implementation)
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{symbol.replace('=X', '')}_{timestamp}.csv"
        filepath = config.RAW_DATA_DIR / filename
        df.to_csv(filepath, index=False)
        logger.info(f"Data saved to {filepath}")
        return filepath


# Create global scraper instance
scraper = FXScraper()
