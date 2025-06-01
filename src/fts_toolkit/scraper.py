"""
FX Data Scraper for Financial Time Series Toolkit
Scrapes historical FX data from free sources
"""

import requests
import pandas as pd
import time
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
from .config import config
import logging

# Set up logging
logging.basicConfig(level=getattr(logging, config.LOG_LEVEL))
logger = logging.getLogger(__name__)

class FXScraper:
    """Web scraper for FX historical data"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    def get_fx_data_yahoo(self, symbol='EURUSD=X', days=365):
        """
        Scrape FX data from Yahoo Finance
        
        Args:
            symbol: FX pair symbol (e.g., 'EURUSD=X')
            days: Number of days to look back
            
        Returns:
            pandas.DataFrame with OHLCV data
        """
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            # Yahoo Finance URL format
            url = f"https://finance.yahoo.com/quote/{symbol}/history"
            params = {
                'period1': int(start_date.timestamp()),
                'period2': int(end_date.timestamp()),
                'interval': '1d',
                'filter': 'history',
                'frequency': '1d'
            }
            
            logger.info(f"Scraping {symbol} data for {days} days...")
            response = self.session.get(url, params=params)
            response.raise_for_status()
            
            # Parse with BeautifulSoup
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find the data table
            table = soup.find('table', {'data-test': 'historical-prices'})
            if not table:
                logger.warning("Could not find data table on Yahoo Finance")
                return self._get_mock_data(symbol, days)
            
            # Extract data from table
            rows = table.find('tbody').find_all('tr')
            data = []
            
            for row in rows:
                cols = row.find_all('td')
                if len(cols) >= 6:
                    try:
                        date_str = cols[0].text.strip()
                        open_price = float(cols[1].text.replace(',', ''))
                        high_price = float(cols[2].text.replace(',', ''))
                        low_price = float(cols[3].text.replace(',', ''))
                        close_price = float(cols[4].text.replace(',', ''))
                        volume = cols[5].text.replace(',', '')
                        
                        data.append({
                            'Date': pd.to_datetime(date_str),
                            'Open': open_price,
                            'High': high_price,
                            'Low': low_price,
                            'Close': close_price,
                            'Volume': volume
                        })
                    except (ValueError, AttributeError) as e:
                        logger.debug(f"Skipping row due to parsing error: {e}")
                        continue
            
            if not data:
                logger.warning("No data extracted, using mock data")
                return self._get_mock_data(symbol, days)
            
            df = pd.DataFrame(data)
            df = df.sort_values('Date').reset_index(drop=True)
            
            logger.info(f"Successfully scraped {len(df)} data points for {symbol}")
            return df
            
        except requests.RequestException as e:
            logger.error(f"Network error scraping {symbol}: {e}")
            return self._get_mock_data(symbol, days)
        except Exception as e:
            logger.error(f"Unexpected error scraping {symbol}: {e}")
            return self._get_mock_data(symbol, days)
    
    def _get_mock_data(self, symbol='EURUSD=X', days=365):
        """
        Generate mock FX data for testing when scraping fails
        
        Args:
            symbol: FX pair symbol
            days: Number of days to generate
            
        Returns:
            pandas.DataFrame with mock OHLCV data
        """
        import numpy as np
        
        logger.info(f"Generating mock data for {symbol}")
        
        # Set random seed for reproducible data
        np.random.seed(config.RANDOM_SEED)
        
        # Create date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # Generate realistic FX price movement
        base_price = 1.1000  # Starting EUR/USD rate
        price_changes = np.random.normal(0, 0.002, len(dates))  # 0.2% daily volatility
        prices = base_price * np.exp(np.cumsum(price_changes))
        
        data = []
        for i, date in enumerate(dates):
            if i == 0:
                open_price = base_price
            else:
                open_price = data[i-1]['Close']
            
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
    
    def save_data(self, df, symbol, filename=None):
        """
        Save scraped data to CSV file
        
        Args:
            df: DataFrame to save
            symbol: FX pair symbol
            filename: Optional custom filename
        """
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{symbol.replace('=X', '')}_{timestamp}.csv"
        
        filepath = config.RAW_DATA_DIR / filename
        df.to_csv(filepath, index=False)
        logger.info(f"Data saved to {filepath}")
        return filepath

# Create global scraper instance
scraper = FXScraper()
