import os
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier

# Alpaca-py imports
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, OrderType, TimeInForce

# ----------------- CONFIGURATION -----------------
# The S&P 100 (100 largest, most stable US companies)
TICKERS = [
    "AAPL", "MSFT", "AMZN", "NVDA", "GOOGL", "GOOG", "META", "BRK.B", "TSLA", "UNH",
    "JLI", "JPM", "XOM", "V", "PG", "MA", "AVGO", "HD", "HD", "CVX",
    "MRK", "ABBV", "ADBE", "COST", "PEP", "KO", "TMO", "MCD", "WMT", "BAC",
    "CSCO", "CRM", "ACN", "LLY", "ABT", "ORCL", "VZ", "INTC", "TXN", "QCOM",
    "CMCSA", "AMGN", "NFLX", "AMD", "DIS", "PM", "NKE", "COP", "HON", "T",
    "IBM", "JCI", "GE", "UNP", "LOW", "AXP", "INTU", "SPGI", "SBUX", "EL",
    "PLD", "AMAT", "MDLZ", "CAT", "GILD", "RTX", "LMT", "BKNG", "TJX", "ADI",
    "C", "ISRG", "SYK", "REGN", "MDG", "VRTX", "ADP", "MMC", "CI", "DHR",
    "MU", "LRCX", "SLB", "GEHC", "NOW", "PANW", "SNPS", "MELI", "CDNS", "EQIX",
    "SO", "D", "KDP", "CL", "WM", "NOC", "WM", "BSX", "GPN", "HCA"
]

# Clean any potential duplicate tickers or formatting issues
TICKERS = list(set([t.replace(".", "-") for t in TICKERS]))[:100]

# API Credentials pulled securely from GitHub Secrets
API_KEY = os.environ.get("ALPACA_API_KEY")
SECRET_KEY = os.environ.get("ALPACA_SECRET_KEY")

class MLMultiTrader:
    def __init__(self):
        # paper=True guarantees no real money is used
        self.trading_client = TradingClient(API_KEY, SECRET_KEY, paper=True)
        
    def get_historical_data(self, ticker):
        """Downloads historical data for a specific ticker."""
        # Grab past 60 days of hourly data for feature modeling
        df = yf.download(ticker, period="60d", interval="1h", progress=False)
        if df.empty or len(df) < 20:
            return None
        return df

    def create_features(self, df):
        """Calculates features for the Machine Learning model."""
        df = df.copy()
        
        # Calculate standard 15-hour Simple Moving Average (SMA)
        df['SMA_15'] = df['Close'].rolling(window=15).mean()
        
        # Calculate Hourly Returns
        df['Returns'] = df['Close'].pct_change()
        
        # Target: 1 if next hour's price is UP, 0 if DOWN
        df['Target'] = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)
        
        # Drop rows with missing rolling values
        df.dropna(inplace=True)
        return df

    def train_and_predict(self, df):
        """Trains a Random Forest classifier and predicts the next move."""
        # Select Features and Target
        features = ['SMA_15', 'Returns']
        X = df[features].values
        y = df['Target'].values
        
        # Train model on all data except the very last bar (which has no target)
        model = RandomForestClassifier(n_estimators=50, random_state=42)
        model.fit(X[:-1], y[:-1])
        
        # Predict on the current hour (last row)
        last_row = X[-1].reshape(1, -1)
        prediction = model.predict(last_row)[0]
        
        return "UP" if prediction == 1 else "DOWN"

    def get_portfolio_positions(self):
        """Returns a list of ticker symbols we currently own."""
        positions = self.trading_client.get_all_positions()
        return [pos.symbol for pos in positions]

    def execute_trade(self, ticker, prediction, owned_tickers):
        """Places a buy or sell order depending on ML prediction and holdings."""
        is_owned = ticker in owned_tickers
        
        try:
            if prediction == "UP" and not is_owned:
                # Submit Buy Order for 1 share
                buy_order = MarketOrderRequest(
                    symbol=ticker,
                    qty=1,
                    side=OrderSide.BUY,
                    type=OrderType.MARKET,
                    time_in_force=TimeInForce.DAY
                )
                self.trading_client.submit_order(buy_order)
                print(f"🛒 {ticker}: Submitting BUY order...")
                
            elif prediction == "DOWN" and is_owned:
                # Submit Sell Order to close out the position
                sell_order = MarketOrderRequest(
                    symbol=ticker,
                    qty=1,
                    side=OrderSide.SELL,
                    type=OrderType.MARKET,
                    time_in_force=TimeInForce.DAY
                )
                self.trading_client.submit_order(sell_order)
                print(f"🔥 {ticker}: Submitting SELL order...")
                
            else:
                print(f"😴 {ticker}: Holding state (Prediction: {prediction}, Owned: {is_owned})")
                
        except Exception as e:
            print(f"❌ Error trading {ticker}: {e}")

    def run(self):
        print("🤖 Starting Top 100 ML Portfolio Scan...")
        
        # 1. Fetch current portfolio
        try:
            owned_tickers = self.get_portfolio_positions()
            print(f"Currently owning shares in: {owned_tickers}\n")
        except Exception as e:
            print(f"Error fetching portfolio positions: {e}")
            owned_tickers = []
            
        # 2. Process all 100 tickers
        for idx, ticker in enumerate(TICKERS, 1):
            print(f"[{idx}/100] Analyzing {ticker}...")
            
            # Download
            df = self.get_historical_data(ticker)
            if df is None:
                print(f"⚠️ Skipped {ticker} (Insufficient historical data)")
                continue
                
            # Process & Train
            try:
                df_features = self.create_features(df)
                prediction = self.train_and_predict(df_features)
                
                # Trade
                self.execute_trade(ticker, prediction, owned_tickers)
            except Exception as e:
                print(f"⚠️ Error processing brain for {ticker}: {e}")

        print("\n🎉 Portfolio scanning complete!")

if __name__ == "__main__":
    bot = MLMultiTrader()
    bot.run()
