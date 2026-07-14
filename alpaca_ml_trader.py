import os
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# Alpaca official SDK imports
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

# ==========================================
# CONFIGURATION - PUT YOUR ALPACA KEYS HERE
# ==========================================
API_KEY = "PKRQKICOCXMVDRWR2YO2FRZBXW"
SECRET_KEY = "7TUzwKBTZsVkwCPy4DYV9kQCCuScumLT6uqsyAgaYDnz"
TICKER = "AAPL"  # The stock we want to trade

class MLTradingBot:
    def __init__(self, ticker):
        self.ticker = ticker
        # Initialize the Alpaca Client (set paper=True for safety!)
        self.trading_client = TradingClient(API_KEY, SECRET_KEY, paper=True)
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        
    def prepare_data(self):
        print(f"📥 Downloading historical data for {self.ticker}...")
        # Get 2 years of daily data to train our brain
        df = yf.download(self.ticker, period="2y", interval="1d")
        
        # Clean multi-index columns if present in yfinance
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        # Create features (Technical Indicators)
        df['Return'] = df['Close'].pct_change()
        df['MA_5'] = df['Close'].rolling(window=5).mean()
        df['MA_20'] = df['Close'].rolling(window=20).mean()
        df['Volatility'] = df['Return'].rolling(window=5).std()
        
        # Target variable: Will tomorrow's close be HIGHER than today's? (1 = Yes, 0 = No)
        df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
        
        df.dropna(inplace=True)
        return df

    def train_brain(self, df):
        print("🧠 Training the Machine Learning brain...")
        # Features used to make predictions
        feature_cols = ['Close', 'Return', 'MA_5', 'MA_20', 'Volatility']
        X = df[feature_cols]
        y = df['Target']
        
        # Split data: Use older data to train, leave the absolute latest row to test tomorrow
        X_train = X.iloc[:-1]
        y_train = y.iloc[:-1]
        
        self.model.fit(X_train, y_train)
        
        # Grab the latest row of data (today's real market values)
        latest_market_data = X.iloc[[-1]]
        prediction = self.model.predict(latest_market_data)[0]
        
        # Calculate a quick baseline accuracy check on training data
        train_acc = self.model.score(X_train, y_train)
        print(f"✅ Brain trained. Training Fit Accuracy: {train_acc*100:.1f}%")
        
        return prediction

    def execute_trade(self, prediction):
        print(f"🔮 Model Prediction for next market close: {'UP' if prediction == 1 else 'DOWN'}")
        
        # Check current positions to see if we already own it
        positions = self.trading_client.get_all_positions()
        already_owned = any(p.symbol == self.ticker for p in positions)
        
        if prediction == 1 and not already_owned:
            print("🚀 Signal is UP and we don't own it. Submitting BUY order...")
            market_order_data = MarketOrderRequest(
                symbol=self.ticker,
                qty=1,
                side=OrderSide.BUY,
                time_in_force=TimeInForce.DAY
            )
            self.trading_client.submit_order(order_data=market_order_data)
            print("🛒 Buy order sent successfully!")
            
        elif prediction == 0 and already_owned:
            print("📉 Signal is DOWN and we hold a position. Submitting SELL order...")
            market_order_data = MarketOrderRequest(
                symbol=self.ticker,
                qty=1,
                side=OrderSide.SELL,
                time_in_force=TimeInForce.DAY
            )
            self.trading_client.submit_order(order_data=market_order_data)
            print("💰 Sell order sent successfully!")
            
        else:
            print("💤 No action required. Holding current state.")

# ==========================================
# RUN THE BOT
# ==========================================
if __name__ == "__main__":
    bot = MLTradingBot(ticker=TICKER)
    data = bot.prepare_data()
    next_day_prediction = bot.train_brain(data)
    bot.execute_trade(next_day_prediction)