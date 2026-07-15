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
TICKERS = [
    "AAPL", "MSFT", "AMZN", "NVDA", "GOOGL", "GOOG", "META", "BRK-B", "TSLA", "UNH",
    "JNJ", "JPM", "XOM", "V", "PG", "MA", "AVGO", "HD", "CVX", "MRK", 
    "ABBV", "ADBE", "COST", "PEP", "KO", "TMO", "MCD", "WMT", "BAC", "CSCO", 
    "CRM", "ACN", "LLY", "ABT", "ORCL", "VZ", "INTC", "TXN", "QCOM", "CMCSA", 
    "AMGN", "NFLX", "AMD", "DIS", "PM", "NKE", "COP", "HON", "T", "IBM", 
    "JCI", "GE", "UNP", "LOW", "AXP", "INTU", "SPGI", "SBUX", "EL", "PLD", 
    "AMAT", "MDLZ", "CAT", "GILD", "RTX", "LMT", "BKNG", "TJX", "ADI", "C", 
    "ISRG", "SYK", "REGN", "VRTX", "ADP", "MMC", "CI", "DHR", "MU", "LRCX", 
    "SLB", "GEHC", "NOW", "PANW", "SNPS", "MELI", "CDNS", "EQIX", "SO", "D", 
    "KDP", "CL", "WM", "NOC", "BSX", "GPN", "HCA"
]

TICKERS = list(set([t.replace(".", "-") for t in TICKERS]))[:100]

# Secure API Credentials
API_KEY = os.environ.get("ALPACA_KEY")
SECRET_KEY = os.environ.get("ALPACA_SECRET")

class ML15MinAdaptiveTrader:
    def __init__(self):
        self.trading_client = TradingClient(API_KEY, SECRET_KEY, paper=True)
        
    def get_historical_data(self, ticker):
        """Downloads historical 15-minute interval data and cuts it to the last 90 bars."""
        # 10 days of trading covers roughly 260 candles (15-min intervals)
        df = yf.download(
            ticker, 
            period="10d", 
            interval="15m", # CHANGED: Now pulls 15-minute intervals instead of 1h
            progress=False, 
            multi_level_index=False
        )
        
        if df.empty or len(df) < 90:
            return None
            
        # Cut to exactly the last 90 trading candles (each representing 15 minutes)
        return df.tail(90)

    def create_features(self, df):
        """Calculates indicators based on the 15-minute intervals."""
        df = df.copy()
        df['SMA_15'] = df['Close'].rolling(window=15).mean()
        df['Returns'] = df['Close'].pct_change()
        df['Target'] = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)
        df.dropna(inplace=True)
        return df

    def evaluate_recent_mistakes(self, df):
        """
        LEARNING LOOP: Simulates trading over the last 24 periods (6 hours of trading).
        Finds out exactly when the AI made mistakes and calculates our current "trust score."
        """
        features = ['SMA_15', 'Returns']
        X = df[features].values
        y = df['Target'].values
        
        test_intervals = 24 # Evaluate decisions candle-by-candle over the last 24 periods
        if len(df) < (test_intervals + 10):
            return 0.50, 0 

        mistakes = 0
        correct = 0
        
        for i in range(len(df) - test_intervals, len(df) - 1):
            test_model = RandomForestClassifier(n_estimators=50, random_state=42)
            test_model.fit(X[:i], y[:i])
            
            pred = test_model.predict(X[i].reshape(1, -1))[0]
            if pred == y[i]:
                correct += 1
            else:
                mistakes += 1
                
        accuracy = correct / (correct + mistakes) if (correct + mistakes) > 0 else 0.50
        return accuracy, mistakes

    def predict_next_interval(self, df):
        """Trains on all data to predict the movement for the upcoming 15 minutes."""
        features = ['SMA_15', 'Returns']
        X = df[features].values
        y = df['Target'].values
        
        model = RandomForestClassifier(n_estimators=50, random_state=42)
        model.fit(X[:-1], y[:-1])
        
        last_row = X[-1].reshape(1, -1)
        prediction = model.predict(last_row)[0]
        
        return "UP" if prediction == 1 else "DOWN"

    def get_portfolio_positions(self):
        """Returns symbols currently owned."""
        positions = self.trading_client.get_all_positions()
        return [pos.symbol for pos in positions]

    def execute_trade(self, ticker, prediction, accuracy, mistakes, owned_tickers):
        """Executes trades only if the AI has proven to be reliable over the last 24 intervals."""
        is_owned = ticker in owned_tickers
        ACCURACY_THRESHOLD = 0.55 # Must maintain a 55%+ accuracy rate to place new trades
        
        try:
            if prediction == "UP" and not is_owned:
                if accuracy >= ACCURACY_THRESHOLD:
                    buy_order = MarketOrderRequest(
                        symbol=ticker,
                        qty=1,
                        side=OrderSide.BUY,
                        type=OrderType.MARKET,
                        time_in_force=TimeInForce.DAY
                    )
                    self.trading_client.submit_order(buy_order)
                    print(f"🛒 {ticker}: BUY order! 15m Accuracy: {accuracy:.1%} (Mistakes in last 24 intervals: {mistakes})")
                else:
                    print(f"🛑 {ticker}: BLOCKED BUY (15m Accuracy: {accuracy:.1%} | Mistakes: {mistakes}) - Safeguarding capital.")
                
            elif prediction == "DOWN" and is_owned:
                # If prediction flips to DOWN, sell to protect cash
                sell_order = MarketOrderRequest(
                    symbol=ticker,
                    qty=1,
                    side=OrderSide.SELL,
                    type=OrderType.MARKET,
                    time_in_force=TimeInForce.DAY
                )
                self.trading_client.submit_order(sell_order)
                print(f"🔥 {ticker}: SELLING position (15m prediction turned DOWN)!")
                
            else:
                print(f"😴 {ticker}: Holding state (Pred: {prediction}, Owned: {is_owned}, Current Mistakes: {mistakes})")
                
        except Exception as e:
            print(f"❌ Error executing transaction for {ticker}: {e}")

    def run(self):
        print("🤖 Starting 15-Minute Self-Correcting Portfolio Scan...")
        
        try:
            owned_tickers = self.get_portfolio_positions()
            print(f"Currently owning shares in: {owned_tickers}\n")
        except Exception as e:
            print(f"Error fetching portfolio: {e}")
            owned_tickers = []
            
        for idx, ticker in enumerate(TICKERS, 1):
            df = self.get_historical_data(ticker)
            if df is None:
                print(f"⚠️ Skipped {ticker} (Insufficient 15m data)")
                continue
                
            try:
                df_features = self.create_features(df)
                
                # Evaluate recent performance & mistake count
                accuracy, mistakes = self.evaluate_recent_mistakes(df_features)
                
                # Predict next interval's direction
                prediction = self.predict_next_interval(df_features)
                
                # Execute filtered trades
                print(f"[{idx}/100] 15m Analysis for {ticker}...")
                self.execute_trade(ticker, prediction, accuracy, mistakes, owned_tickers)
                
            except Exception as e:
                print(f"⚠️ Error processing 15m trade run for {ticker}: {e}")

        print("\n🎉 15-Minute adaptive cycle run complete!")

if __name__ == "__main__":
    bot = ML15MinAdaptiveTrader()
    bot.run()
