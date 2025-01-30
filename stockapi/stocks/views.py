import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from datetime import datetime
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from xgboost import XGBClassifier
from dateutil.relativedelta import relativedelta
# from .utils import fetch_stock_data

class StockListView(APIView):
    def get(self, request):
        # Fetch stock symbols dynamically from yfinance
        tickers = yf.Tickers('AAPL GOOGL MSFT AMZN')
        symbols = list(tickers.tickers.keys())
        
        search_query = request.query_params.get('search', None)  # Get search query

        stocks = []
        for symbol in symbols:
            # If search query is provided, filter stocks by symbol or name
            if search_query and search_query.lower() not in symbol.lower():
                continue

            # Fetch data for the stock
            today = datetime.today().strftime('%Y-%m-%d')
            start_date = "2024-01-01"
            data = yf.download(symbol, start=start_date, end=today)

            if data.empty:
                continue

            # Feature Engineering
            data['RSI'] = self.calculate_rsi(data)
            data['MA_20'], data['Upper_Band'], data['Lower_Band'] = self.calculate_bb(data)
            data['Price_Change'] = data['Close'].diff()
            data['5_MA'] = data['Close'].rolling(window=5).mean()
            data['10_MA'] = data['Close'].rolling(window=10).mean()
            data['Tomorrow_Trend'] = (data['Close'].shift(-1) > data['Close']).astype(int)
            data = data.dropna()

            # Features and target
            X = data[['Close', '10_MA', 'RSI', 'Upper_Band', 'Lower_Band']]
            y = data['Tomorrow_Trend']

            # Split the Data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Train the Model
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)

            # Predict Tomorrow's Trend
            latest_data = X.iloc[-1].values.reshape(1, -1)
            predicted_trend = model.predict(latest_data.flatten().reshape(1, -1))[0]
            action = "Buy" if predicted_trend == 1 else "Sell"

            # Append stock data
            stock = {
                'symbol': symbol,
                'name': symbol,  # Replace with actual name if available
                'price': float(data['Close'].iloc[-1]),
                'high': float(data['High'].iloc[-1]),
                'low': float(data['Low'].iloc[-1]),
                'open': float(data['Open'].iloc[-1]),
                'close': float(data['Close'].iloc[-1]),
                'predicted_action': action
            }
            stocks.append(stock)

        return Response(stocks, status=status.HTTP_200_OK)

    def calculate_rsi(self, data, window=45):
        delta = data['Close'].diff()
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = pd.Series(gain, index=data.index).rolling(window=window).mean()
        avg_loss = pd.Series(loss, index=data.index).rolling(window=window).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def calculate_bb(self, data):
        MA_20 = data['Close'].rolling(window=20).mean()
        Upper_Band = MA_20 + (data['Close'].rolling(window=20).std() * 2)
        Lower_Band = MA_20 - (data['Close'].rolling(window=20).std() * 2)
        return MA_20, Upper_Band, Lower_Band

# class StockDetailView(APIView):
#     def get(self, request, symbol):
#         # Fetch stock data using yfinance
#         today = datetime.today().strftime('%Y-%m-%d')
#         start_date = "2024-01-01"
#         data = yf.download(symbol, start=start_date, end=today)

#         # Feature Engineering
#         data['RSI'] = self.calculate_rsi(data)
#         data['MA_20'], data['Upper_Band'], data['Lower_Band'] = self.calculate_bb(data)
#         data['Price_Change'] = data['Close'].diff()
#         data['5_MA'] = data['Close'].rolling(window=5).mean()
#         data['10_MA'] = data['Close'].rolling(window=10).mean()
#         data['Tomorrow_Trend'] = (data['Close'].shift(-1) > data['Close']).astype(int)
#         data = data.dropna()

#         # Features and target
#         X = data[['Close', '10_MA', 'RSI', 'Upper_Band', 'Lower_Band']]
#         y = data['Tomorrow_Trend']

#         # Train-test split
#         X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#         # Train the Random Forest model
#         model = RandomForestClassifier(n_estimators=100, random_state=42)
#         model.fit(X_train, y_train)

#         # Predict tomorrow's trend
#         latest_data = X.iloc[-1].values.reshape(1, -1)
#         predicted_trend = model.predict(latest_data)[0]
#         action = "Buy" if predicted_trend == 1 else "Sell"

#         # Prepare historical data for the response
#         historical_data = []
#         for index, row in data.iterrows():
#             historical_data.append({
#                 'date': index.strftime('%Y-%m-%d'),
#                 'open': row['Open'],
#                 'high': row['High'],
#                 'low': row['Low'],
#                 'close': row['Close'],
#                 'volume': row['Volume'],
#                 'rsi': row['RSI'],
#                 'ma_20': row['MA_20'],
#                 'upper_band': row['Upper_Band'],
#                 'lower_band': row['Lower_Band'],
#             })

#         # Prepare the response
#         response_data = {
#             'symbol': symbol,
#             'name': symbol,  # Replace with actual name if available
#             'latest_price': data['Close'].iloc[-1],
#             'predicted_action': action,
#             'historical_data': historical_data,
#         }

#         return Response(response_data, status=status.HTTP_200_OK)

#     @staticmethod
#     def calculate_rsi(data, window=45):
#         delta = data['Close'].diff()  # Price change
#         gain = np.where(delta > 0, delta, 0)  # Gains
#         loss = np.where(delta < 0, -delta, 0)  # Losses

#         # Ensure gain and loss are 1D
#         gain = gain.ravel()
#         loss = loss.ravel()

#         # Calculate rolling averages
#         avg_gain = pd.Series(gain, index=data.index).rolling(window=window).mean()
#         avg_loss = pd.Series(loss, index=data.index).rolling(window=window).mean()

#         # Calculate RSI
#         rs = avg_gain / avg_loss
#         rsi = 100 - (100 / (1 + rs))

#         return rsi

#     @staticmethod
#     def calculate_bb(data):
#         MA_20 = data['Close'].rolling(window=20).mean()
#         Upper_Band = MA_20 + (data['Close'].rolling(window=20).std() * 2)
#         Lower_Band = MA_20 - (data['Close'].rolling(window=20).std() * 2)
#         return MA_20, Upper_Band, Lower_Band


class StockDetailView(APIView):
    def get(self, request, symbol):
        # Fetch stock data using yfinance
        # Today's date in 'YYYY-MM-DD' format
        today = datetime.today().strftime('%Y-%m-%d')
        # start_date = (datetime.today() - relativedelta(months=2)).strftime('%Y-%m-%d')
        start_date = "2024-01-01"  # Start date for historical data
        data = yf.download(symbol, start=start_date, end=today)

        # Feature Engineering
        data["Price Change"] = data["Close"].diff()  # Difference between today & yesterday
        data["SMA_5"] = data["Close"].rolling(window=5).mean()  # 5-day SMA
        data["SMA_10"] = data["Close"].rolling(window=10).mean()  # 10-day SMA
        data["RSI"] = 100 - (100 / (1 + (data["Close"].pct_change() + 1).rolling(45).mean()))

        # Bollinger Bands Calculation (20-day SMA & 2 Standard Deviations)
        data["SMA_20"] = data["Close"].rolling(window=20).mean()
        data["BB_Std"] = data["Close"].rolling(window=20).std(ddof=0)  # Ensures single-column output
        data["BB_Upper"] = data["SMA_20"] + (2 * data["BB_Std"])
        data["BB_Lower"] = data["SMA_20"] - (2 * data["BB_Std"])
        data["BB_Width"] = data["BB_Upper"] - data["BB_Lower"]

        # Create the target variable (Buy=1, Sell=0)
        data["Target"] = (data["Close"].shift(-1) > data["Close"]).astype(int)  # If tomorrow's close is higher, Buy (1)

        # Drop NaN values (caused by rolling calculations)
        data.dropna(inplace=True)

        # Select Features
        X = data[["Close", "SMA_5", "SMA_10", "RSI", "BB_Upper", "BB_Lower"]]
        y = data["Target"]

        # Train-Test Split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

        # Train Random Forest Model
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_model.fit(X_train, y_train)

        # Train XGBoost Model
        xgb_model = XGBClassifier(n_estimators=100, learning_rate=0.05, max_depth=5, use_label_encoder=False, eval_metric="logloss")
        xgb_model.fit(X_train, y_train)

        # Predict tomorrow's trend using both models
        latest_data = X.iloc[-1].values.reshape(1, -1)
        rf_prediction = rf_model.predict(latest_data)[0]
        xgb_prediction = xgb_model.predict(latest_data)[0]

        rf_action = "Buy" if rf_prediction == 1 else "Sell"
        xgb_action = "Buy" if xgb_prediction == 1 else "Sell"

        # Prepare historical data for the response
        historical_data = []
        for index, row in data.iterrows():
           historical_data.append({
                'date': index.strftime('%Y-%m-%d'),
                'open': float(row['Open']),  # Convert to float to ensure it's a single number
                'high': float(row['High']),  # Convert to float to ensure it's a single number
                'low': float(row['Low']),    # Convert to float to ensure it's a single number
                'close': float(row['Close']),  # Convert to float to ensure it's a single number
                'volume': int(row['Volume']),  # Convert to int to ensure it's a single number
                'rsi': float(row['RSI']),  # Convert to float to ensure it's a single number
                'sma_5': float(row['SMA_5']),  # Convert to float to ensure it's a single number
                'sma_10': float(row['SMA_10']),  # Convert to float to ensure it's a single number
                'sma_20': float(row['SMA_20']),  # Convert to float to ensure it's a single number
                'bb_upper': float(row['BB_Upper']),  # Convert to float to ensure it's a single number
                'bb_lower': float(row['BB_Lower']),  # Convert to float to ensure it's a single number
                'bb_width': float(row['BB_Width']),  # Convert to float to ensure it's a single number
            })

        # Prepare the response
        response_data = {
            'symbol': symbol,
            'name': symbol,  # Replace with actual name if available
            'latest_price': float(data['Close'].iloc[-1]),
            'rf_predicted_action': rf_action,
            'xgb_predicted_action': xgb_action,
            'historical_data': historical_data,
        }

        return Response(response_data, status=status.HTTP_200_OK)