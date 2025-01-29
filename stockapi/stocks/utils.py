import requests

ALPHA_VANTAGE_API_KEY = 'your_api_key_here'  # Replace with your Alpha Vantage API key

def fetch_stock_data(symbol):
    url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}&apikey={ALPHA_VANTAGE_API_KEY}'
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    return None