from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .utils import fetch_stock_data


class StockListView(APIView):
    def get(self, request):
        # Example list of stock symbols
        symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN']
        search_query = request.query_params.get('search', None)  # Get search query

        stocks = []
        for symbol in symbols:
            # If search query is provided, filter stocks by symbol or name
            if search_query and search_query.lower() not in symbol.lower():
                continue

            data = fetch_stock_data(symbol)
            if data and 'Time Series (Daily)' in data:
                latest_data = list(data['Time Series (Daily)'].values())[0]
                stock = {
                    'symbol': symbol,
                    'name': symbol,  # Replace with actual name if available
                    'price': float(latest_data['4. close']),
                    'high': float(latest_data['2. high']),
                    'low': float(latest_data['3. low']),
                    'open': float(latest_data['1. open']),
                    'close': float(latest_data['4. close']),
                }
                stocks.append(stock)

        return Response(stocks, status=status.HTTP_200_OK)

class StockDetailView(APIView):
    def get(self, request, symbol):
        data = fetch_stock_data(symbol)
        if data and 'Time Series (Daily)' in data:
            time_series = data['Time Series (Daily)']
            labels = list(time_series.keys())  # Dates
            values = [{
                'open': float(day_data['1. open']),
                'high': float(day_data['2. high']),
                'low': float(day_data['3. low']),
                'close': float(day_data['4. close']),
            } for day_data in time_series.values()]

            response_data = {
                'symbol': symbol,
                'name': symbol,  # Replace with actual name if available
                'labels': labels,
                'data': values,
            }
            return Response(response_data, status=status.HTTP_200_OK)
        return Response({'error': 'Stock not found'}, status=status.HTTP_404_NOT_FOUND)