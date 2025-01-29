from rest_framework import serializers

class StockSerializer(serializers.Serializer):
    symbol = serializers.CharField()
    name = serializers.CharField()
    price = serializers.FloatField()
    high = serializers.FloatField()
    low = serializers.FloatField()
    open = serializers.FloatField()
    close = serializers.FloatField()