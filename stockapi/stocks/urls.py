from django.urls import path
from .views import StockListView, StockDetailView

urlpatterns = [
    path('stocks/', StockListView.as_view(), name='stock-list'),
    path('stocks/<str:symbol>/', StockDetailView.as_view(), name='stock-detail'),
]