# In predictor/urls.py

from django.urls import path
from .views import get_stock_prediction

urlpatterns = [
    path('predict/<str:ticker>/', get_stock_prediction),
]