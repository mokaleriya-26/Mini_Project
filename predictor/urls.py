# predictor/urls.py

from django.urls import path
from . import views

urlpatterns = [
    path('predict/<str:ticker>/', views.get_stock_prediction),
]