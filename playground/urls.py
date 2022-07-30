from django.urls import path
from . import views

urlpatterns = [
    path('hello/', views.hello),
    path('plot/', views.plot),
    path('', views.plot2),
    path('pltest/', views.pltest)
]
