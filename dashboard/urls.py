from django.urls import path

from . import views

app_name = "dashboard"

urlpatterns = [
    path("", views.index, name="index"),
    path("betting-calculator/", views.betting_calculator, name="betting_calculator"),
]
