from django.urls import path
from . import views

app_name = "api_football"

urlpatterns = [
    path("sync/", views.sync_page, name="sync"),
    path("competitions/", views.competition_list, name="competition_list"),
]
