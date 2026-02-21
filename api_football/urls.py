from django.urls import path
from . import views

app_name = "api_football"

urlpatterns = [
    path("sync/", views.sync_page, name="sync"),
    path("competitions/", views.competition_list, name="competition_list"),
    path("countries/<slug:slug>/", views.country_detail, name="country_detail"),
    path("competitions/<int:pk>/", views.competition_detail, name="competition_detail"),
    path("competitions/<int:pk>/delete/", views.competition_delete, name="competition_delete"),
    path("competitions/delete-selected/", views.competition_bulk_delete, name="competition_bulk_delete"),
    path("games/<int:pk>/statistics/", views.game_statistics, name="game_statistics"),
    path("predictions/", views.gemini_predictions, name="gemini_predictions"),
]
