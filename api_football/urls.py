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
    path("games/today/", views.game_list_today, name="game_list_today"),
    path("games/<int:pk>/statistics/", views.game_statistics, name="game_statistics"),
    path("sync-day-results/", views.sync_day_results, name="sync_day_results"),
    path("predictions/", views.gemini_predictions, name="gemini_predictions"),
    # Bet journal: list + Add new entry → country → games (last 48h) → over/under → save
    path("journal/", views.journal_index, name="journal_index"),
    path("journal/add/", views.journal_add, name="journal_add"),
    path("journal/entries/", views.journal_list, name="journal_list"),
    path("journal/record/<int:pk>/", views.journal_record, name="journal_record"),
    path("journal/<slug:slug>/", views.journal_games, name="journal_games"),
    # Cron: update today's results (call every hour; protect with CRON_SECRET)
    path("cron/update-today-results/", views.cron_update_today_results, name="cron_update_today_results"),
]
