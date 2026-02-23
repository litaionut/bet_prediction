from django.contrib import admin

from .models import BetJournalEntry


@admin.register(BetJournalEntry)
class BetJournalEntryAdmin(admin.ModelAdmin):
    list_display = ("game", "choice", "result", "created_at")
    list_filter = ("result", "choice")
    search_fields = ("game__home_team__name", "game__away_team__name")
    readonly_fields = ("result",)
    ordering = ("-created_at",)
