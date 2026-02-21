from django.db import models
from django.utils.text import slugify


class Country(models.Model):
    """Country from API-Football (countries endpoint). Used for sync and navigation."""
    name = models.CharField(max_length=100, unique=True)
    code = models.CharField(max_length=10, blank=True)
    slug = models.SlugField(max_length=100, unique=True, blank=True)

    class Meta:
        ordering = ["name"]
        verbose_name_plural = "countries"

    def save(self, *args, **kwargs):
        if not self.slug and self.name:
            self.slug = slugify(self.name)
        super().save(*args, **kwargs)

    def __str__(self):
        return self.name


class Competition(models.Model):
    """League/cup from API-Football (leagues endpoint)."""
    api_id = models.IntegerField(unique=True)
    name = models.CharField(max_length=200)
    country = models.CharField(max_length=100, blank=True)
    logo_url = models.URLField(blank=True)
    type = models.CharField(max_length=50, blank=True)
    rank = models.IntegerField(null=True, blank=True, db_index=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["country", "rank", "name"]

    def __str__(self):
        if self.country:
            return f"{self.name} ({self.country})"
        return self.name

    @classmethod
    def get_primary_for_country(cls, country_name):
        qs = cls.objects.filter(country=country_name)
        if not qs.exists():
            return None
        with_rank = qs.exclude(rank__isnull=True).order_by("rank").first()
        if with_rank:
            return with_rank
        league_type = qs.filter(type__iexact="League").order_by("api_id").first()
        if league_type:
            return league_type
        return qs.order_by("api_id").first()


class Team(models.Model):
    """Team from API-Football (teams endpoint)."""
    api_id = models.IntegerField(unique=True)
    name = models.CharField(max_length=200)
    code = models.CharField(max_length=20, blank=True)
    country = models.CharField(max_length=100, blank=True)
    logo_url = models.URLField(blank=True)
    founded = models.IntegerField(null=True, blank=True)
    venue_name = models.CharField(max_length=200, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["name"]

    def __str__(self):
        return self.name or f"Team {self.api_id}"


class Game(models.Model):
    """Fixture from API-Football (fixtures endpoint)."""
    class Status(models.TextChoices):
        TBD = "TBD", "TBD"
        NS = "NS", "Not Started"
        LIVE = "LIVE", "Live"
        FT = "FT", "Finished"
        AET = "AET", "After Extra Time"
        PST = "PST", "Postponed"
        CANC = "CANC", "Cancelled"
        AWD = "AWD", "Awarded"
        WO = "WO", "Walkover"

    api_id = models.IntegerField(unique=True)
    competition = models.ForeignKey(
        Competition, on_delete=models.CASCADE, related_name="games", null=True, blank=True
    )
    season = models.IntegerField(null=True, blank=True)
    round_label = models.CharField(max_length=100, blank=True)
    home_team = models.ForeignKey(Team, on_delete=models.CASCADE, related_name="home_games")
    away_team = models.ForeignKey(Team, on_delete=models.CASCADE, related_name="away_games")
    kickoff = models.DateTimeField(null=True, blank=True)
    status = models.CharField(max_length=10, choices=Status.choices, default=Status.NS, db_index=True)
    home_goals = models.IntegerField(null=True, blank=True)
    away_goals = models.IntegerField(null=True, blank=True)
    venue_name = models.CharField(max_length=200, blank=True)
    referee = models.CharField(max_length=200, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-kickoff"]

    def __str__(self):
        return f"{self.home_team.name} vs {self.away_team.name}"

    @property
    def score_display(self):
        if self.status == self.Status.FT and self.home_goals is not None:
            return f"{self.home_goals} - {self.away_goals}"
        return "-"

    def is_finished(self):
        return self.status in (self.Status.FT, self.Status.AET, self.Status.AWD, self.Status.WO)


class GameStatistics(models.Model):
    """Per-team statistics for a fixture from API-Football (fixtures/statistics)."""
    game = models.ForeignKey(Game, on_delete=models.CASCADE, related_name="statistics_rows")
    team = models.ForeignKey(Team, on_delete=models.CASCADE, related_name="fixture_statistics")
    statistics = models.JSONField(default=list)

    class Meta:
        unique_together = [["game", "team"]]
        ordering = ["game", "team"]
        verbose_name_plural = "game statistics"

    def __str__(self):
        return f"{self.game} – {self.team}"


class GamePrediction(models.Model):
    """API-Football prediction for a fixture. One per game."""
    game = models.OneToOneField(Game, on_delete=models.CASCADE, related_name="prediction")
    raw = models.JSONField(default=dict)
    odds = models.JSONField(default=list, blank=True)

    class Meta:
        ordering = ["game"]

    def __str__(self):
        return f"Prediction for {self.game}"


class APIRequestLog(models.Model):
    """Track API-Football requests per day (e.g. 7500/day limit)."""
    date = models.DateField(unique=True, db_index=True)
    request_count = models.PositiveIntegerField(default=0)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-date"]
        verbose_name = "API request log"
        verbose_name_plural = "API request logs"

    def __str__(self):
        return f"{self.date}: {self.request_count} requests"
