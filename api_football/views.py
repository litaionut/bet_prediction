from django.shortcuts import render, redirect, get_object_or_404
from django.urls import reverse
from django.contrib import messages
from django.conf import settings
from django.db.models import Count
from django.utils import timezone
from django.utils.text import slugify
from urllib.parse import urlencode
from itertools import groupby

from .models import Country, Competition, Game


def _country_name_from_slug(slug):
    """Resolve URL slug to country name (from Country or Competition)."""
    c = Country.objects.filter(slug=slug).first()
    if c:
        return c.name
    for name in (
        Competition.objects.exclude(country="")
        .values_list("country", flat=True)
        .distinct()
    ):
        if (slugify(name) or name.lower().replace(" ", "-")) == slug:
            return name
    return None


def competition_list(request):
    """List all countries that have competitions (data) downloaded."""
    qs = (
        Competition.objects.exclude(country="")
        .values("country")
        .annotate(competition_count=Count("id"), game_count=Count("games"))
        .order_by("country")
    )
    countries_with_data = [
        {
            "country": row["country"],
            "slug": slugify(row["country"]) or row["country"].lower().replace(" ", "-"),
            "competition_count": row["competition_count"],
            "game_count": row["game_count"],
        }
        for row in qs
    ]
    context = {"countries_with_data": countries_with_data}
    return render(request, "api_football/competition_list.html", context)


def country_detail(request, slug):
    """List leagues (competitions) for this country. Click a league to see fixtures."""
    country_name = _country_name_from_slug(slug)
    if not country_name:
        return render(
            request,
            "api_football/country_detail.html",
            {"country_name": None, "country_slug": slug, "competitions": [], "primary_competition": None},
        )
    primary = Competition.get_primary_for_country(country_name)
    competitions = (
        Competition.objects.filter(country=country_name)
        .annotate(game_count=Count("games"))
        .order_by("rank", "api_id", "name")
    )
    context = {
        "country_name": country_name,
        "country_slug": slug,
        "competitions": competitions,
        "primary_competition": primary,
    }
    return render(request, "api_football/country_detail.html", context)


def competition_detail(request, pk):
    """Show one league: seasons and fixtures (upcoming + past)."""
    competition = get_object_or_404(Competition, pk=pk)
    now = timezone.now()
    season_param = request.GET.get("season")
    try:
        season_param = int(season_param) if season_param else None
    except (TypeError, ValueError):
        season_param = None

    base_qs = Game.objects.filter(competition=competition).select_related("home_team", "away_team")
    if season_param is not None:
        base_qs = base_qs.filter(season=season_param)

    seasons = (
        Game.objects.filter(competition=competition)
        .exclude(season__isnull=True)
        .values_list("season", flat=True)
        .distinct()
        .order_by("-season")
    )

    # Upcoming: not finished, order by kickoff ascending (soonest first)
    upcoming = list(
        base_qs.exclude(
            status__in=(Game.Status.FT, Game.Status.AET, Game.Status.AWD, Game.Status.WO, Game.Status.CANC)
        ).order_by("kickoff")
    )
    # Past: finished, order by kickoff descending (most recent first)
    past = list(
        base_qs.filter(
            status__in=(Game.Status.FT, Game.Status.AET, Game.Status.AWD, Game.Status.WO)
        ).order_by("-kickoff")
    )

    country_slug = slugify(competition.country) or competition.country.lower().replace(" ", "-") if competition.country else ""
    context = {
        "competition": competition,
        "country_slug": country_slug,
        "seasons": seasons,
        "season_param": season_param,
        "upcoming": upcoming,
        "past": past,
    }
    return render(request, "api_football/competition_detail.html", context)


from .client import remaining_requests_today
from .sync import (
    sync_countries,
    sync_leagues,
    sync_teams_for_league,
    sync_fixtures,
    sync_fixture_statistics,
    sync_stats_for_games,
    sync_predictions_for_games,
    sync_odds_for_games,
)


def _parse_years(years_str):
    """Parse year(s) string: '2024', '2022,2023,2024', '2020-2024' -> list of ints."""
    if not years_str or not years_str.strip():
        return []
    seen = set()
    for part in years_str.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            try:
                lo, hi = int(a.strip()), int(b.strip())
                if lo > hi:
                    lo, hi = hi, lo
                for y in range(lo, hi + 1):
                    if 2000 <= y <= 2100:
                        seen.add(y)
            except ValueError:
                try:
                    y = int(part.replace("-", "").strip())
                    if 2000 <= y <= 2100:
                        seen.add(y)
                except ValueError:
                    pass
        else:
            try:
                y = int(part)
                if 2000 <= y <= 2100:
                    seen.add(y)
            except ValueError:
                pass
    return sorted(seen)


def sync_page(request):
    """Step 1: Load countries. Step 2: Select countries → load leagues. Step 3: Select leagues + season → Download. Optional: sync stats, predictions, odds."""
    api_key = (getattr(settings, "API_FOOTBALL_KEY", "") or "").strip()
    if not api_key:
        if request.method == "POST":
            messages.error(request, "API_FOOTBALL_KEY is not set. Add it to .env or environment.")
        return render(
            request,
            "api_football/sync.html",
            {
                "countries": [],
                "competitions_by_country": [],
                "api_configured": False,
                "remaining_requests": 0,
                "games_without_stats": 0,
                "games_without_prediction": 0,
                "api_key_hint": "Add API_FOOTBALL_KEY=your_key to a .env file in the project root (same folder as manage.py), then restart the server.",
            },
        )

    if request.method == "POST":
        action = request.POST.get("action")
        if not action:
            messages.warning(request, "No action selected.")
            return redirect("api_football:sync")

        try:
            if action == "load_countries":
                c, u = sync_countries()
                messages.success(request, f"Countries: {c} created, {u} updated. Select countries and click Load competitions.")
            elif action == "load_leagues":
                selected = request.POST.getlist("country")
                if not selected:
                    messages.warning(request, "Select at least one country.")
                    return redirect("api_football:sync")
                total_c, total_u = 0, 0
                for country_name in selected:
                    cn, un = sync_leagues(country=country_name)
                    total_c += cn
                    total_u += un
                messages.success(request, f"Leagues: {total_c} created, {total_u} updated. Select competitions and year, then Download.")
                qs = urlencode([("country", n) for n in selected], doseq=True)
                return redirect(reverse("api_football:sync") + "?" + qs)
            elif action == "download":
                selected = request.POST.getlist("competition")
                years_str = (request.POST.get("years", "").strip() or request.POST.get("year", "").strip())
                if not selected or not years_str:
                    messages.warning(request, "Select at least one competition and enter year(s).")
                    return redirect("api_football:sync")
                years = _parse_years(years_str)
                if not years:
                    messages.warning(request, "Enter at least one valid year (e.g. 2024 or 2022,2023,2024 or 2020-2024).")
                    return redirect("api_football:sync")
                total_teams_c, total_teams_u = 0, 0
                total_fix_c, total_fix_u = 0, 0
                for comp_id in selected:
                    comp_id = int(comp_id)
                    for year in years:
                        tc, tu = sync_teams_for_league(comp_id, year)
                        total_teams_c += tc
                        total_teams_u += tu
                        fc, fu = sync_fixtures(league_id=comp_id, season=year)
                        total_fix_c += fc
                        total_fix_u += fu
                messages.success(
                    request,
                    f"Teams: {total_teams_c} created, {total_teams_u} updated. Fixtures: {total_fix_c} created, {total_fix_u} updated.",
                )
                qs = urlencode(
                    [("country", n) for n in request.POST.getlist("country")] + [("years", years_str)],
                    doseq=True,
                )
                if qs:
                    return redirect(reverse("api_football:sync") + "?" + qs)
            elif action == "sync_stats":
                limit_str = request.POST.get("stats_limit", "").strip()
                limit = int(limit_str) if limit_str else None
                synced, skipped = sync_stats_for_games(limit=limit)
                messages.success(
                    request,
                    f"Fixture statistics: {synced} games synced, {skipped} skipped.",
                )
            elif action == "sync_predictions":
                limit_str = request.POST.get("predictions_limit", "").strip()
                limit = int(limit_str) if limit_str else None
                cr, up, sk = sync_predictions_for_games(limit=limit)
                messages.success(
                    request,
                    f"Predictions: {cr} created, {up} updated, {sk} skipped.",
                )
            elif action == "sync_odds":
                limit_str = request.POST.get("odds_limit", "").strip()
                limit = int(limit_str) if limit_str else None
                _, up, sk = sync_odds_for_games(limit=limit)
                messages.success(
                    request,
                    f"Odds: {up} updated, {sk} skipped.",
                )
            else:
                messages.warning(request, "Unknown action.")
        except ValueError as e:
            messages.error(request, str(e))
        except Exception as e:
            messages.error(request, f"Sync failed: {e}")
        return redirect("api_football:sync")

    # GET: show form
    countries = list(Country.objects.order_by("name"))
    selected_countries = request.GET.getlist("country")
    years_param = request.GET.get("years", "").strip() or request.GET.get("year", "2024").strip()

    if selected_countries:
        selected_lower = [s.lower() for s in selected_countries]
        all_comp_countries = set(
            Competition.objects.exclude(country="").values_list("country", flat=True).distinct()
        )
        matching_countries = [
            c for c in all_comp_countries
            if c and c.lower() in selected_lower
        ]
        comps = (
            Competition.objects.filter(country__in=matching_countries)
            .annotate(game_count=Count("games"))
            .order_by("country", "rank", "api_id", "name")
        )
        competitions_by_country = [
            (country, list(g), Competition.get_primary_for_country(country))
            for country, g in groupby(comps, key=lambda c: c.country)
        ]
        actual_countries_in_db = (
            list(
                Competition.objects.exclude(country="")
                .values_list("country", flat=True)
                .distinct()
                .order_by("country")
            )
            if not competitions_by_country and selected_countries
            else []
        )
    else:
        competitions_by_country = []
        actual_countries_in_db = []

    # Games without statistics (finished only)
    games_without_stats = (
        Game.objects.filter(
            status__in=(Game.Status.FT, Game.Status.AET, Game.Status.AWD, Game.Status.WO)
        )
        .annotate(num_stats=Count("statistics_rows"))
        .filter(num_stats=0)
        .count()
    )
    try:
        games_without_prediction = Game.objects.filter(prediction__isnull=True).count()
    except Exception:
        games_without_prediction = 0

    try:
        remaining = remaining_requests_today()
    except Exception:
        remaining = 0

    context = {
        "countries": countries,
        "selected_countries": selected_countries,
        "competitions_by_country": competitions_by_country,
        "years_param": years_param,
        "api_configured": True,
        "actual_countries_in_db": actual_countries_in_db,
        "remaining_requests": remaining,
        "games_without_stats": games_without_stats,
        "games_without_prediction": games_without_prediction,
    }
    return render(request, "api_football/sync.html", context)
