"""
Sync API-Football response data into Competition, Team, Game, GameStatistics models.
"""
from datetime import datetime
from django.utils import timezone
from django.utils.text import slugify
from django.db.models import Count

from .client import (
    get_countries,
    get_leagues,
    get_teams,
    get_fixtures,
    get_fixture_statistics,
    get_fixture_predictions,
    get_fixture_odds,
    remaining_requests_today,
)
from .models import Country, Competition, Team, Game, GameStatistics, GamePrediction


def sync_countries():
    """Fetch countries from API and create/update Country records. Returns (created, updated)."""
    raw = get_countries()
    created = updated = 0
    for item in raw:
        if not item:
            continue
        if isinstance(item, dict):
            name = item.get("name") or item.get("country") or ""
            code = str(item.get("code", "") or "")[:10]
        else:
            name = str(item)
            code = ""
        if not name:
            continue
        slug = slugify(name) or name.lower().replace(" ", "-")
        defaults = {"code": code, "slug": slug}
        obj, was_created = Country.objects.update_or_create(name=name, defaults=defaults)
        if was_created:
            created += 1
        else:
            updated += 1
    return created, updated


def _parse_datetime(s):
    if not s:
        return None
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            return timezone.make_aware(dt, timezone.get_current_timezone())
        return dt
    except (ValueError, TypeError):
        return None


def sync_leagues(country=None, season=None):
    """Fetch leagues and create/update Competition records. Returns (created, updated)."""
    raw = get_leagues(country=country, season=season)
    created = updated = 0
    for item in raw:
        if not item or "league" not in item:
            continue
        league = item["league"]
        api_id = league.get("id")
        if not api_id:
            continue
        country_val = league.get("country", "")
        if not country_val and item.get("country"):
            co = item["country"]
            country_val = co.get("name", co.get("country", "")) if isinstance(co, dict) else str(co)
        rank_val = league.get("rank")
        if rank_val is not None and not isinstance(rank_val, int):
            try:
                rank_val = int(rank_val)
            except (TypeError, ValueError):
                rank_val = None
        defaults = {
            "name": league.get("name", ""),
            "country": country_val or "",
            "logo_url": league.get("logo", "") or "",
            "type": league.get("type", "") or "",
            "rank": rank_val,
        }
        obj, was_created = Competition.objects.update_or_create(
            api_id=api_id, defaults=defaults
        )
        if was_created:
            created += 1
        else:
            updated += 1
    return created, updated


def sync_teams_for_league(league_id, season):
    """Fetch teams for a league+season and create/update Team records. Returns (created, updated)."""
    league_id = getattr(league_id, "api_id", league_id)
    raw = get_teams(league_id, season)
    created = updated = 0
    for item in raw:
        if not item:
            continue
        api_id = item.get("team", {}).get("id") if isinstance(item.get("team"), dict) else item.get("id")
        if not api_id:
            continue
        team_data = item.get("team", item) if isinstance(item.get("team"), dict) else item
        defaults = {
            "name": team_data.get("name", ""),
            "code": str(team_data.get("code", "") or "")[:20],
            "country": team_data.get("country", "") or "",
            "logo_url": team_data.get("logo", "") or "",
            "founded": team_data.get("founded") or None,
            "venue_name": team_data.get("venue", {}).get("name", "") if isinstance(team_data.get("venue"), dict) else (team_data.get("venue_name") or ""),
        }
        obj, was_created = Team.objects.update_or_create(api_id=api_id, defaults=defaults)
        if was_created:
            created += 1
        else:
            updated += 1
    return created, updated


def sync_fixtures(league_id, season, date_str=None, next_n=None):
    """Fetch fixtures and create/update Game records. Returns (created, updated)."""
    league_id = getattr(league_id, "api_id", league_id) if league_id else None
    raw = get_fixtures(league_id=league_id, season=season, date_str=date_str, next_n=next_n)
    created = updated = 0
    comp_cache = {}
    for item in raw:
        if not item:
            continue
        api_id = item.get("fixture", {}).get("id") if isinstance(item.get("fixture"), dict) else item.get("id")
        if not api_id:
            continue
        fixture = item.get("fixture", item) if isinstance(item.get("fixture"), dict) else item
        league_data = item.get("league", {})
        teams_data = item.get("teams", {})
        goals_data = item.get("goals", {})

        comp_id = league_data.get("id") if league_data else None
        competition = None
        if comp_id:
            if comp_id not in comp_cache:
                comp_cache[comp_id] = Competition.objects.filter(api_id=comp_id).first()
            competition = comp_cache[comp_id]

        season_val = league_data.get("season") if league_data else season
        home_data = teams_data.get("home", {}) if isinstance(teams_data, dict) else {}
        away_data = teams_data.get("away", {}) if isinstance(teams_data, dict) else {}
        home_id = home_data.get("id") if isinstance(home_data, dict) else None
        away_id = away_data.get("id") if isinstance(away_data, dict) else None
        if not home_id or not away_id:
            continue
        home_team = Team.objects.filter(api_id=home_id).first()
        away_team = Team.objects.filter(api_id=away_id).first()
        if not home_team or not away_team:
            continue

        status_short = fixture.get("status", {}).get("short", "NS") if isinstance(fixture.get("status"), dict) else fixture.get("status", "NS")
        if status_short not in [c for c, _ in Game.Status.choices]:
            status_short = Game.Status.NS

        kickoff = _parse_datetime(fixture.get("date"))

        home_goals = goals_data.get("home") if isinstance(goals_data, dict) else None
        away_goals = goals_data.get("away") if isinstance(goals_data, dict) else None
        if home_goals is not None:
            try:
                home_goals = int(home_goals)
            except (TypeError, ValueError):
                home_goals = None
        if away_goals is not None:
            try:
                away_goals = int(away_goals)
            except (TypeError, ValueError):
                away_goals = None

        venue = fixture.get("venue", {}) if isinstance(fixture.get("venue"), dict) else {}
        venue_name = venue.get("name", "") if venue else (fixture.get("venue_name") or "")
        venue_name = (venue_name or "")[:200]
        referee = (item.get("referee") or "")[:200]

        defaults = {
            "competition": competition,
            "season": season_val,
            "round_label": (league_data.get("round") or "")[:100],
            "home_team": home_team,
            "away_team": away_team,
            "kickoff": kickoff,
            "status": status_short,
            "home_goals": home_goals,
            "away_goals": away_goals,
            "venue_name": venue_name,
            "referee": referee,
        }
        try:
            obj, was_created = Game.objects.update_or_create(
                api_id=api_id, defaults=defaults
            )
            if was_created:
                created += 1
            else:
                updated += 1
        except Exception:
            continue
    return created, updated


def sync_fixture_statistics(fixture_api_id):
    """Fetch statistics for one fixture and save as GameStatistics. Returns (created_count, updated_count)."""
    raw = get_fixture_statistics(fixture_api_id)
    game = Game.objects.filter(api_id=fixture_api_id).first()
    if not game:
        raise ValueError(f"Game with api_id={fixture_api_id} not found. Sync fixtures first.")
    created = updated = 0
    for item in raw:
        if not item:
            continue
        team_data = item.get("team", {}) if isinstance(item.get("team"), dict) else {}
        team_api_id = team_data.get("id") if team_data else item.get("team_id")
        if not team_api_id:
            continue
        team = Team.objects.filter(api_id=team_api_id).first()
        if not team:
            continue
        stats_list = item.get("statistics")
        if not isinstance(stats_list, list):
            stats_list = []
        stats_list = [
            {"type": str(s.get("type", "")), "value": str(s.get("value", "")) if s.get("value") is not None else ""}
            for s in stats_list if isinstance(s, dict)
        ]
        _, was_created = GameStatistics.objects.update_or_create(
            game=game, team=team, defaults={"statistics": stats_list}
        )
        if was_created:
            created += 1
        else:
            updated += 1
    return created, updated


def sync_stats_for_games(queryset=None, limit=None):
    """
    Fetch statistics for games that don't have any GameStatistics yet.
    queryset: Game queryset (default: finished games without statistics).
    limit: max requests (1 request per game). Returns (synced_count, skipped).
    """
    if queryset is None:
        queryset = (
            Game.objects.filter(
                status__in=(Game.Status.FT, Game.Status.AET, Game.Status.AWD, Game.Status.WO)
            )
            .annotate(num_stats=Count("statistics_rows"))
            .filter(num_stats=0)
        )
    synced = skipped = 0
    games = queryset.order_by("-kickoff")[:limit] if limit else queryset.order_by("-kickoff")
    for game in games:
        if remaining_requests_today() <= 0:
            skipped += 1
            continue
        try:
            sync_fixture_statistics(game.api_id)
            synced += 1
        except Exception:
            skipped += 1
    return synced, skipped


def sync_fixture_predictions(fixture_api_id):
    raw_list = get_fixture_predictions(fixture_api_id)
    game = Game.objects.filter(api_id=fixture_api_id).first()
    if not game:
        raise ValueError(f"Game with api_id={fixture_api_id} not found.")
    if not raw_list or not isinstance(raw_list, list):
        return 0, 0
    data = raw_list[0] if raw_list else {}
    if not isinstance(data, dict):
        return 0, 0
    _, was_created = GamePrediction.objects.update_or_create(game=game, defaults={"raw": data})
    return (1, 0) if was_created else (0, 1)


def sync_fixture_odds(fixture_api_id):
    game = Game.objects.filter(api_id=fixture_api_id).first()
    if not game:
        raise ValueError(f"Game with api_id={fixture_api_id} not found.")
    raw_odds = get_fixture_odds(fixture_api_id)
    if not raw_odds:
        return 0, 0
    if isinstance(raw_odds, dict) and "bookmakers" in raw_odds:
        raw_odds = [raw_odds]
    elif not isinstance(raw_odds, list):
        raw_odds = []
    if not raw_odds:
        return 0, 0
    pred, _ = GamePrediction.objects.get_or_create(game=game, defaults={"raw": {}})
    pred.odds = raw_odds
    pred.save(update_fields=["odds"])
    return 0, 1


def sync_predictions_for_games(queryset=None, limit=None):
    if queryset is None:
        queryset = Game.objects.filter(prediction__isnull=True)
    created = updated = skipped = 0
    for game in (queryset.order_by("-kickoff")[:limit] if limit else queryset.order_by("-kickoff")):
        if remaining_requests_today() <= 0:
            skipped += 1
            continue
        try:
            c, u = sync_fixture_predictions(game.api_id)
            created += c
            updated += u
        except Exception:
            skipped += 1
    return created, updated, skipped


def sync_odds_for_games(queryset=None, limit=None):
    if queryset is None:
        queryset = Game.objects.filter(prediction__isnull=False)
    updated = skipped = 0
    for game in (queryset.order_by("-kickoff")[:limit] if limit else queryset.order_by("-kickoff")):
        if remaining_requests_today() <= 0:
            skipped += 1
            continue
        try:
            _, u = sync_fixture_odds(game.api_id)
            updated += u
        except Exception:
            skipped += 1
    return 0, updated, skipped
