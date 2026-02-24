"""
ML Over/Under 2.5: feature computation for a single game.
All features use only data before the match kickoff (no data leakage).
"""
import csv
from django.db.models import Q

from api_football.models import Game, GameStatistics, Competition

FINISHED_STATUSES = (Game.Status.FT, Game.Status.AET, Game.Status.AWD, Game.Status.WO)

# Optional: league slug -> (name Q, country) for build_gemini_dataset -l league
LEAGUE_REGISTRY = {
    "premier": (
        Q(name__icontains="Premier League") | Q(name__icontains="English Premier League"),
        "England",
    ),
    "laliga": (
        Q(name__icontains="La Liga") | Q(name__icontains="LaLiga") | Q(name__icontains="Primera"),
        "Spain",
    ),
    "ligue1": (
        Q(name__icontains="Ligue 1") | Q(name__icontains="Ligue 1 Uber Eats"),
        "France",
    ),
}


def _safe_avg(values):
    if not values:
        return None
    return sum(values) / len(values)


def _last_n_home_games(team_id, before_dt, competition_id, n=5):
    """Last N finished home games for team, before given datetime, same competition."""
    return (
        Game.objects.filter(
            home_team_id=team_id,
            competition_id=competition_id,
            kickoff__lt=before_dt,
            status__in=FINISHED_STATUSES,
        )
        .order_by("-kickoff")[:n]
        .select_related("home_team", "away_team")
    )


def _last_n_away_games(team_id, before_dt, competition_id, n=5):
    """Last N finished away games for team, before given datetime, same competition."""
    return (
        Game.objects.filter(
            away_team_id=team_id,
            competition_id=competition_id,
            kickoff__lt=before_dt,
            status__in=FINISHED_STATUSES,
        )
        .order_by("-kickoff")[:n]
        .select_related("home_team", "away_team")
    )


def _last_n_h2h(home_team_id, away_team_id, before_dt, competition_id, n=3):
    """Last N direct meetings (any venue) before given datetime, same competition."""
    return (
        Game.objects.filter(
            Q(home_team_id=home_team_id, away_team_id=away_team_id)
            | Q(home_team_id=away_team_id, away_team_id=home_team_id),
            competition_id=competition_id,
            kickoff__lt=before_dt,
            status__in=FINISHED_STATUSES,
        )
        .order_by("-kickoff")[:n]
    )


def _shots_on_goal_from_stats(statistics_list):
    """Extract shots on goal value from API statistics list."""
    for item in statistics_list or []:
        t = (item.get("type") or "").strip().lower()
        if "shots on goal" in t or "shots on target" in t:
            try:
                return int(item.get("value") or 0)
            except (TypeError, ValueError):
                return None
    return None


def _team_shots_on_goal_for_game(game, team):
    """Shots on goal for the given team in this game (from GameStatistics)."""
    row = (
        GameStatistics.objects.filter(game=game, team=team)
        .values_list("statistics", flat=True)
        .first()
    )
    return _shots_on_goal_from_stats(row) if row is not None else None


def get_game_features(game):
    """
    Compute ML features and target for one finished game.
    Returns a dict with display-friendly keys and values, or None if not applicable.
    """
    if not game.competition_id:
        return None
    if not game.is_finished():
        return None
    kickoff = game.kickoff
    if not kickoff:
        return None

    competition_id = game.competition_id
    home_home_5 = list(_last_n_home_games(game.home_team_id, kickoff, competition_id, 5))
    away_away_5 = list(_last_n_away_games(game.away_team_id, kickoff, competition_id, 5))
    home_home_10 = list(_last_n_home_games(game.home_team_id, kickoff, competition_id, 10))
    away_away_10 = list(_last_n_away_games(game.away_team_id, kickoff, competition_id, 10))

    home_attack_5 = _safe_avg([g.home_goals or 0 for g in home_home_5])
    away_attack_5 = _safe_avg([g.away_goals or 0 for g in away_away_5])
    home_attack_10 = _safe_avg([g.home_goals or 0 for g in home_home_10])
    away_attack_10 = _safe_avg([g.away_goals or 0 for g in away_away_10])
    home_def_5 = _safe_avg([g.away_goals or 0 for g in home_home_5])
    away_def_5 = _safe_avg([g.home_goals or 0 for g in away_away_5])
    home_def_10 = _safe_avg([g.away_goals or 0 for g in home_home_10])
    away_def_10 = _safe_avg([g.home_goals or 0 for g in away_away_10])

    def _shots_avg(games, team):
        vals = []
        for g in games:
            v = _team_shots_on_goal_for_game(g, team)
            if v is not None:
                vals.append(v)
        return _safe_avg(vals)

    home_shots_5 = _shots_avg(home_home_5, game.home_team)
    away_shots_5 = _shots_avg(away_away_5, game.away_team)
    home_shots_10 = _shots_avg(home_home_10, game.home_team)
    away_shots_10 = _shots_avg(away_away_10, game.away_team)
    h2h_games = list(_last_n_h2h(game.home_team_id, game.away_team_id, kickoff, competition_id, 3))
    h2h_avg = _safe_avg([(g.home_goals or 0) + (g.away_goals or 0) for g in h2h_games])

    total_goals = (game.home_goals or 0) + (game.away_goals or 0)
    is_over_2_5 = 1 if total_goals > 2.5 else 0

    def _fmt(v):
        if v is None:
            return "—"
        return round(v, 4)

    return {
        "home_attack_form_5": _fmt(home_attack_5),
        "away_attack_form_5": _fmt(away_attack_5),
        "home_defensive_fragility_5": _fmt(home_def_5),
        "away_defensive_fragility_5": _fmt(away_def_5),
        "home_shots_on_goal_avg_5": _fmt(home_shots_5),
        "away_shots_on_goal_avg_5": _fmt(away_shots_5),
        "home_attack_form_10": _fmt(home_attack_10),
        "away_attack_form_10": _fmt(away_attack_10),
        "home_defensive_fragility_10": _fmt(home_def_10),
        "away_defensive_fragility_10": _fmt(away_def_10),
        "home_shots_on_goal_avg_10": _fmt(home_shots_10),
        "away_shots_on_goal_avg_10": _fmt(away_shots_10),
        "h2h_total_goals_avg_3": _fmt(h2h_avg),
        "total_goals_actual": total_goals,
        "is_over_2_5": is_over_2_5,
    }


POISSON_FEATURE_COLUMNS = [
    "home_attack_form_5", "away_attack_form_5",
    "home_defensive_fragility_5", "away_defensive_fragility_5",
    "home_shots_on_goal_avg_5", "away_shots_on_goal_avg_5",
    "home_attack_form_10", "away_attack_form_10",
    "home_defensive_fragility_10", "away_defensive_fragility_10",
    "home_shots_on_goal_avg_10", "away_shots_on_goal_avg_10",
    "h2h_total_goals_avg_3",
]
POISSON_TARGET_COLUMN = "total_goals_actual"


def _get_game_features_raw(game, for_prediction=False, league_agnostic=True):
    """
    Same as get_game_features but returns raw numeric values (None if missing).
    league_agnostic=True: any competition. If for_prediction=False, only finished games.
    """
    if not game.competition_id or not game.kickoff:
        return None
    if not league_agnostic:
        return None
    if not for_prediction and not game.is_finished():
        return None

    kickoff = game.kickoff
    competition_id = game.competition_id
    home_home_5 = list(_last_n_home_games(game.home_team_id, kickoff, competition_id, 5))
    away_away_5 = list(_last_n_away_games(game.away_team_id, kickoff, competition_id, 5))
    home_home_10 = list(_last_n_home_games(game.home_team_id, kickoff, competition_id, 10))
    away_away_10 = list(_last_n_away_games(game.away_team_id, kickoff, competition_id, 10))

    home_attack_5 = _safe_avg([g.home_goals or 0 for g in home_home_5])
    away_attack_5 = _safe_avg([g.away_goals or 0 for g in away_away_5])
    home_attack_10 = _safe_avg([g.home_goals or 0 for g in home_home_10])
    away_attack_10 = _safe_avg([g.away_goals or 0 for g in away_away_10])
    home_def_5 = _safe_avg([g.away_goals or 0 for g in home_home_5])
    away_def_5 = _safe_avg([g.home_goals or 0 for g in away_away_5])
    home_def_10 = _safe_avg([g.away_goals or 0 for g in home_home_10])
    away_def_10 = _safe_avg([g.home_goals or 0 for g in away_away_10])

    def _shots_avg(games, team):
        vals = []
        for g in games:
            v = _team_shots_on_goal_for_game(g, team)
            if v is not None:
                vals.append(v)
        return _safe_avg(vals)

    home_shots_5 = _shots_avg(home_home_5, game.home_team)
    away_shots_5 = _shots_avg(away_away_5, game.away_team)
    home_shots_10 = _shots_avg(home_home_10, game.home_team)
    away_shots_10 = _shots_avg(away_away_10, game.away_team)
    h2h_games = list(_last_n_h2h(game.home_team_id, game.away_team_id, kickoff, competition_id, 3))
    h2h_avg = _safe_avg([(g.home_goals or 0) + (g.away_goals or 0) for g in h2h_games])

    total_goals = (game.home_goals or 0) + (game.away_goals or 0) if game.is_finished() else None
    is_over_2_5 = (1 if total_goals > 2.5 else 0) if total_goals is not None else None

    def _num(v):
        return round(v, 4) if v is not None else None

    return {
        "home_attack_form_5": _num(home_attack_5),
        "away_attack_form_5": _num(away_attack_5),
        "home_defensive_fragility_5": _num(home_def_5),
        "away_defensive_fragility_5": _num(away_def_5),
        "home_shots_on_goal_avg_5": _num(home_shots_5),
        "away_shots_on_goal_avg_5": _num(away_shots_5),
        "home_attack_form_10": _num(home_attack_10),
        "away_attack_form_10": _num(away_attack_10),
        "home_defensive_fragility_10": _num(home_def_10),
        "away_defensive_fragility_10": _num(away_def_10),
        "home_shots_on_goal_avg_10": _num(home_shots_10),
        "away_shots_on_goal_avg_10": _num(away_shots_10),
        "h2h_total_goals_avg_3": _num(h2h_avg),
        "total_goals_actual": total_goals,
        "is_over_2_5": is_over_2_5,
    }


def get_competition_for_league(slug):
    """Return Competition for league slug ('premier', 'laliga', ...) or None."""
    entry = LEAGUE_REGISTRY.get((slug or "").strip().lower())
    if not entry:
        return None
    name_q, country = entry
    return Competition.objects.filter(name_q).filter(country__icontains=country).first()


def get_league_slug_for_competition(competition):
    """Return league slug if competition is in LEAGUE_REGISTRY, else None."""
    if not competition:
        return None
    for slug, (name_q, country) in LEAGUE_REGISTRY.items():
        if Competition.objects.filter(pk=competition.pk).filter(name_q).filter(country__icontains=country).exists():
            return slug
    return None


def get_model_filename_for_league(slug):
    """Return model filename for league slug."""
    s = (slug or "").strip().lower()
    if s == "premier":
        return "gemini_poisson.json"
    if s in LEAGUE_REGISTRY:
        return f"gemini_poisson_{s}.json"
    return "gemini_poisson.json"


def get_model_filename_for_competition(competition):
    """Return model filename for a competition (slug-based or gemini_poisson_{api_id}.json)."""
    if not competition:
        return None
    slug = get_league_slug_for_competition(competition)
    if slug is not None:
        return get_model_filename_for_league(slug)
    api_id = getattr(competition, "api_id", None)
    if api_id is not None:
        return f"gemini_poisson_{api_id}.json"
    return None


def get_league_dataset(league_slug, output_path=None, limit=None):
    """Build dataset for a league (premier, laliga, ...). Returns (list of rows, competition or None)."""
    comp = get_competition_for_league(league_slug)
    if not comp:
        return [], None
    rows = list(build_dataset_rows(comp.id, output_path=output_path, limit=limit))
    return rows, comp


def build_dataset_rows(competition_id, output_path=None, limit=None):
    """
    Iterate finished games for competition chronologically; yield one row per game
    with POISSON_FEATURE_COLUMNS + target. Optionally write CSV to output_path.
    If limit is set, use only the last `limit` games (by kickoff) to keep build time low for big leagues.
    """
    qs = (
        Game.objects.filter(
            competition_id=competition_id,
            status__in=FINISHED_STATUSES,
            kickoff__isnull=False,
        )
        .select_related("home_team", "away_team", "competition")
    )
    if limit:
        # Last N games by kickoff, then process in chronological order
        games = list(qs.order_by("-kickoff")[:limit])
        games.reverse()
    else:
        games = list(qs.order_by("kickoff"))
    fieldnames = POISSON_FEATURE_COLUMNS + [POISSON_TARGET_COLUMN, "is_over_2_5"]
    csv_file = None
    if output_path:
        csv_file = open(output_path, "w", newline="", encoding="utf-8")
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()

    try:
        for game in games:
            row_raw = _get_game_features_raw(game, league_agnostic=True)
            if row_raw is None:
                continue
            row = {k: (row_raw[k] if row_raw[k] is not None else "") for k in fieldnames}
            if "is_over_2_5" in row_raw:
                row["is_over_2_5"] = row_raw["is_over_2_5"]
            if csv_file:
                writer.writerow(row)
            yield row
    finally:
        if csv_file:
            csv_file.close()
