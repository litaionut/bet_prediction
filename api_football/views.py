from io import StringIO
from django.http import JsonResponse
from django.shortcuts import render, redirect, get_object_or_404
from django.urls import reverse
from django.contrib import messages
from django.conf import settings
from django.core.management import call_command
from django.core.paginator import Paginator
from django.db.models import Count
from django.utils import timezone
from django.utils.text import slugify
from datetime import timedelta
from urllib.parse import urlencode
from itertools import groupby

from .models import Country, Competition, Game, GameStatistics, BetJournalEntry

# Optional ML Over/Under 2.5 (Poisson); avoid import errors if deps missing
try:
    from ml_gemini.features import (
        get_game_features,
        get_model_filename_for_competition,
        get_model_filename_for_league,
        get_competition_for_league,
    )
    from ml_gemini.poisson_model import predict_lambda_for_game, predict_lambdas_for_games
    from ml_gemini.poisson_probability import poisson_probabilities
except Exception:
    get_game_features = None
    get_model_filename_for_competition = None
    get_model_filename_for_league = None
    get_competition_for_league = None
    predict_lambda_for_game = None
    predict_lambdas_for_games = None
    poisson_probabilities = None


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
    """List all countries that have competitions (league list only; no bulk download). Download teams/fixtures per league from Sync or from the league page."""
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
    context = {
        "countries_with_data": countries_with_data,
        "current_year": timezone.now().year,
    }
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


def competition_delete(request, pk):
    """Delete a competition (league). Removes the league and all its fixtures; it will no longer be synced."""
    competition = get_object_or_404(
        Competition.objects.annotate(game_count=Count("games")), pk=pk
    )
    if request.method == "POST":
        country_name = competition.country
        country_slug = slugify(country_name) or (country_name.lower().replace(" ", "-") if country_name else None)
        name = competition.name
        competition.delete()
        messages.success(request, f"Deleted league «{name}» and all its fixtures. It will not be synced anymore.")
        if country_slug:
            return redirect("api_football:country_detail", slug=country_slug)
        return redirect("api_football:competition_list")

    return render(
        request,
        "api_football/competition_confirm_delete.html",
        {"competition": competition},
    )


def competition_bulk_delete(request):
    """Delete multiple competitions (leagues) selected by checkbox. POST only."""
    if request.method != "POST":
        return redirect("api_football:competition_list")

    raw_ids = request.POST.getlist("competition_ids")
    ids = []
    for i in raw_ids:
        try:
            ids.append(int(i))
        except (TypeError, ValueError):
            continue

    if not ids:
        messages.warning(request, "No leagues selected.")
        country_slug = (request.POST.get("country_slug") or "").strip()
        if country_slug:
            return redirect("api_football:country_detail", slug=country_slug)
        return redirect("api_football:competition_list")

    to_delete = Competition.objects.filter(pk__in=ids)
    count = to_delete.count()
    to_delete.delete()
    messages.success(
        request,
        f"Deleted {count} league{'s' if count != 1 else ''} and all their fixtures. They will not be synced anymore.",
    )
    country_slug = (request.POST.get("country_slug") or "").strip()
    if country_slug:
        return redirect("api_football:country_detail", slug=country_slug)
    return redirect("api_football:competition_list")


def _run_ml_command(request, competition, step_name, run_build, run_train):
    """Run build_gemini_dataset and/or train_gemini_poisson for this competition. Return (success, message)."""
    api_id = getattr(competition, "api_id", None)
    if api_id is None:
        return False, "Competition has no api_id."
    dataset_name = f"gemini_dataset_{api_id}.csv"
    model_name = f"gemini_poisson_{api_id}.json"
    out = StringIO()
    err = StringIO()
    build_msg = ""
    train_msg = ""
    try:
        if run_build:
            call_command(
                "build_gemini_dataset",
                competition=competition.pk,
                output=dataset_name,
                stdout=out,
                stderr=err,
            )
            out.seek(0)
            build_msg = (out.read() or "").strip() or "Dataset built."
            if err.getvalue():
                build_msg += " " + err.getvalue().strip()
            out.truncate(0)
            out.seek(0)
            err.truncate(0)
            err.seek(0)
        if run_train:
            call_command(
                "train_gemini_poisson",
                dataset=dataset_name,
                output=model_name,
                stdout=out,
                stderr=err,
            )
            out.seek(0)
            train_msg = (out.read() or "").strip() or "Model saved."
            if err.getvalue():
                train_msg += " " + err.getvalue().strip()
        if run_build and run_train:
            return True, (build_msg + " " + train_msg).strip()
        return True, (build_msg if run_build else train_msg).strip()
    except Exception as e:
        return False, str(e)


def competition_detail(request, pk):
    """Show one league: seasons and fixtures (upcoming + past). POST action=download_league: download teams + fixtures. POST action=build_dataset/train_model/build_and_train: run ML training."""
    competition = get_object_or_404(Competition, pk=pk)
    if request.method == "POST":
        action = request.POST.get("action")
        if action == "download_league":
            from .sync import sync_teams_for_league, sync_fixtures
            years_str = (request.POST.get("years") or request.POST.get("year") or "").strip() or str(timezone.now().year)
            years = _parse_years(years_str)
            if not years:
                messages.warning(request, "Enter at least one valid year (e.g. 2024).")
                return redirect("api_football:competition_detail", pk=pk)
            total_tc, total_tu, total_fc, total_fu = 0, 0, 0, 0
            for year in years:
                try:
                    tc, tu = sync_teams_for_league(competition.api_id, year)
                    total_tc += tc
                    total_tu += tu
                    fc, fu = sync_fixtures(league_id=competition.api_id, season=year)
                    total_fc += fc
                    total_fu += fu
                except Exception as e:
                    messages.warning(request, f"{competition.name} {year}: {e}")
            messages.success(
                request,
                f"Downloaded {competition.name}: teams {total_tc} created, {total_tu} updated; fixtures {total_fc} created, {total_fu} updated.",
            )
            return redirect("api_football:competition_detail", pk=pk)
        if action in ("build_dataset", "train_model", "build_and_train"):
            run_build = action in ("build_dataset", "build_and_train")
            run_train = action in ("train_model", "build_and_train")
            ok, msg = _run_ml_command(request, competition, action, run_build=run_build, run_train=run_train)
            if ok:
                messages.success(request, msg)
            else:
                messages.error(request, msg)
            return redirect("api_football:competition_detail", pk=pk)

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

    # Pagination: only load one page of each (faster)
    per_page = 25
    upcoming_qs = base_qs.exclude(
        status__in=(Game.Status.FT, Game.Status.AET, Game.Status.AWD, Game.Status.WO, Game.Status.CANC)
    ).order_by("kickoff")
    past_qs = base_qs.filter(
        status__in=(Game.Status.FT, Game.Status.AET, Game.Status.AWD, Game.Status.WO)
    ).order_by("-kickoff")

    upcoming_paginator = Paginator(upcoming_qs, per_page)
    past_paginator = Paginator(past_qs, per_page)
    upcoming_page_num = request.GET.get("upcoming_page", 1)
    past_page_num = request.GET.get("past_page", 1)
    try:
        upcoming_page = upcoming_paginator.page(int(upcoming_page_num))
    except (ValueError, TypeError):
        upcoming_page = upcoming_paginator.page(1)
    try:
        past_page = past_paginator.page(int(past_page_num))
    except (ValueError, TypeError):
        past_page = past_paginator.page(1)

    upcoming = list(upcoming_page.object_list)
    past = list(past_page.object_list)

    country_slug = slugify(competition.country) or competition.country.lower().replace(" ", "-") if competition.country else ""
    game_ml_odds = {}
    # ML only for games on current page (load model once via predict_lambdas_for_games)
    page_games = upcoming + past
    if page_games and get_model_filename_for_competition and predict_lambdas_for_games and poisson_probabilities:
        models_dir = getattr(settings, "ML_MODELS_DIR", None) or getattr(settings, "BASE_DIR", None)
        if models_dir and competition:
            filename = get_model_filename_for_competition(competition)
            if filename:
                model_path = models_dir / filename
                if model_path.exists():
                    for g, lam in predict_lambdas_for_games(page_games, model_path):
                        p_over = poisson_probabilities(lam).get("prob_over_2_5")
                        if p_over is not None:
                            game_ml_odds[g.pk] = {
                                "ml_over_pct": round(100 * p_over),
                                "ml_under_pct": round(100 * (1 - p_over)),
                            }
    upcoming_with_ml = [(g, game_ml_odds.get(g.pk)) for g in upcoming]
    # For past games: add whether ML prediction matched the result (green=correct, red=wrong)
    past_with_ml = []
    for g in past:
        ml = game_ml_odds.get(g.pk)
        correct = None
        if ml is not None:
            total_goals = (g.home_goals or 0) + (g.away_goals or 0)
            actual_over = total_goals > 2.5
            predicted_over = ml["ml_over_pct"] >= 50
            correct = actual_over == predicted_over
        past_with_ml.append((g, ml, correct))

    # ML accuracy for this league: green (correct) / total past games with prediction
    ml_accuracy = None
    if get_model_filename_for_competition and predict_lambdas_for_games and poisson_probabilities:
        models_dir = getattr(settings, "ML_MODELS_DIR", None) or getattr(settings, "BASE_DIR", None)
        if models_dir and competition:
            filename = get_model_filename_for_competition(competition)
            if filename:
                model_path = models_dir / filename
                if model_path.exists():
                    all_past_qs = base_qs.filter(
                        status__in=(Game.Status.FT, Game.Status.AET, Game.Status.AWD, Game.Status.WO)
                    ).order_by("kickoff")[:2000]
                    all_past_list = list(all_past_qs)
                    if all_past_list:
                        total_with_pred = 0
                        correct_count = 0
                        current_missed = 0
                        max_missed_row = 0
                        for g, lam in predict_lambdas_for_games(all_past_list, model_path):
                            p_over = poisson_probabilities(lam).get("prob_over_2_5")
                            if p_over is not None:
                                total_with_pred += 1
                                total_goals = (g.home_goals or 0) + (g.away_goals or 0)
                                actual_over = total_goals > 2.5
                                predicted_over = p_over >= 0.5
                                correct = actual_over == predicted_over
                                if correct:
                                    correct_count += 1
                                    current_missed = 0
                                else:
                                    current_missed += 1
                                    if current_missed > max_missed_row:
                                        max_missed_row = current_missed
                        if total_with_pred > 0:
                            ml_accuracy = {
                                "correct": correct_count,
                                "total": total_with_pred,
                                "pct": round(100 * correct_count / total_with_pred),
                                "max_missed_row": max_missed_row,
                            }

    context = {
        "competition": competition,
        "country_slug": country_slug,
        "seasons": seasons,
        "season_param": season_param,
        "upcoming": upcoming,
        "past": past,
        "upcoming_with_ml": upcoming_with_ml,
        "past_with_ml": past_with_ml,
        "game_ml_odds": game_ml_odds,
        "ml_accuracy": ml_accuracy,
        "upcoming_page": upcoming_page,
        "past_page": past_page,
        "upcoming_paginator": upcoming_paginator,
        "past_paginator": past_paginator,
    }
    return render(request, "api_football/competition_detail.html", context)


def _finished_game_statuses():
    return (Game.Status.FT, Game.Status.AET, Game.Status.AWD, Game.Status.WO)


def journal_index(request):
    """Journal tab: show recorded entries first, with 'Add new entry' button above."""
    entries = list(
        BetJournalEntry.objects.select_related("game__home_team", "game__away_team", "game__competition")
        .order_by("-created_at")[:200]
    )
    wins = sum(1 for e in entries if e.result == BetJournalEntry.Result.WIN)
    losses = sum(1 for e in entries if e.result == BetJournalEntry.Result.LOSS)
    context = {
        "entries": entries,
        "wins": wins,
        "losses": losses,
        "total": len(entries),
    }
    return render(request, "api_football/journal_index.html", context)


def journal_add(request):
    """Add new entry: step 1 – select country (dropdown). POST redirects to games list for that country."""
    if request.method == "POST":
        country_slug = (request.POST.get("country_slug") or "").strip()
        if country_slug:
            return redirect("api_football:journal_games", slug=country_slug)
        messages.warning(request, "Please select a country.")
    # Countries that have at least one finished game in the last 48 hours
    cutoff = timezone.now() - timedelta(hours=48)
    qs = (
        Competition.objects.exclude(country="")
        .filter(
            games__status__in=_finished_game_statuses(),
            games__kickoff__gte=cutoff,
        )
        .values("country")
        .annotate(finished_count=Count("games", distinct=True))
        .order_by("country")
    )
    countries = [
        {
            "country": row["country"],
            "slug": slugify(row["country"]) or row["country"].lower().replace(" ", "-"),
            "finished_count": row["finished_count"],
        }
        for row in qs
    ]
    context = {"countries": countries}
    return render(request, "api_football/journal_add.html", context)


def cron_update_today_results(request):
    """
    Run update_today_results command. Protected by CRON_SECRET (query param or header).
    For external schedulers (e.g. cron-job.org) to call every hour.
    """
    secret = (request.GET.get("secret") or request.headers.get("X-Cron-Secret") or "").strip()
    expected = (getattr(settings, "CRON_SECRET", None) or "").strip()
    if not expected or secret != expected:
        return JsonResponse({"ok": False, "error": "unauthorized"}, status=403)
    out = StringIO()
    err = StringIO()
    try:
        call_command("update_today_results", stdout=out, stderr=err)
        out.seek(0)
        return JsonResponse({"ok": True, "message": (out.read() or "").strip() or "Done."})
    except Exception as e:
        err.seek(0)
        return JsonResponse(
            {"ok": False, "error": str(e), "stderr": err.getvalue()},
            status=500,
        )


def journal_games(request, slug):
    """List finished games for the selected country that ended in the last 48 hours."""
    country_name = _country_name_from_slug(slug)
    if not country_name:
        return render(
            request,
            "api_football/journal_games.html",
            {"country_name": None, "country_slug": slug, "games_with_recorded": []},
        )
    cutoff = timezone.now() - timedelta(hours=48)
    games = (
        Game.objects.filter(
            competition__country=country_name,
            status__in=_finished_game_statuses(),
            kickoff__gte=cutoff,
        )
        .select_related("home_team", "away_team", "competition")
        .order_by("-kickoff")
    )
    # Mark which games already have a journal entry
    game_ids = [g.pk for g in games]
    recorded_ids = set(
        BetJournalEntry.objects.filter(game_id__in=game_ids).values_list("game_id", flat=True)
    )
    games_with_recorded = [(g, g.pk in recorded_ids) for g in games]
    context = {
        "country_name": country_name,
        "country_slug": slug,
        "games": games,
        "games_with_recorded": games_with_recorded,
    }
    return render(request, "api_football/journal_games.html", context)


def journal_record(request, pk):
    """Record a bet for a finished game: choose Over 2.5 or Under 2.5; save as win or loss from the result."""
    game = get_object_or_404(
        Game.objects.select_related("home_team", "away_team", "competition"), pk=pk
    )
    if not game.is_finished():
        messages.warning(request, "You can only record bets for finished games.")
        return redirect("api_football:journal_index")

    existing = getattr(game, "journal_entry", None)

    if request.method == "POST":
        choice_str = (request.POST.get("choice") or "").strip().lower()
        if choice_str not in (BetJournalEntry.Choice.OVER, BetJournalEntry.Choice.UNDER):
            messages.warning(request, "Select Over 2.5 or Under 2.5.")
            return redirect("api_football:journal_record", pk=pk)
        total_goals = (game.home_goals or 0) + (game.away_goals or 0)
        result = BetJournalEntry.compute_result(total_goals, choice_str)
        if result is None:
            messages.error(request, "Cannot compute result: score missing.")
            return redirect("api_football:journal_record", pk=pk)
        if existing:
            existing.choice = choice_str
            existing.result = result
            existing.save()
            messages.success(request, f"Updated: you bet {existing.get_choice_display()} → {existing.get_result_display()}.")
        else:
            BetJournalEntry.objects.create(game=game, choice=choice_str, result=result)
            messages.success(request, f"Saved: you bet {choice_str} → {result}.")
        return redirect("api_football:journal_index")

    country_slug = ""
    if game.competition and game.competition.country:
        country_slug = slugify(game.competition.country) or game.competition.country.lower().replace(" ", "-")
    context = {
        "game": game,
        "existing": existing,
        "country_slug": country_slug,
    }
    return render(request, "api_football/journal_record.html", context)


def journal_list(request):
    """Redirect to main Journal tab (index shows the list). Kept for backwards compatibility."""
    return redirect("api_football:journal_index")


def game_list_today(request):
    """Fixtures for a given day (default: today) from all downloaded leagues. Supports ?date=YYYY-MM-DD."""
    from datetime import timedelta, datetime

    today = timezone.localdate()
    date_str = (request.GET.get("date") or "").strip()
    if date_str:
        try:
            selected_date = datetime.strptime(date_str, "%Y-%m-%d").date()
        except ValueError:
            selected_date = today
    else:
        selected_date = today

    games_qs = (
        Game.objects.filter(kickoff__date=selected_date)
        .select_related("home_team", "away_team", "competition")
        .order_by("kickoff")
    )
    paginator = Paginator(games_qs, 50)
    page_num = request.GET.get("page", 1)
    try:
        page = paginator.page(int(page_num))
    except (ValueError, TypeError):
        page = paginator.page(1)
    games = list(page.object_list)

    game_ml_odds = {}
    if games and get_model_filename_for_competition and predict_lambdas_for_games and poisson_probabilities:
        models_dir = getattr(settings, "ML_MODELS_DIR", None) or getattr(settings, "BASE_DIR", None)
        if models_dir:
            by_filename = {}
            for g in games:
                if not g.competition:
                    continue
                fn = get_model_filename_for_competition(g.competition)
                if fn:
                    by_filename.setdefault(fn, []).append(g)
            for _filename, group_list in by_filename.items():
                model_path = models_dir / _filename
                if not model_path.exists():
                    continue
                for game, lam in predict_lambdas_for_games(group_list, model_path):
                    p_over = poisson_probabilities(lam).get("prob_over_2_5")
                    if p_over is not None:
                        game_ml_odds[game.pk] = {"ml_over_pct": round(100 * p_over), "ml_under_pct": round(100 * (1 - p_over))}

    games_with_ml = [(g, game_ml_odds.get(g.pk)) for g in games]

    context = {
        "selected_date": selected_date,
        "today": today,
        "prev_date": selected_date - timedelta(days=1),
        "next_date": selected_date + timedelta(days=1),
        "is_today": selected_date == today,
        "games": games,
        "games_with_ml": games_with_ml,
        "game_ml_odds": game_ml_odds,
        "page": page,
        "paginator": paginator,
    }
    return render(request, "api_football/game_list_today.html", context)


def game_statistics(request, pk):
    """Show match statistics (home vs away) for a finished game."""
    game = get_object_or_404(
        Game.objects.select_related("home_team", "away_team", "competition"), pk=pk
    )
    gemini_row = get_game_features(game) if get_game_features else None
    gemini_poisson = None
    if game.competition and get_model_filename_for_competition and predict_lambda_for_game and poisson_probabilities:
        filename = get_model_filename_for_competition(game.competition)
        if filename:
            models_dir = getattr(settings, "ML_MODELS_DIR", None) or getattr(settings, "BASE_DIR", None)
            model_path = models_dir and (models_dir / filename)
            if model_path and model_path.exists():
                lam = predict_lambda_for_game(game, model_path)
                if lam is not None:
                    gemini_poisson = poisson_probabilities(lam)

    def _stats_context(extra=None):
        d = {
            "game": game,
            "stats_available": False,
            "error": extra.get("error", ""),
            "stats_rows": [],
            "gemini_row": gemini_row,
            "gemini_poisson": gemini_poisson,
        }
        d.update(extra or {})
        return d

    if not game.is_finished():
        return render(
            request,
            "api_football/game_statistics.html",
            _stats_context({"error": "Statistics are available only for finished games."}),
        )
    rows = list(game.statistics_rows.select_related("team").all())
    home_row = next((r for r in rows if r.team_id == game.home_team_id), None)
    away_row = next((r for r in rows if r.team_id == game.away_team_id), None)
    if not home_row or not away_row:
        return render(
            request,
            "api_football/game_statistics.html",
            _stats_context({"error": "No statistics for this fixture. Use Sync league data → Sync fixture statistics."}),
        )
    stat_by_type = {}
    for s in home_row.statistics or []:
        t = (s.get("type") or "").strip()
        if t:
            stat_by_type.setdefault(t, {})["home"] = s.get("value") or ""
    for s in away_row.statistics or []:
        t = (s.get("type") or "").strip()
        if t:
            stat_by_type.setdefault(t, {})["away"] = s.get("value") or ""
    stats_rows = [(t, d.get("home", ""), d.get("away", "")) for t, d in sorted(stat_by_type.items())]
    return render(
        request,
        "api_football/game_statistics.html",
        {
            "game": game,
            "stats_available": True,
            "home_team": game.home_team,
            "away_team": game.away_team,
            "stats_rows": stats_rows,
            "gemini_row": gemini_row,
            "gemini_poisson": gemini_poisson,
        },
    )


def gemini_predictions(request):
    """List games with ML prediction: expected goals (lambda) and P(Over 2.5)."""
    from datetime import timedelta

    league = (request.GET.get("league") or "premier").strip().lower() or "premier"
    comp_pk = request.GET.get("competition")
    try:
        comp_pk = int(comp_pk) if comp_pk else None
    except (TypeError, ValueError):
        comp_pk = None

    # All competitions that have at least one game (for "pick by name" dropdown)
    competitions_list = list(
        Competition.objects.filter(games__isnull=False)
        .distinct()
        .order_by("country", "name")
    )

    context = {
        "games_with_prediction": [],
        "competition": None,
        "model_available": False,
        "league": league,
        "competitions_list": competitions_list,
    }

    comp = None
    model_path = None
    models_dir = getattr(settings, "ML_MODELS_DIR", None) or getattr(settings, "BASE_DIR", None)
    if comp_pk and get_model_filename_for_competition and models_dir:
        comp = Competition.objects.filter(pk=comp_pk).first()
        if comp:
            filename = get_model_filename_for_competition(comp)
            if filename:
                model_path = models_dir / filename
    elif get_competition_for_league and get_model_filename_for_league and models_dir:
        comp = get_competition_for_league(league)
        if comp:
            model_path = models_dir / get_model_filename_for_league(league)

    if not comp:
        return render(request, "api_football/gemini_predictions.html", context)
    context["competition"] = comp
    if not model_path or not model_path.exists():
        return render(request, "api_football/gemini_predictions.html", context)

    context["model_available"] = True
    now = timezone.now()
    start = now - timedelta(days=3)
    end = now + timedelta(days=14)
    games = (
        Game.objects.filter(competition=comp, kickoff__gte=start, kickoff__lte=end)
        .select_related("home_team", "away_team")
        .order_by("kickoff")[:80]
    )
    rows = []
    for game in games:
        lam = predict_lambda_for_game(game, model_path) if predict_lambda_for_game else None
        if lam is not None:
            probs = poisson_probabilities(lam) if poisson_probabilities else None
            rows.append({
                "game": game,
                "lambda": lam,
                "prob_over_2_5": probs.get("prob_over_2_5") if probs else None,
            })
    context["games_with_prediction"] = rows
    return render(request, "api_football/gemini_predictions.html", context)


from .client import get_leagues, remaining_requests_today
from .sync import (
    sync_countries,
    sync_teams_for_league,
    sync_fixtures,
    sync_fixture_statistics,
    sync_stats_for_games,
    sync_predictions_for_games,
    sync_odds_for_games,
    league_api_item_to_session_dict,
    ensure_competition_from_session_dict,
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
                # Fetch leagues from API only; do not create DB records yet.
                available = request.session.get("sync_available_leagues") or {}
                for country_name in selected:
                    raw = get_leagues(country=country_name)
                    leagues = []
                    for item in raw:
                        d = league_api_item_to_session_dict(item)
                        if d:
                            leagues.append(d)
                    if leagues:
                        available[country_name] = leagues
                request.session["sync_available_leagues"] = available
                request.session.modified = True
                total = sum(len(available.get(c, [])) for c in selected)
                messages.success(
                    request,
                    f"Loaded {total} league(s) from API. Select which to download and click Download (only selected will be created).",
                )
                qs = urlencode([("country", n) for n in selected], doseq=True)
                return redirect(reverse("api_football:sync") + "?" + qs)
            elif action == "download":
                selected_api_ids = request.POST.getlist("competition")
                years_str = (request.POST.get("years", "").strip() or request.POST.get("year", "").strip())
                if not selected_api_ids or not years_str:
                    messages.warning(request, "Select at least one competition and enter year(s).")
                    return redirect("api_football:sync")
                years = _parse_years(years_str)
                if not years:
                    messages.warning(request, "Enter at least one valid year (e.g. 2024 or 2022,2023,2024 or 2020-2024).")
                    return redirect("api_football:sync")
                selected_api_ids = [int(x) for x in selected_api_ids]
                # Create Competition only for selected leagues (from session).
                available = request.session.get("sync_available_leagues") or {}
                league_by_api_id = {}
                for country_name, leagues in available.items():
                    for d in leagues:
                        league_by_api_id[d["api_id"]] = d
                missing = [aid for aid in selected_api_ids if aid not in league_by_api_id]
                if missing:
                    messages.warning(
                        request,
                        "Some selected leagues were not in session (session may have expired). Load competitions again, then download.",
                    )
                total_teams_c, total_teams_u = 0, 0
                total_fix_c, total_fix_u = 0, 0
                for api_id in selected_api_ids:
                    d = league_by_api_id.get(api_id)
                    if not d:
                        continue
                    ensure_competition_from_session_dict(d)
                    for year in years:
                        tc, tu = sync_teams_for_league(api_id, year)
                        total_teams_c += tc
                        total_teams_u += tu
                        fc, fu = sync_fixtures(league_id=api_id, season=year)
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
        available = request.session.get("sync_available_leagues") or {}
        # Prefer list from session (API); enrich with game_count and pk from DB.
        if available:
            competitions_by_country = []
            for country_name in selected_countries:
                leagues = available.get(country_name, [])
                if not leagues:
                    continue
                # Enrich each league dict with game_count and pk from DB.
                comps = []
                for d in sorted(leagues, key=lambda x: (x.get("rank") is None, x.get("rank") or 999, x.get("api_id") or 0, x.get("name", ""))):
                    obj = Competition.objects.filter(api_id=d["api_id"]).annotate(game_count=Count("games")).first()
                    comps.append({
                        "api_id": d["api_id"],
                        "name": d.get("name", ""),
                        "type": d.get("type", "") or "",
                        "country": d.get("country", "") or country_name,
                        "game_count": obj.game_count if obj else 0,
                        "pk": obj.pk if obj else None,
                        "rank": d.get("rank"),
                    })
                primary = next((c for c in comps if c.get("rank") == 1), comps[0] if comps else None)
                competitions_by_country.append((country_name, comps, primary))
            actual_countries_in_db = []
        else:
            # Fallback: show already-downloaded competitions from DB.
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
