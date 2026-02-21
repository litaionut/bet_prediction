"""
API-Football (v3) client with request counting for daily limit.
Base URL: https://v3.football.api-sports.io/
Auth: x-apisports-key header
"""
import requests
from datetime import date
from django.conf import settings
from .models import APIRequestLog

DAILY_REQUEST_LIMIT = getattr(settings, "API_FOOTBALL_DAILY_LIMIT", 7500)
BASE_URL = "https://v3.football.api-sports.io"


def get_api_key():
    return getattr(settings, "API_FOOTBALL_KEY", "") or ""


def get_today_request_count():
    today = date.today()
    log, _ = APIRequestLog.objects.get_or_create(date=today, defaults={"request_count": 0})
    return log.request_count


def increment_request_count():
    today = date.today()
    log, created = APIRequestLog.objects.get_or_create(
        date=today, defaults={"request_count": 0}
    )
    if not created:
        log.request_count += 1
        log.save(update_fields=["request_count", "updated_at"])
    else:
        log.request_count = 1
        log.save(update_fields=["request_count", "updated_at"])
    return log.request_count


def remaining_requests_today():
    return max(0, DAILY_REQUEST_LIMIT - get_today_request_count())


def request(method, endpoint, params=None):
    key = get_api_key()
    if not key:
        raise ValueError("API_FOOTBALL_KEY is not set in settings or environment")
    if remaining_requests_today() <= 0:
        raise ValueError(
            f"Daily API limit reached ({DAILY_REQUEST_LIMIT} requests). Try again tomorrow."
        )
    url = f"{BASE_URL}/{endpoint.lstrip('/')}"
    headers = {"x-apisports-key": key}
    resp = requests.request(
        method, url, headers=headers, params=params or {}, timeout=30
    )
    increment_request_count()
    resp.raise_for_status()
    data = resp.json()
    if data.get("errors") and not data.get("response"):
        raise ValueError("API error: " + str(data.get("errors")))
    return data.get("response", [])


def get_leagues(country=None, season=None, type=None):
    params = {}
    if country:
        params["country"] = country
    if season:
        params["season"] = season
    if type:
        params["type"] = type
    return request("GET", "leagues", params)


def get_teams(league_id, season):
    return request("GET", "teams", {"league": league_id, "season": season})


def get_fixtures(league_id=None, season=None, date_str=None, next_n=None):
    params = {}
    if league_id:
        params["league"] = league_id
    if season:
        params["season"] = season
    if date_str:
        params["date"] = date_str
    if next_n:
        params["next"] = next_n
    return request("GET", "fixtures", params)


def get_countries():
    return request("GET", "countries")


def get_fixture_statistics(fixture_id):
    return request("GET", "fixtures/statistics", {"fixture": fixture_id})


def get_fixture_predictions(fixture_id):
    return request("GET", "predictions", {"fixture": fixture_id})


def get_fixture_odds(fixture_id):
    return request("GET", "odds", {"fixture": fixture_id})
