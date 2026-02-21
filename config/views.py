"""
Simple health check view to verify database connectivity.
"""
from django.http import JsonResponse
from django.db import connection


def health_check(request):
    """Return 200 if app and database are OK. Useful for Railway and load balancers."""
    try:
        with connection.cursor() as cursor:
            cursor.execute("SELECT 1")
        return JsonResponse({
            "status": "ok",
            "database": "connected",
        })
    except Exception as e:
        return JsonResponse(
            {"status": "error", "database": "disconnected", "detail": str(e)},
            status=503,
        )
