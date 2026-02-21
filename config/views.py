"""
Simple health check view to verify database connectivity.
"""
from django.db import connection
from django.http import HttpResponse, JsonResponse


def home(request):
    """Simple home page with links to admin and health check."""
    html = """
    <!DOCTYPE html>
    <html>
    <head><title>Bet Prediction</title></head>
    <body style="font-family: sans-serif; max-width: 40em; margin: 3em auto; padding: 0 1em;">
        <h1>Bet Prediction</h1>
        <p>Django app is running.</p>
        <ul>
            <li><a href="/admin/">Admin</a></li>
            <li><a href="/health/">Health check</a></li>
        </ul>
    </body>
    </html>
    """
    return HttpResponse(html)


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
