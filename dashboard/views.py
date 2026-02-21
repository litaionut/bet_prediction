from django.shortcuts import render


def index(request):
    """Dashboard home page."""
    return render(request, "dashboard/index.html")
