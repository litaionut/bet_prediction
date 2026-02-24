from django.shortcuts import render


def index(request):
    """Dashboard home page."""
    return render(request, "dashboard/index.html")


def betting_calculator(request):
    """Render the betting calculator page."""
    return render(request, "dashboard/betting_calculator.html")
