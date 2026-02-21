"""
List competitions in the database with their IDs (pk and api_id) for use with build_gemini_dataset -c.
"""
from django.core.management.base import BaseCommand
from django.db.models import Count, Q

from api_football.models import Competition, Game

FINISHED = (Game.Status.FT, Game.Status.AET, Game.Status.AWD, Game.Status.WO)


class Command(BaseCommand):
    help = "List competitions with pk and api_id so you can use -c <id> without guessing."

    def add_arguments(self, parser):
        parser.add_argument(
            "--min-games",
            type=int,
            default=0,
            help="Only show competitions with at least this many finished games (default: 0).",
        )

    def handle(self, *args, **options):
        min_games = options["min_games"]
        qs = Competition.objects.annotate(
            finished_count=Count("games", filter=Q(games__status__in=FINISHED))
        )
        if min_games > 0:
            qs = qs.filter(finished_count__gte=min_games)
        qs = qs.order_by("country", "name")
        if not qs.exists():
            self.stdout.write(self.style.WARNING("No competitions found."))
            return
        self.stdout.write("Use: python manage.py build_gemini_dataset -c <pk> or -c <api_id>")
        self.stdout.write("")
        for c in qs:
            finished = getattr(c, "finished_count", None)
            extra = " ({} finished games)".format(finished) if finished is not None else ""
            self.stdout.write(
                "  pk={:<6} api_id={:<6}  {} ({}){}".format(
                    c.pk, c.api_id, c.name, c.country or "-", extra
                )
            )
