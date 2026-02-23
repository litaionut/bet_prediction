"""
Update game results for today from API-Football.
Fetches fixtures for the current date and updates (or creates) Game records with latest scores and status.
Run every hour via Railway Cron or system cron to keep today's results up to date.
"""
from django.core.management.base import BaseCommand
from django.utils import timezone

from api_football.sync import sync_fixtures


class Command(BaseCommand):
    help = (
        "Fetch today's fixtures from API-Football and update Game results (scores, status). "
        "Schedule every 1 hour (e.g. Railway Cron) to keep today's results current."
    )

    def add_arguments(self, parser):
        parser.add_argument(
            "--date",
            default=None,
            help="Date to update (YYYY-MM-DD). Default: today in project timezone.",
        )

    def handle(self, *args, **options):
        date_str = options.get("date")
        if date_str:
            # Validate format
            try:
                timezone.datetime.strptime(date_str, "%Y-%m-%d")
            except ValueError:
                self.stdout.write(self.style.ERROR("Invalid --date. Use YYYY-MM-DD."))
                return
        else:
            date_str = timezone.localdate().isoformat()

        self.stdout.write("Updating results for date: %s" % date_str)
        try:
            created, updated = sync_fixtures(
                league_id=None,
                season=None,
                date_str=date_str,
            )
            self.stdout.write(
                self.style.SUCCESS(
                    "Fixtures for %s: %d created, %d updated." % (date_str, created, updated)
                )
            )
        except ValueError as e:
            self.stdout.write(self.style.ERROR(str(e)))
        except Exception as e:
            self.stdout.write(self.style.ERROR("Update failed: %s" % e))
            raise
