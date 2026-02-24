"""
Update game results from API-Football for today and yesterday.
Fetches fixtures for those dates and updates (or creates) Game records with latest scores and status.
Run every hour via Railway Cron or cron endpoint so today's live scores and yesterday's final scores stay current.
"""
from datetime import timedelta

from django.core.management.base import BaseCommand
from django.utils import timezone

from api_football.sync import sync_fixtures


class Command(BaseCommand):
    help = (
        "Fetch fixtures from API-Football for today and yesterday and update Game results (scores, status). "
        "Schedule every 1 hour so today's and yesterday's results stay current."
    )

    def add_arguments(self, parser):
        parser.add_argument(
            "--date",
            default=None,
            help="Single date to update (YYYY-MM-DD). If omitted, syncs both today and yesterday.",
        )
        parser.add_argument(
            "--today-only",
            action="store_true",
            help="Only sync today (default when no --date: sync today + yesterday).",
        )

    def handle(self, *args, **options):
        single_date = options.get("date")
        today_only = options.get("today_only", False)

        if single_date:
            try:
                timezone.datetime.strptime(single_date, "%Y-%m-%d")
            except ValueError:
                self.stdout.write(self.style.ERROR("Invalid --date. Use YYYY-MM-DD."))
                return
            dates_to_sync = [single_date]
        else:
            today = timezone.localdate()
            if today_only:
                dates_to_sync = [today.isoformat()]
            else:
                yesterday = today - timedelta(days=1)
                dates_to_sync = [yesterday.isoformat(), today.isoformat()]

        total_created = total_updated = 0
        for date_str in dates_to_sync:
            self.stdout.write("Updating results for date: %s" % date_str)
            try:
                created, updated = sync_fixtures(
                    league_id=None,
                    season=None,
                    date_str=date_str,
                )
                total_created += created
                total_updated += updated
                self.stdout.write(
                    self.style.SUCCESS(
                        "  %s: %d created, %d updated." % (date_str, created, updated)
                    )
                )
            except ValueError as e:
                self.stdout.write(self.style.ERROR("  %s: %s" % (date_str, e)))
            except Exception as e:
                self.stdout.write(self.style.ERROR("  %s failed: %s" % (date_str, e)))
                raise

        self.stdout.write(
            self.style.SUCCESS("Total: %d created, %d updated." % (total_created, total_updated))
        )
