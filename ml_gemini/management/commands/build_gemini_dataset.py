# Build ML Over/Under 2.5 dataset - see Bet_simulator for full implementation
from pathlib import Path
from django.conf import settings
from django.core.management.base import BaseCommand
from api_football.models import Competition
from ml_gemini.features import LEAGUE_REGISTRY, get_league_dataset, build_dataset_rows


class Command(BaseCommand):
    help = "Build dataset for a league (-l) or competition (-c). Writes CSV."

    def add_arguments(self, parser):
        parser.add_argument("-l", "--league", default=None, help="League slug: premier, laliga, ligue1.")
        parser.add_argument("-c", "--competition", type=int, default=None, metavar="ID", help="Competition pk or api_id.")
        parser.add_argument("-o", "--output", default=None, help="Output CSV path.")
        parser.add_argument(
            "--limit",
            type=int,
            default=None,
            metavar="N",
            help="Use only the last N finished games (default: all).",
        )

    def handle(self, *args, **options):
        league = options.get("league") or "premier"
        comp_id_or_api = options.get("competition")
        output = options.get("output")
        comp = None

        if comp_id_or_api is not None:
            comp = Competition.objects.filter(pk=comp_id_or_api).first() or Competition.objects.filter(api_id=comp_id_or_api).first()
            if not comp:
                self.stdout.write(self.style.ERROR("Competition not found."))
                return
            output = output or "gemini_dataset_%s.csv" % comp.api_id
        else:
            output = output or "gemini_dataset_%s.csv" % league

        output_path = Path(output)
        if not output_path.is_absolute():
            output_path = settings.BASE_DIR / output_path

        limit = options.get("limit")
        if limit:
            self.stdout.write("Using last %d finished games (--limit)." % limit)
        if comp_id_or_api is not None:
            rows = list(build_dataset_rows(comp.id, output_path=str(output_path), limit=limit))
        else:
            rows, comp = get_league_dataset(league, output_path=str(output_path), limit=limit)
            if not comp:
                self.stdout.write(self.style.WARNING("League not found. Use -c <competition_id>."))
                return

        if not rows:
            self.stdout.write(self.style.WARNING("No finished games."))
            return
        self.stdout.write(self.style.SUCCESS("Wrote %s rows to %s" % (len(rows), output_path)))
        if comp:
            self.stdout.write("Competition: %s (id=%s, api_id=%s)" % (comp.name, comp.pk, comp.api_id))
