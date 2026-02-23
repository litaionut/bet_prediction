"""
Build dataset and train Poisson model for every competition with enough finished games.
"""
from pathlib import Path

from django.conf import settings
from django.core.management.base import BaseCommand
from django.db.models import Count, Q

from api_football.models import Competition, Game
from ml_gemini.features import (
    FINISHED_STATUSES,
    build_dataset_rows,
    get_model_filename_for_competition,
)
from ml_gemini.poisson_model import (
    load_dataset,
    train_poisson_model,
    save_model,
)

MIN_FINISHED_GAMES = 30


class Command(BaseCommand):
    help = (
        "Build dataset and train Poisson model for all competitions with at least "
        f"{MIN_FINISHED_GAMES} finished games."
    )

    def add_arguments(self, parser):
        parser.add_argument(
            "--min-games",
            type=int,
            default=MIN_FINISHED_GAMES,
            help=f"Minimum finished games per competition (default: {MIN_FINISHED_GAMES}).",
        )
        parser.add_argument(
            "--competition",
            type=int,
            default=None,
            metavar="API_ID",
            help="Train only this competition (api_id or pk).",
        )
        parser.add_argument("--n-estimators", type=int, default=200)
        parser.add_argument("--max-depth", type=int, default=5)
        parser.add_argument("--learning-rate", type=float, default=0.05)

    def handle(self, *args, **options):
        base = Path(settings.BASE_DIR)
        models_dir = Path(settings.ML_MODELS_DIR)
        min_games = options["min_games"]
        comp_filter = options.get("competition")

        finished = Game.objects.filter(
            status__in=FINISHED_STATUSES,
            kickoff__isnull=False,
        )
        comp_ids_with_count = (
            finished.values("competition_id")
            .annotate(finished_count=Count("id"))
            .filter(finished_count__gte=min_games)
        )
        comp_ids = list(comp_ids_with_count.values_list("competition_id", flat=True))
        if not comp_ids:
            self.stdout.write(
                self.style.WARNING(f"No competition has at least {min_games} finished games.")
            )
            return
        qs = Competition.objects.filter(id__in=comp_ids).order_by("country", "name")
        if comp_filter is not None:
            qs = qs.filter(Q(pk=comp_filter) | Q(api_id=comp_filter))
        competitions = list(qs)
        if not competitions:
            self.stdout.write(
                self.style.WARNING("No matching competition (or none with enough finished games).")
            )
            return

        for comp in competitions:
            self.stdout.write(
                f"\n--- {comp.name} ({comp.country}, api_id={comp.api_id}) ---"
            )
            csv_path = base / f"gemini_dataset_{comp.api_id}.csv"
            rows = list(build_dataset_rows(comp.id, output_path=str(csv_path)))
            if not rows:
                self.stdout.write(self.style.WARNING(f"  No rows built for {comp.name}. Skipped."))
                continue
            self.stdout.write(f"  Dataset: {len(rows)} rows -> {csv_path.name}")

            try:
                X, y = load_dataset(str(csv_path))
            except Exception as e:
                self.stdout.write(self.style.ERROR(f"  Load failed: {e}"))
                continue
            if len(y) == 0:
                self.stdout.write(self.style.WARNING("  No rows after load."))
                continue

            model = train_poisson_model(
                X, y,
                n_estimators=options["n_estimators"],
                max_depth=options["max_depth"],
                learning_rate=options["learning_rate"],
            )
            filename = get_model_filename_for_competition(comp)
            if not filename:
                self.stdout.write(self.style.ERROR("  No model filename. Skipped."))
                continue
            model_path = models_dir / filename
            save_model(model, str(model_path))
            self.stdout.write(self.style.SUCCESS(f"  Model saved: {model_path.name}"))

        self.stdout.write("")
