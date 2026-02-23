from pathlib import Path
from django.conf import settings
from django.core.management.base import BaseCommand
from ml_gemini.features import POISSON_FEATURE_COLUMNS, POISSON_TARGET_COLUMN
from ml_gemini.poisson_model import load_dataset, save_model, train_poisson_model
from ml_gemini.poisson_probability import poisson_probabilities

class Command(BaseCommand):
    help = "Train XGBoost Poisson on dataset. Predicts total goals (lambda) for Over/Under 2.5."

    def add_arguments(self, parser):
        parser.add_argument("-d", "--dataset", default="gemini_dataset.csv")
        parser.add_argument("-o", "--output", default="gemini_poisson.json")
        parser.add_argument("--n-estimators", type=int, default=200)
        parser.add_argument("--max-depth", type=int, default=5)
        parser.add_argument("--learning-rate", type=float, default=0.05)

    def handle(self, *args, **options):
        dataset_path = options["dataset"]
        output_path = options["output"]
        if not Path(dataset_path).is_absolute():
            dataset_path = str(settings.BASE_DIR / dataset_path)
        if not Path(output_path).is_absolute():
            output_path = str(settings.ML_MODELS_DIR / output_path)
        self.stdout.write("Loading dataset...")
        try:
            X, y = load_dataset(dataset_path)
        except FileNotFoundError as e:
            self.stdout.write(self.style.ERROR(str(e)))
            return
        except ValueError as e:
            self.stdout.write(self.style.ERROR(str(e)))
            return
        n = len(y)
        if n == 0:
            self.stdout.write(self.style.WARNING("No rows."))
            return
        self.stdout.write("Training on %s samples" % n)
        model = train_poisson_model(X, y, n_estimators=options["n_estimators"], max_depth=options["max_depth"], learning_rate=options["learning_rate"])
        save_model(model, output_path)
        self.stdout.write(self.style.SUCCESS("Model saved to %s" % output_path))
        probs = poisson_probabilities(float(model.predict(X).mean()))
        self.stdout.write("P(Over 2.5) at mean lambda: %.4f" % probs["prob_over_2_5"])
