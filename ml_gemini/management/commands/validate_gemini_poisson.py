from pathlib import Path
from django.conf import settings
from django.core.management.base import BaseCommand
from ml_gemini.poisson_model import load_dataset_for_validation, train_poisson_model, evaluate_over25, save_model


class Command(BaseCommand):
    help = "80%% train / 20%% test (chronological), evaluate Over 2.5: Log Loss and Accuracy."

    def add_arguments(self, parser):
        parser.add_argument("-d", "--dataset", default="gemini_dataset.csv")
        parser.add_argument("-o", "--output", default=None)
        parser.add_argument("--train-ratio", type=float, default=0.8)
        parser.add_argument("--n-estimators", type=int, default=200)
        parser.add_argument("--max-depth", type=int, default=5)
        parser.add_argument("--learning-rate", type=float, default=0.05)

    def handle(self, *args, **options):
        dataset_path = options["dataset"]
        if not Path(dataset_path).is_absolute():
            dataset_path = str(settings.BASE_DIR / dataset_path)
        try:
            X_train, y_train, X_test, y_test, y_test_over25 = load_dataset_for_validation(dataset_path, train_ratio=options["train_ratio"])
        except (FileNotFoundError, ValueError) as e:
            self.stdout.write(self.style.ERROR(str(e)))
            return
        self.stdout.write("Train: %s, Test: %s" % (len(y_train), len(y_test_over25)))
        model = train_poisson_model(X_train, y_train, n_estimators=options["n_estimators"], max_depth=options["max_depth"], learning_rate=options["learning_rate"])
        if options["output"]:
            out_path = options["output"]
            if not Path(out_path).is_absolute():
                out_path = str(settings.BASE_DIR / out_path)
            save_model(model, out_path)
            self.stdout.write(self.style.SUCCESS("Model saved to %s" % out_path))
        metrics = evaluate_over25(model, X_test, y_test_over25)
        self.stdout.write(self.style.SUCCESS("Over 2.5 test: Log Loss %.4f, Accuracy %.4f" % (metrics["log_loss"], metrics["accuracy"])))
