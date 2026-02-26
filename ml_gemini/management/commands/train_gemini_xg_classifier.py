from pathlib import Path

import numpy as np
from django.conf import settings
from django.core.management.base import BaseCommand

from ml_gemini.xg_classifier import (
    evaluate_classifier,
    load_dataset_split,
    save_model,
    train_classifier,
    XG_INPUT_COLUMNS,
)


class Command(BaseCommand):
    help = "Train logistic/gradient classifier on xG-style features for Over/Under 2.5."

    def add_arguments(self, parser):
        parser.add_argument("-d", "--dataset", default="gemini_dataset.csv")
        parser.add_argument("-o", "--output", default="gemini_xg.pkl")
        parser.add_argument("--train-ratio", type=float, default=0.85)
        parser.add_argument("--solver", default="lbfgs")
        parser.add_argument("--max-iter", type=int, default=500)
        parser.add_argument(
            "--penalty",
            default="l2",
            choices=["l1", "l2", "elasticnet", "none"],
            help="Logistic regression penalty",
        )

    def handle(self, *args, **options):
        dataset_path = options["dataset"]
        output_path = options["output"]
        train_ratio = max(0.6, min(0.95, options["train_ratio"]))

        if not Path(dataset_path).is_absolute():
            dataset_path = str(settings.BASE_DIR / dataset_path)
        if not Path(output_path).is_absolute():
            output_path = str(settings.ML_MODELS_DIR / output_path)

        self.stdout.write("Loading dataset...")
        try:
            X_train, y_train, X_test, y_test = load_dataset_split(
                dataset_path,
                train_ratio=train_ratio,
                feature_columns=XG_INPUT_COLUMNS,
            )
        except Exception as exc:
            self.stdout.write(self.style.ERROR(str(exc)))
            return

        if X_train.empty:
            self.stdout.write(self.style.WARNING("Dataset contains no rows"))
            return

        self.stdout.write(f"Training on {len(X_train)} samples (test {len(X_test)})")
        # sklearn 1.8+: use l1_ratio/C instead of penalty to avoid deprecation
        penalty = options["penalty"]
        if penalty == "none":
            l1_ratio = None
            reg_C = np.inf
        elif penalty == "l1":
            l1_ratio = 1.0
            reg_C = 1.0
        elif penalty == "elasticnet":
            l1_ratio = 0.5
            reg_C = 1.0
        else:
            l1_ratio = 0.0  # l2
            reg_C = 1.0
        model = train_classifier(
            X_train,
            y_train,
            solver=options["solver"],
            max_iter=options["max_iter"],
            C=reg_C,
            l1_ratio=l1_ratio,
        )

        if not X_test.empty:
            metrics = evaluate_classifier(model, X_test, y_test)
            self.stdout.write(
                "Test accuracy: {acc:.3f}, log loss: {ll:.3f} ({n} samples)".format(
                    acc=metrics["accuracy"], ll=metrics["log_loss"], n=metrics["n_test"]
                )
            )

        save_model(model, output_path)
        self.stdout.write(self.style.SUCCESS(f"Classifier saved to {output_path}"))
