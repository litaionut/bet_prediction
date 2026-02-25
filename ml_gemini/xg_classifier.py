"""xG-informed Over/Under 2.5 classifier utilities."""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss

from ml_gemini.features import (
    XG_FEATURE_COLUMNS,
    POISSON_FEATURE_COLUMNS,
    _get_game_features_raw,
)

XG_INPUT_COLUMNS = list(dict.fromkeys(XG_FEATURE_COLUMNS + POISSON_FEATURE_COLUMNS))
TARGET_COLUMN = "is_over_2_5"


def load_dataset(csv_path, feature_columns=None):
    """Load dataset CSV for classifier training."""
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path)
    features = feature_columns or XG_INPUT_COLUMNS

    required = set(features + [TARGET_COLUMN])
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing column(s) in dataset: {', '.join(missing)}")

    df = df.copy()
    for col in features:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df[TARGET_COLUMN] = pd.to_numeric(df[TARGET_COLUMN], errors="coerce")

    df = df.dropna(subset=[TARGET_COLUMN])
    df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(int)
    df[features] = df[features].fillna(0.0)

    X = df[features].astype(float)
    y = df[TARGET_COLUMN]
    return X, y


def load_dataset_split(csv_path, train_ratio=0.8, feature_columns=None):
    """Chronological split of dataset into train/test sets."""
    X, y = load_dataset(csv_path, feature_columns=feature_columns)
    n = len(X)
    split_idx = int(n * train_ratio)
    if split_idx <= 0 or split_idx >= n:
        raise ValueError("Dataset too small for requested train/test split.")

    X_train = X.iloc[:split_idx]
    y_train = y.iloc[:split_idx]
    X_test = X.iloc[split_idx:]
    y_test = y.iloc[split_idx:]
    return X_train, y_train, X_test, y_test


def train_classifier(X, y, **kwargs):
    """Train a logistic regression classifier for Over/Under."""
    params = {
        "solver": kwargs.pop("solver", "lbfgs"),
        "max_iter": kwargs.pop("max_iter", 500),
        "class_weight": kwargs.pop("class_weight", "balanced"),
    }
    params.update(kwargs)
    model = LogisticRegression(**params)
    model.fit(X, y)
    return model


def evaluate_classifier(model, X_test, y_test):
    """Return log loss and accuracy for the classifier."""
    probs = model.predict_proba(X_test)[:, 1]
    eps = 1e-15
    probs = np.clip(probs, eps, 1.0 - eps)
    ll = log_loss(y_test, probs)
    acc = accuracy_score(y_test, (probs >= 0.5).astype(int))
    return {"log_loss": ll, "accuracy": acc, "n_test": len(y_test)}


def save_model(model, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_model(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Classifier not found: {path}")
    return joblib.load(path)


def predict_probability_for_game(game, model_path, feature_columns=None):
    """Return P(Over 2.5) for a single game (float or None)."""
    row_raw = _get_game_features_raw(game, for_prediction=True, league_agnostic=True)
    if row_raw is None:
        return None
    features = feature_columns or XG_INPUT_COLUMNS
    X = pd.DataFrame([{col: row_raw.get(col, 0.0) or 0.0 for col in features}], columns=features)
    model = load_model(model_path)
    probs = model.predict_proba(X)
    return float(probs[0, 1])


def predict_probabilities_for_games(games, model_path, feature_columns=None):
    """Batch prediction helper returning list of (game, probability)."""
    path = Path(model_path)
    if not path.exists():
        return []
    model = load_model(path)
    features = feature_columns or XG_INPUT_COLUMNS
    rows = []
    for game in games:
        row_raw = _get_game_features_raw(game, for_prediction=True, league_agnostic=True)
        if row_raw is None:
            continue
        X = pd.DataFrame([{col: row_raw.get(col, 0.0) or 0.0 for col in features}], columns=features)
        prob = model.predict_proba(X)[0, 1]
        rows.append((game, float(prob)))
    return rows
