"""
ML Over/Under 2.5: XGBoost Poisson regression.
Predicts total_goals_actual (expected lambda). Uses objective='count:poisson'.
Validation: 80/20 chronological split, Log Loss and Accuracy for Over 2.5.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import log_loss

from ml_gemini.features import POISSON_FEATURE_COLUMNS, POISSON_TARGET_COLUMN
from ml_gemini.poisson_probability import prob_over_2_5


def load_dataset(csv_path):
    """
    Load dataset CSV; return (X, y). Impute missing features with 0.
    Target is clipped to non-negative int for Poisson.
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path)
    for col in POISSON_FEATURE_COLUMNS + [POISSON_TARGET_COLUMN]:
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")

    for col in POISSON_FEATURE_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df[POISSON_TARGET_COLUMN] = pd.to_numeric(df[POISSON_TARGET_COLUMN], errors="coerce")

    df = df.dropna(subset=[POISSON_TARGET_COLUMN])
    df[POISSON_FEATURE_COLUMNS] = df[POISSON_FEATURE_COLUMNS].fillna(0.0)
    df[POISSON_TARGET_COLUMN] = df[POISSON_TARGET_COLUMN].clip(lower=0).astype(int)

    X = df[POISSON_FEATURE_COLUMNS].astype(float)
    y = df[POISSON_TARGET_COLUMN]
    return X, y


def load_dataset_for_validation(csv_path, train_ratio=0.8):
    """
    Load dataset and split chronologically: first train_ratio for train, last (1-train_ratio) for test.
    Returns (X_train, y_train, X_test, y_test, y_test_over25).
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path)
    required = POISSON_FEATURE_COLUMNS + [POISSON_TARGET_COLUMN, "is_over_2_5"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")

    for col in POISSON_FEATURE_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df[POISSON_TARGET_COLUMN] = pd.to_numeric(df[POISSON_TARGET_COLUMN], errors="coerce")
    df["is_over_2_5"] = pd.to_numeric(df["is_over_2_5"], errors="coerce")

    df = df.dropna(subset=[POISSON_TARGET_COLUMN, "is_over_2_5"])
    df[POISSON_FEATURE_COLUMNS] = df[POISSON_FEATURE_COLUMNS].fillna(0.0)
    df[POISSON_TARGET_COLUMN] = df[POISSON_TARGET_COLUMN].clip(lower=0).astype(int)
    df["is_over_2_5"] = df["is_over_2_5"].astype(int)

    n = len(df)
    split_idx = int(n * train_ratio)
    if split_idx == 0 or split_idx >= n:
        raise ValueError("Dataset too small for 80/20 split.")

    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]

    X_train = train_df[POISSON_FEATURE_COLUMNS].astype(float)
    y_train = train_df[POISSON_TARGET_COLUMN]
    X_test = test_df[POISSON_FEATURE_COLUMNS].astype(float)
    y_test = test_df[POISSON_TARGET_COLUMN]
    y_test_over25 = np.array(test_df["is_over_2_5"], dtype=int)

    return X_train, y_train, X_test, y_test, y_test_over25


def evaluate_over25(model, X_test, y_test_over25):
    """
    Predict lambda for test set, compute P(Over 2.5), then Log Loss and Accuracy.
    Returns dict with log_loss, accuracy, n_test.
    """
    lambdas = model.predict(X_test)
    probs = np.array([prob_over_2_5(lam) for lam in lambdas], dtype=float)
    eps = 1e-15
    probs = np.clip(probs, eps, 1.0 - eps)

    ll = log_loss(y_test_over25, probs)
    pred_class = (probs >= 0.5).astype(int)
    acc = (pred_class == y_test_over25).mean()
    return {"log_loss": ll, "accuracy": acc, "n_test": len(y_test_over25)}


def train_poisson_model(X, y, **kwargs):
    """
    Train XGBoost with objective='count:poisson' to predict total goals (lambda).
    """
    default_params = {
        "objective": "count:poisson",
        "random_state": 42,
        "n_estimators": kwargs.pop("n_estimators", 200),
        "max_depth": kwargs.pop("max_depth", 5),
        "learning_rate": kwargs.pop("learning_rate", 0.05),
    }
    default_params.update(kwargs)
    model = xgb.XGBRegressor(**default_params)
    model.fit(X, y)
    return model


def save_model(model, path):
    """Save trained model to path (XGBoost native format)."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    model.save_model(path)


def load_model(path):
    """Load XGBoost Poisson model from path."""
    m = xgb.XGBRegressor(objective="count:poisson")
    m.load_model(path)
    return m


def predict_lambda_for_game(game, model_path):
    """
    Get feature row for game (raw), predict expected goals (lambda). Returns float or None.
    """
    from ml_gemini.features import _get_game_features_raw, POISSON_FEATURE_COLUMNS

    path = Path(model_path)
    if not path.exists():
        return None
    row_raw = _get_game_features_raw(game, for_prediction=True, league_agnostic=True)
    if row_raw is None:
        return None
    X = pd.DataFrame([{
        col: row_raw.get(col, 0.0) or 0.0
        for col in POISSON_FEATURE_COLUMNS
    }], columns=POISSON_FEATURE_COLUMNS)
    model = load_model(path)
    pred = model.predict(X)
    return float(pred[0])


def predict_lambdas_for_games(games, model_path):
    """
    Load model once and predict lambda for each game. Returns list of (game, lambda) for games
    where features could be computed; skips others.
    """
    from ml_gemini.features import _get_game_features_raw, POISSON_FEATURE_COLUMNS

    path = Path(model_path)
    if not path.exists():
        return []
    model = load_model(path)
    rows = []
    for game in games:
        row_raw = _get_game_features_raw(game, for_prediction=True, league_agnostic=True)
        if row_raw is None:
            continue
        X = pd.DataFrame([{
            col: row_raw.get(col, 0.0) or 0.0
            for col in POISSON_FEATURE_COLUMNS
        }], columns=POISSON_FEATURE_COLUMNS)
        pred = model.predict(X)
        rows.append((game, float(pred[0])))
    return rows
