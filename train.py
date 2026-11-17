"""Entraînement du modèle d'annulation v2 — Escale Atlantique (mission FastIA).

Démarche : notebooks/02_modelisation.ipynb et RESULTATS.md.
Usage : python train.py, base démarrée.
"""

import json
import os
import time
from datetime import date
from pathlib import Path

import joblib
import pandas as pd
import sklearn
from dotenv import load_dotenv
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    make_scorer,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sqlalchemy import create_engine, text

from features import (
    CATEGORICAL_FEATURES,
    LEAKY_COLUMNS,
    NEW_CATEGORICAL,
    NEW_NUMERICAL,
    NUMERICAL_FEATURES,
    RAW_FEATURES,
    TARGET,
    add_features,
)

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"
TEST_MONTHS = 4

# Réglages retenus en tâche 7 (notebooks/essais_hyperparametres.csv)
HYPERPARAMETERS = {
    "learning_rate": 0.02,
    "max_iter": 300,
    "max_depth": 4,
    "min_samples_leaf": 20,
    "l2_regularization": 1,
    "class_weight": "balanced",
}

# Métrique principale : F2, le rappel pèse deux fois plus que la précision
F2 = make_scorer(fbeta_score, beta=2)
CV = TimeSeriesSplit(n_splits=5)

QUERY = text("""
    SELECT h.code AS hotel, r.*
    FROM reservations AS r
    JOIN hotels AS h ON h.hotel_id = r.hotel_id
    ORDER BY r.arrival_date, r.reservation_id
""")

load_dotenv(ROOT / ".env")


def load_data(engine) -> pd.DataFrame:
    """Réservations avec le code de leur hôtel, triées par date d'arrivée."""
    return pd.read_sql(QUERY, engine, parse_dates=["arrival_date", "reservation_status_date"])


def split_train_test(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Découpage temporel : les TEST_MONTHS derniers mois d'arrivée en test."""
    limit = df["arrival_date"].max() - pd.DateOffset(months=TEST_MONTHS)
    return df[df["arrival_date"] <= limit], df[df["arrival_date"] > limit]


def build_pipeline(classifier=None) -> Pipeline:
    """Données brutes -> variables nouvelles -> préparation -> modèle."""
    preprocessing = ColumnTransformer(
        [
            (
                "cat",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="constant", fill_value="Unknown")),
                        (
                            "encode",
                            OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=50),
                        ),
                    ]
                ),
                CATEGORICAL_FEATURES,
            ),
            (
                "num",
                # Mise à l'échelle : utile à la régression logistique, sans effet sur les arbres
                Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]),
                NUMERICAL_FEATURES,
            ),
        ],
        sparse_threshold=0,
    )
    if classifier is None:
        classifier = HistGradientBoostingClassifier(random_state=42, **HYPERPARAMETERS)
    return Pipeline(
        [
            ("features", FunctionTransformer(add_features)),
            ("preprocessing", preprocessing),
            ("classifier", classifier),
        ]
    )


def evaluate(model: Pipeline, X: pd.DataFrame, y: pd.Series) -> dict:
    """Métriques de la classe « annulée » et matrice de confusion, au seuil de 0,5."""
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]
    tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
    return {
        "f2": round(float(fbeta_score(y, y_pred, beta=2)), 4),
        "recall": round(float(recall_score(y, y_pred)), 4),
        "precision": round(float(precision_score(y, y_pred)), 4),
        "f1": round(float(f1_score(y, y_pred)), 4),
        "roc_auc": round(float(roc_auc_score(y, y_proba)), 4),
        "average_precision": round(float(average_precision_score(y, y_proba)), 4),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def cross_validation_metrics(X: pd.DataFrame, y: pd.Series) -> dict:
    """Validation croisée temporelle sur l'entraînement."""
    scores = pd.DataFrame(
        cross_validate(
            build_pipeline(),
            X,
            y,
            cv=CV,
            scoring={"f2": F2, "recall": "recall", "precision": "precision"},
            return_train_score=True,
            n_jobs=-1,
        )
    )
    return {
        "scheme": f"TimeSeriesSplit(n_splits={CV.n_splits})",
        "validation_f2_mean": round(float(scores["test_f2"].mean()), 4),
        "validation_f2_std": round(float(scores["test_f2"].std()), 4),
        "train_f2_mean": round(float(scores["train_f2"].mean()), 4),
        "validation_recall_mean": round(float(scores["test_recall"].mean()), 4),
        "validation_precision_mean": round(float(scores["test_precision"].mean()), 4),
    }


def main() -> None:
    engine = create_engine(os.environ["DATABASE_URL"])
    df = load_data(engine)
    train, test = split_train_test(df)
    print(f"Entraînement : {len(train)} réservations | test : {len(test)} réservations")

    # RAW_FEATURES exclut la cible et LEAKY_COLUMNS
    X_train, y_train = train[RAW_FEATURES], train[TARGET]
    X_test, y_test = test[RAW_FEATURES], test[TARGET]

    cv_metrics = cross_validation_metrics(X_train, y_train)
    print("Validation croisée sur l'entraînement :", cv_metrics)

    model = build_pipeline()
    start = time.perf_counter()
    model.fit(X_train, y_train)
    training_seconds = round(time.perf_counter() - start, 1)

    metrics = evaluate(model, X_test, y_test)
    print("Métriques sur le jeu de test :", metrics)

    model_path = MODEL_DIR / "cancellation_model_v2.joblib"
    joblib.dump(model, model_path)
    info = {
        "model_name": "cancellation_model",
        "version": "2.0.0",
        "trained_on": date.today().isoformat(),
        "scikit_learn_version": sklearn.__version__,
        "features": list(X_train.columns),
        "new_features": NEW_CATEGORICAL + NEW_NUMERICAL,
        "excluded_leaky_columns": LEAKY_COLUMNS,
        "target": TARGET,
        "train_period": [str(train["arrival_date"].min().date()), str(train["arrival_date"].max().date())],
        "test_period": [str(test["arrival_date"].min().date()), str(test["arrival_date"].max().date())],
        "hyperparameters": {
            k: v for k, v in model.get_params().items()
            if k.startswith("classifier__") and isinstance(v, (int, float, str, bool, type(None)))
        },
        "main_metric": "f2",
        "decision_threshold": 0.5,
        "cross_validation": cv_metrics,
        "test_metrics": metrics,
        "training_seconds": training_seconds,
        "model_size_mb": round(model_path.stat().st_size / 1e6, 2),
    }
    (MODEL_DIR / "model_info_v2.json").write_text(
        json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("Modèle et fiche sauvegardés dans", MODEL_DIR)


if __name__ == "__main__":
    main()
