"""Colonnes et variables nouvelles du modèle v2.

`add_features` doit être importable là où le modèle est chargé (train.py, notebooks, API).
"""

import pandas as pd

TARGET = "is_canceled"

# Colonnes inconnues au moment de la réservation
LEAKY_COLUMNS = [
    "reservation_status",       # statut final, équivalent à la cible
    "reservation_status_date",  # date du statut final
    "assigned_room_type",       # attribuée à l'arrivée
    "booking_changes",          # cumul jusqu'au dénouement
]
# Colonnes sans information utile pour une réservation future
USELESS_COLUMNS = [
    "reservation_id",     # identifiant chronologique
    "hotel_id",           # doublon de `hotel`
    "arrival_date_year",  # années futures absentes de l'historique
]

# Colonnes brutes en entrée du pipeline
CATEGORICAL_INPUTS = [
    "hotel",
    "arrival_date_month",
    "meal",
    "country",
    "market_segment",
    "distribution_channel",
    "reserved_room_type",
    "deposit_type",
    "customer_type",
]
NUMERICAL_INPUTS = [
    "lead_time",
    "arrival_date_week_number",
    "arrival_date_day_of_month",
    "stays_in_weekend_nights",
    "stays_in_week_nights",
    "adults",
    "children",
    "babies",
    "is_repeated_guest",
    "previous_cancellations",
    "previous_bookings_not_canceled",
    "agent",
    "company",
    "days_in_waiting_list",
    "adr",
    "required_car_parking_spaces",
    "total_of_special_requests",
]
RAW_FEATURES = CATEGORICAL_INPUTS + NUMERICAL_INPUTS

# Variables créées par add_features
NEW_CATEGORICAL = ["agent_id"]
NEW_NUMERICAL = [
    "total_nights",
    "has_children",
    "has_company",
    "previous_cancel_rate",
]

# agent et company servent seulement à calculer agent_id et has_company
CATEGORICAL_FEATURES = CATEGORICAL_INPUTS + NEW_CATEGORICAL
NUMERICAL_FEATURES = [c for c in NUMERICAL_INPUTS if c not in ("agent", "company")] + NEW_NUMERICAL


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Ajoute les variables nouvelles à partir des seules données brutes."""
    df = df.copy()
    df["total_nights"] = df["stays_in_weekend_nights"] + df["stays_in_week_nights"]
    # to_numeric : accepte None dans une ligne isolée
    children = pd.to_numeric(df["children"]).fillna(0)
    df["has_children"] = ((children + df["babies"]) > 0).astype(int)
    df["has_company"] = df["company"].notna().astype(int)
    history = df["previous_cancellations"] + df["previous_bookings_not_canceled"]
    df["previous_cancel_rate"] = (df["previous_cancellations"] / history).where(history > 0, 0.0)
    # "aucune" : réservation sans agence
    df["agent_id"] = pd.to_numeric(df["agent"]).astype("Int64").astype("string").fillna("aucune").astype(object)
    return df
