import numpy as np
import pandas as pd

PROMISED_HOURS = {
    "Express": 36,
    "Standard": 60,
    "Economy": 96
}

FEATURES = [
    "promised_hours",
    "distance_km",
    "parcel_weight_kg",
    "log_declared_value",
    "n_stops",
    "rain_flag",
    "sale_event_flag",
    "is_cod",
    "is_weekend",
    "dispatch_hour",
    "hub_tier",
    "hub_daily_capacity",
    "label_print_batch",
    "scanner_build_no",
    "tape_roll_no",
    "sticker_serial_mod",
    "shelf_slot_no",
]

NOISE_COLUMNS = [
    "label_print_batch",
    "scanner_build_no",
    "tape_roll_no",
    "sticker_serial_mod",
    "shelf_slot_no",
]


def clean_shipments(raw, medians=None):
    """
    Clean raw shipment data and return:
        clean_df, medians
    """

    df = raw.copy()

    # 1. Drop exact duplicates
    df = df.drop_duplicates()

    # 2. Clean service_type
    df["service_type"] = (
        df["service_type"]
        .astype(str)
        .str.strip()
        .str.title()
    )

    # 3. payment_mode
    df["payment_mode"] = (
        df["payment_mode"]
        .fillna("Unknown")
    )

    # 4. declared_value -> declared_value_inr

    df["declared_value_inr"] = (
        df["declared_value"]
        .astype(str)
        .str.replace("₹", "", regex=False)
        .str.replace(",", "", regex=False)
    )

    df.loc[
        df["declared_value_inr"].isin(["nan", "None"]),
        "declared_value_inr"
    ] = np.nan

    df["declared_value_inr"] = pd.to_numeric(
        df["declared_value_inr"],
        errors="coerce"
    )

    df.drop(
        columns=["declared_value"],
        inplace=True
    )

    # 5. dispatch_time to datetime

    df["dispatch_time"] = pd.to_datetime(
        df["dispatch_time"]
    )

    # 6. Sentinel parcel weights

    df.loc[
        df["parcel_weight_kg"] > 100,
        "parcel_weight_kg"
    ] = np.nan

    # 7. Compute medians

    if medians is None:
        medians = {
            "distance_km":
                float(df["distance_km"].median()),
            "parcel_weight_kg":
                float(df["parcel_weight_kg"].median()),
            "declared_value_inr":
                float(df["declared_value_inr"].median())
        }

    for col in medians:
        df[col] = df[col].fillna(
            medians[col]
        )

    return df, medians


def build_features(clean, hubs):
    """
    Attach hub information and create model features.
    """

    df = clean.copy()

    # Left merge

    df = df.merge(
        hubs[
            [
                "hub_id",
                "hub_city",
                "region",
                "hub_tier",
                "hub_daily_capacity"
            ]
        ],
        on="hub_id",
        how="left"
    )

    # promised_hours

    df["promised_hours"] = (
        df["service_type"]
        .map(PROMISED_HOURS)
    ).astype(int)

    # is_cod

    df["is_cod"] = (
        df["payment_mode"] == "COD"
    ).astype(int)

    # weekend

    df["is_weekend"] = (
        df["dispatch_time"]
        .dt.dayofweek >= 5
    ).astype(int)

    # dispatch hour

    df["dispatch_hour"] = (
        df["dispatch_time"]
        .dt.hour
    )

    # log declared value

    df["log_declared_value"] = np.log(
        df["declared_value_inr"]
    )

    return df
