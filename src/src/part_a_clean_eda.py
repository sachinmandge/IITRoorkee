import pandas as pd
import numpy as np

from src.shipment_prep import (
    clean_shipments,
    build_features
)

raw = pd.read_csv("data/raw_shipments.csv")
hubs = pd.read_csv("data/hubs.csv")

clean_df, medians = clean_shipments(raw)

model_df = build_features(clean_df, hubs)

model_df["is_delayed"] = (
    model_df["delivery_hours"] >
    model_df["promised_hours"]
).astype(int)

model_df.to_csv(
    "data/modelling_table.csv",
    index=False
)

print("Rows:", len(model_df))
print("Delayed:", model_df["is_delayed"].sum())
print("Delay Rate:", round(model_df["is_delayed"].mean(), 4))
