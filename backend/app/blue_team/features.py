"""Vectorized feature engineering for the Blue Team detector (CLAUDE.md §6, PRD §7.3).

Combines Day 1-2 clean transactions with Day 3 injected transactions into one
dataset, then engineers velocity, behavioral, and novelty features. All
rolling-window features use shift-then-rolling (strictly prior transactions
only, never the current row) — approved to prevent self-referential leakage.
No row-wise Python loops over the transaction DataFrame.
"""

from typing import Dict, List

import numpy as np
import pandas as pd

ROLLING_WINDOWS: Dict[str, str] = {"5m": "5min", "1h": "1h", "24h": "24h", "7d": "7d"}


def combine_clean_and_injected(
    clean_transactions: pd.DataFrame, injected_transactions: pd.DataFrame
) -> pd.DataFrame:
    """Pad clean (TransactionBase) rows with is_fraud/attack_family/genome_id
    and concatenate with Day 3's already-InjectedTransaction-shaped rows.
    """
    clean = clean_transactions.copy()
    clean["is_fraud"] = 0
    clean["attack_family"] = None
    clean["genome_id"] = None

    injected = injected_transactions.copy()

    combined = pd.concat([clean, injected], ignore_index=True)
    return combined.sort_values(["customer_id", "timestamp"]).reset_index(drop=True)


def add_novelty_flags(df: pd.DataFrame, customers: pd.DataFrame) -> pd.DataFrame:
    """is_new_device / is_new_beneficiary / is_new_location via vectorized
    membership merges against each customer's known devices/beneficiaries/
    base_location — never a per-row Python loop.
    """
    df = df.copy()

    known_devices = (
        customers[["customer_id", "primary_devices"]]
        .explode("primary_devices")
        .rename(columns={"primary_devices": "device_id"})
    )
    known_devices["_known_device"] = True
    df = df.merge(known_devices, on=["customer_id", "device_id"], how="left")
    df["is_new_device"] = df["_known_device"].isna().astype(int)
    df = df.drop(columns=["_known_device"])

    known_beneficiaries = (
        customers[["customer_id", "usual_beneficiaries"]]
        .explode("usual_beneficiaries")
        .rename(columns={"usual_beneficiaries": "beneficiary_id"})
    )
    known_beneficiaries["_known_beneficiary"] = True
    df = df.merge(known_beneficiaries, on=["customer_id", "beneficiary_id"], how="left")
    df["is_new_beneficiary"] = df["_known_beneficiary"].isna().astype(int)
    df = df.drop(columns=["_known_beneficiary"])

    base_location_map = customers.set_index("customer_id")["base_location"]
    df["_base_location"] = df["customer_id"].map(base_location_map)
    df["is_new_location"] = (df["location"] != df["_base_location"]).astype(int)
    df = df.drop(columns=["_base_location"])

    return df


def add_rolling_velocity_features(df: pd.DataFrame) -> pd.DataFrame:
    """Rolling transaction counts/sums (5m/1h/24h/7d) per customer, excluding
    the current row: compute the inclusive rolling stat, then subtract the
    current row's own contribution (count -1, sum -amount).
    """
    df = df.sort_values(["customer_id", "timestamp"]).reset_index(drop=True)
    ts_indexed = df.set_index("timestamp")

    for label, offset in ROLLING_WINDOWS.items():
        grouped = ts_indexed.groupby("customer_id")["amount"]
        count_incl = grouped.rolling(offset).count().reset_index(drop=True)
        sum_incl = grouped.rolling(offset).sum().reset_index(drop=True)
        df[f"count_{label}"] = count_incl.to_numpy() - 1
        df[f"sum_{label}"] = sum_incl.to_numpy() - df["amount"].to_numpy()

    return df


def add_amount_deviation_ratio(df: pd.DataFrame) -> pd.DataFrame:
    """amount_deviation_ratio = amount / (avg_amount_7d + 1e-6) (CLAUDE.md §6,
    exact), where avg_amount_7d uses only strictly-prior transactions.
    """
    df = df.copy()
    avg_amount_7d = np.where(df["count_7d"] > 0, df["sum_7d"] / df["count_7d"].replace(0, np.nan), 0.0)
    avg_amount_7d = np.nan_to_num(avg_amount_7d, nan=0.0)
    df["amount_deviation_ratio"] = df["amount"] / (avg_amount_7d + 1e-6)
    return df


def engineer_features(df: pd.DataFrame, customers: pd.DataFrame) -> pd.DataFrame:
    """Orchestrates novelty flags + rolling velocity + behavioral ratio.
    Graph features are added separately (graph_engine.py) after the
    train/test split, since the graph must be built on train-only rows.
    """
    df = add_novelty_flags(df, customers)
    df = add_rolling_velocity_features(df)
    df = add_amount_deviation_ratio(df)
    return df
