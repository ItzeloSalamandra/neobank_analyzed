"""
base_clean.py
=============
Proyecto: Predicción de Churn - Neobank
Autora: Itzel
Fuente: BigQuery -> quiet-seer-299408.neobank (users, transactions, devices, notifications)

Este script documenta y aplica, desde cero, todas las decisiones de limpieza
que fui tomando a lo largo del análisis. La idea es que cualquiera que lea
este archivo (incluida yo misma en unos meses) entienda el "por qué" detrás
de cada filtro, no solo el "qué".
"""

import pandas as pd
import numpy as np
from google.cloud import bigquery

# ---------------------------------------------------------------------------
# 0. Configuración general
# ---------------------------------------------------------------------------
PROJECT_ID = "quiet-seer-299408"
DATASET = "neobank"

client = bigquery.Client(project=PROJECT_ID)

# Fecha de referencia fija para TODOS los cálculos de edad/antigüedad.
REFERENCE_DATE = pd.Timestamp("2026-07-01")  # TODO: ajustar tras revisar MAX(created_date) real

# ---------------------------------------------------------------------------
# 1. Extracción desde BigQuery
# ---------------------------------------------------------------------------

def load_table(table_name: str) -> pd.DataFrame:
    query = f"SELECT * FROM `{PROJECT_ID}.{DATASET}.{table_name}`"
    return client.query(query).to_dataframe()

users = load_table("users")
transactions = load_table("transactions")
notifications = load_table("notifications")
devices_raw = load_table("devices")  # pendiente de mapeo de columnas

# ---------------------------------------------------------------------------
# 2. Limpieza de `users`
# ---------------------------------------------------------------------------

def clean_users(df: pd.DataFrame, reference_date: pd.Timestamp):
    df = df.copy()
    df["age"] = reference_date.year - df["birth_year"]

    MIN_AGE, MAX_AGE = 18, 122
    n_before = len(df)
    age_mask = df["age"].between(MIN_AGE, MAX_AGE)
    df_out_of_range = df.loc[~age_mask]
    df = df.loc[age_mask].copy()

    print(f"[users] Filtrados por edad implausible: {n_before - len(df)} "
          f"de {n_before} registros ({(n_before - len(df)) / n_before:.2%})")

    for col in ["attributes_notifications_marketing_push",
                "attributes_notifications_marketing_email"]:
        if col in df.columns:
            null_pct = df[col].isna().mean()
            print(f"[users] {col}: {null_pct:.2%} nulls -> imputados como 'unknown'")
            df[col] = df[col].astype("object")
            df[col] = df[col].where(df[col].notna(), "unknown")

    df["created_date"] = pd.to_datetime(df["created_date"])
    return df, df_out_of_range


def create_age_group(df: pd.DataFrame, age_col: str = "age") -> pd.DataFrame:
    df = df.copy()
    df["age_group"] = pd.qcut(
        df[age_col],
        q=4,
        labels=["18-33", "34-39", "40-48", "49+"],  # TODO: ajustar tras confirmar cortes exactos
    )
    print("[users] Distribución de age_group:")
    print(df["age_group"].value_counts().sort_index())
    return df


users_clean, users_dropped_age = clean_users(users, REFERENCE_DATE)
users_clean = create_age_group(users_clean)

# ---------------------------------------------------------------------------
# 3. Limpieza de `transactions`
# ---------------------------------------------------------------------------

def clean_transactions(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    n_before = len(df)
    df = df[df["amount_usd"] > 0].copy()
    print(f"[transactions] Filtradas por amount_usd <= 0: "
          f"{n_before - len(df)} de {n_before} registros")

    p995 = df["amount_usd"].quantile(0.995)
    n_outliers = (df["amount_usd"] > p995).sum()
    print(f"[transactions] {n_outliers} transacciones por encima del "
          f"percentil 99.5 (${p995:,.2f}) marcadas para revisión manual, "
          f"NO eliminadas automáticamente")
    df["amount_outlier_flag"] = df["amount_usd"] > p995

    merchant_cols = ["ea_cardholderpresence", "ea_merchant_mcc",
                      "ea_merchant_city", "ea_merchant_country"]
    for col in merchant_cols:
        if col in df.columns:
            null_pct = df[col].isna().mean()
            print(f"[transactions] {col}: {null_pct:.2%} nulls -> "
                  f"imputados como 'NOT_APPLICABLE'")
            df[col] = df[col].astype("object")
            df[col] = df[col].where(df[col].notna(), "NOT_APPLICABLE")

    df["created_date"] = pd.to_datetime(df["created_date"])
    return df


transactions_clean = clean_transactions(transactions)

# ---------------------------------------------------------------------------
# 4. Limpieza de `notifications`
# ---------------------------------------------------------------------------

def clean_notifications(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    null_report = df.isna().mean()
    print("[notifications] % de nulls por columna:")
    print(null_report[null_report > 0])
    df["created_date"] = pd.to_datetime(df["created_date"])
    return df


notifications_clean = clean_notifications(notifications)

# ---------------------------------------------------------------------------
# 5. `devices` (pendiente)
# ---------------------------------------------------------------------------

devices_pending = devices_raw.rename(
    columns={"string_field_0": "col_0_pending_mapping",
             "string_field_1": "col_1_pending_mapping"}
)

# ---------------------------------------------------------------------------
# 6. Resumen y export
# ---------------------------------------------------------------------------
print("\n--- Resumen de limpieza ---")
print(f"users: {len(users)} -> {len(users_clean)}")
print(f"transactions: {len(transactions)} -> {len(transactions_clean)}")
print(f"notifications: {len(notifications)} -> {len(notifications_clean)}")
print("devices: sin limpiar (pendiente de mapeo de columnas)")

users_clean.to_parquet("users_clean.parquet", index=False)
transactions_clean.to_parquet("transactions_clean.parquet", index=False)
notifications_clean.to_parquet("notifications_clean.parquet", index=False)