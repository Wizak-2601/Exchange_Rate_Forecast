import pandas as pd
from datasets import load_dataset

def load_exchange_data(univariate=False, target_col="OT"):
    dataset = load_dataset("ts-arena/exchange_rate")

    df_train = dataset["train"].to_pandas()
    df_val   = dataset["validation"].to_pandas()
    df_test  = dataset["test"].to_pandas()

    # Rename currency columns
    rename_dict = {str(i): f"Usd_vs_Curr{i+1}" for i in range(7)}
    df_train.rename(columns=rename_dict, inplace=True)
    df_val.rename(columns=rename_dict, inplace=True)
    df_test.rename(columns=rename_dict, inplace=True)

    if "timestamp_idx" in df_train.columns:
        df_train = df_train.drop(columns=["timestamp_idx"])
        df_val   = df_val.drop(columns=["timestamp_idx"])
        df_test  = df_test.drop(columns=["timestamp_idx"])

    if univariate:
        df_train = df_train[[target_col]]
        df_val   = df_val[[target_col]]
        df_test  = df_test[[target_col]]

    return (
        df_train.values.astype("float32"),
        df_val.values.astype("float32"),
        df_test.values.astype("float32")
    )
