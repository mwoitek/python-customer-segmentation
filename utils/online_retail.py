from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd


def get_clean_data(file_path: Path) -> pd.DataFrame:
    cols = ["InvoiceNo", "InvoiceDate", "CustomerID", "Quantity", "UnitPrice"]
    df = pd.read_excel(file_path, usecols=cols, dtype={col: object for col in cols}).loc[:, cols]
    df = cast(pd.DataFrame, df)

    df = df.dropna()

    df = df.loc[df["Quantity"] > 0, :]
    df = cast(pd.DataFrame, df)

    df = df.loc[df["UnitPrice"] != 0.0, :]
    df = cast(pd.DataFrame, df)

    df = df.reset_index(drop=True)

    df["InvoiceNo"] = df["InvoiceNo"].astype("category")
    df["CustomerID"] = df["CustomerID"].astype("category")
    df["Quantity"] = df["Quantity"].astype(np.int_)
    df["UnitPrice"] = df["UnitPrice"].astype(np.float_)

    df["InvoiceDate"] = df["InvoiceDate"].dt.normalize()

    return df


def compute_total_price(df_group: pd.DataFrame) -> pd.Series:
    return pd.Series(
        data={
            "InvoiceDate": df_group["InvoiceDate"].iloc[0],
            "CustomerID": df_group["CustomerID"].iloc[0],
            "TotalPrice": (df_group["Quantity"] * df_group["UnitPrice"]).sum(),
        }
    )


def prepare_and_save_data(file_path: Path) -> None:
    clean_data = get_clean_data(file_path)
    aggregated_data = clean_data.groupby("InvoiceNo", observed=True).apply(compute_total_price).reset_index()
    out_file = file_path.with_suffix(".csv")
    aggregated_data.to_csv(out_file, index=False)
