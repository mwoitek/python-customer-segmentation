from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd

from utils.outliers import remove_outliers
from utils.rfm import add_rfm_scores, label_customers


def get_clean_data(file_path: Path) -> pd.DataFrame:
    cols = ["InvoiceNo", "InvoiceDate", "CustomerID", "Quantity", "UnitPrice"]
    df = pd.read_excel(
        file_path,
        usecols=cols,
        dtype={col: object for col in filter(lambda c: c != "InvoiceDate", cols)},
        parse_dates=["InvoiceDate"],
    ).loc[:, cols]
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
    aggregated_data = (
        clean_data.groupby("InvoiceNo", observed=True)
        .apply(compute_total_price, include_groups=False)
        .reset_index()
    )
    out_file = file_path.with_suffix(".csv")
    aggregated_data.to_csv(out_file, index=False)


def read_prepared_data(file_path: Path) -> pd.DataFrame:
    return pd.read_csv(
        file_path,
        dtype={
            "InvoiceNo": "category",
            "CustomerID": "category",
            "TotalPrice": np.float_,
        },
        parse_dates=["InvoiceDate"],
    )


def compute_rfm_attributes(df: pd.DataFrame) -> pd.DataFrame:
    customer_groups = df.groupby(by="CustomerID", observed=True)

    today = df["InvoiceDate"].max() + pd.Timedelta(days=1)
    today = cast(pd.Timestamp, today)

    df_rfm = customer_groups.InvoiceDate.max().to_frame().rename(columns={"InvoiceDate": "LastPurchaseDate"})
    df_rfm = cast(pd.DataFrame, df_rfm)
    df_rfm["Recency"] = (today - df_rfm["LastPurchaseDate"]).dt.days
    df_rfm = df_rfm.drop(columns="LastPurchaseDate")

    df_rfm["Frequency"] = customer_groups.InvoiceNo.count()
    df_rfm["Monetary"] = customer_groups.TotalPrice.sum()

    return df_rfm


def compute_and_save_rfm_scores(
    file_path: Path,
    num_bins: int = 5,
    outlier_cols: str | list[str] | None = None,
) -> None:
    df_rfm = read_prepared_data(file_path).pipe(compute_rfm_attributes)
    if outlier_cols is not None:
        df_rfm = remove_outliers(df_rfm, outlier_cols)
    add_rfm_scores(df_rfm, num_bins).to_csv(
        file_path.parent / f"rfm_scores_{num_bins}.csv",
        index=True,
    )


def read_rfm_scores(file_path: Path) -> pd.DataFrame:
    df = pd.read_csv(
        file_path,
        dtype={
            "CustomerID": "category",
            "Recency": np.int_,
            "Frequency": np.int_,
            "Monetary": np.float_,
            "RScore": "category",
            "FScore": "category",
            "MScore": "category",
            "RFMCell": object,
            "RFMScore": np.float_,
        },
        index_col="CustomerID",
    )

    score_cols = ["RScore", "FScore", "MScore"]
    df[score_cols] = df[score_cols].transform(lambda col: col.cat.as_ordered())

    return df


def add_labels_and_save(file_path: Path, segments_dict: dict[str, str]) -> None:
    df = read_rfm_scores(file_path)
    df = label_customers(df, segments_dict)
    out_file = file_path.parent / "rfm_segments.csv"
    df.to_csv(out_file, index=True)
