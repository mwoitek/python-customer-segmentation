import pandas as pd

from utils.decorators import count_dropped


def compute_outlier_bounds(df: pd.DataFrame, columns: str | list[str]) -> pd.DataFrame:
    if isinstance(columns, str):
        columns = [columns]
    return (
        pd.concat(
            [
                df[columns].quantile(q=0.25).rename("Q1"),
                df[columns].quantile(q=0.75).rename("Q3"),
            ],
            axis=1,
        )
        .assign(
            IQR=lambda x: x.Q3 - x.Q1,
            LowerBound=lambda x: x.Q1 - 1.5 * x.IQR,
            UpperBound=lambda x: x.Q3 + 1.5 * x.IQR,
        )
        .transpose()
    ).loc[["LowerBound", "UpperBound"], :]


@count_dropped
def remove_outliers(df: pd.DataFrame, columns: str | list[str]) -> pd.DataFrame:
    bounds = compute_outlier_bounds(df, columns)
    idxs = df.index

    if isinstance(columns, str):
        columns = [columns]

    for column in columns:
        mask_lower = df[column] < bounds.loc["LowerBound", column]
        idxs = idxs.drop(df[mask_lower].index, errors="ignore")

        mask_upper = df[column] > bounds.loc["UpperBound", column]
        idxs = idxs.drop(df[mask_upper].index, errors="ignore")

    return df.loc[idxs, :]
