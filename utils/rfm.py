from typing import Literal, get_args

import numpy as np
import pandas as pd

RFMAttribute = Literal["Recency", "Frequency", "Monetary"]


def add_score_column(df: pd.DataFrame, attr: RFMAttribute, num_bins: int = 5) -> pd.DataFrame:
    score_name = f"{attr[0]}Score"
    score_labels = list(range(num_bins, 0, -1)) if attr == "Recency" else list(range(1, num_bins + 1))

    rank = df[attr].rank(method="min").astype(np.int_)
    df[score_name] = pd.cut(rank, num_bins, labels=score_labels)
    if attr == "Recency":
        df[score_name] = df[score_name].cat.reorder_categories(score_labels[::-1], ordered=True)

    return df


def add_rfm_scores(df: pd.DataFrame, num_bins: int = 5) -> pd.DataFrame:
    for attr in get_args(RFMAttribute):
        df = add_score_column(df, attr, num_bins)

    score_cols = [f"{attr[0]}Score" for attr in get_args(RFMAttribute)]
    df["RFMCell"] = df[score_cols].agg(lambda r: f"{r.iloc[0]},{r.iloc[1]},{r.iloc[2]}", axis="columns")
    df["RFMScore"] = df[score_cols].astype(np.int_).agg("mean", axis="columns")

    return df
