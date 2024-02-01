from pathlib import Path
from typing import Literal, cast, get_args

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.ticker import AutoMinorLocator
from more_itertools import unique_everseen

RFMAttribute = Literal["Recency", "Frequency", "Monetary"]

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
IMG_DIR = Path(__file__).resolve().parents[1] / "img"

SEGMENTS_5 = {
    "5,5,5": "Champions",
    "5,5,4": "Champions",
    "5,4,5": "Champions",
    "5,4,4": "Champions",
    "4,5,5": "Champions",
    "4,5,4": "Champions",
    "4,4,5": "Champions",
    "5,4,3": "Loyal",
    "4,4,4": "Loyal",
    "4,3,5": "Loyal",
    "3,5,5": "Loyal",
    "3,5,4": "Loyal",
    "3,4,5": "Loyal",
    "3,4,4": "Loyal",
    "3,3,5": "Loyal",
    "5,5,3": "Potential Loyalists",
    "5,5,2": "Potential Loyalists",
    "5,5,1": "Potential Loyalists",
    "5,4,2": "Potential Loyalists",
    "5,4,1": "Potential Loyalists",
    "5,3,3": "Potential Loyalists",
    "5,3,2": "Potential Loyalists",
    "5,3,1": "Potential Loyalists",
    "4,5,3": "Potential Loyalists",
    "4,5,2": "Potential Loyalists",
    "4,5,1": "Potential Loyalists",
    "4,4,2": "Potential Loyalists",
    "4,4,1": "Potential Loyalists",
    "4,3,3": "Potential Loyalists",
    "4,3,2": "Potential Loyalists",
    "4,3,1": "Potential Loyalists",
    "4,2,3": "Potential Loyalists",
    "3,5,3": "Potential Loyalists",
    "3,5,2": "Potential Loyalists",
    "3,5,1": "Potential Loyalists",
    "3,4,2": "Potential Loyalists",
    "3,4,1": "Potential Loyalists",
    "3,3,3": "Potential Loyalists",
    "3,2,3": "Potential Loyalists",
    "5,1,2": "New Customers",
    "5,1,1": "New Customers",
    "4,2,2": "New Customers",
    "4,2,1": "New Customers",
    "4,1,2": "New Customers",
    "4,1,1": "New Customers",
    "3,1,1": "New Customers",
    "5,2,5": "Promising",
    "5,2,4": "Promising",
    "5,2,3": "Promising",
    "5,2,2": "Promising",
    "5,2,1": "Promising",
    "5,1,5": "Promising",
    "5,1,4": "Promising",
    "5,1,3": "Promising",
    "4,2,5": "Promising",
    "4,2,4": "Promising",
    "4,1,5": "Promising",
    "4,1,4": "Promising",
    "4,1,3": "Promising",
    "3,1,5": "Promising",
    "3,1,4": "Promising",
    "3,1,3": "Promising",
    "5,3,5": "Need Attention",
    "5,3,4": "Need Attention",
    "4,4,3": "Need Attention",
    "4,3,4": "Need Attention",
    "3,4,3": "Need Attention",
    "3,3,4": "Need Attention",
    "3,2,5": "Need Attention",
    "3,2,4": "Need Attention",
    "3,3,1": "About to Sleep",
    "3,2,1": "About to Sleep",
    "3,1,2": "About to Sleep",
    "2,2,1": "About to Sleep",
    "2,1,3": "About to Sleep",
    "2,1,5": "Cannot Lose Them",
    "2,1,4": "Cannot Lose Them",
    "1,5,5": "Cannot Lose Them",
    "1,5,4": "Cannot Lose Them",
    "1,4,4": "Cannot Lose Them",
    "1,1,5": "Cannot Lose Them",
    "1,1,4": "Cannot Lose Them",
    "1,1,3": "Cannot Lose Them",
    "2,5,5": "At Risk",
    "2,5,4": "At Risk",
    "2,5,3": "At Risk",
    "2,5,2": "At Risk",
    "2,4,5": "At Risk",
    "2,4,4": "At Risk",
    "2,4,3": "At Risk",
    "2,4,2": "At Risk",
    "2,3,5": "At Risk",
    "2,3,4": "At Risk",
    "2,2,5": "At Risk",
    "2,2,4": "At Risk",
    "1,5,3": "At Risk",
    "1,5,2": "At Risk",
    "1,4,5": "At Risk",
    "1,4,3": "At Risk",
    "1,4,2": "At Risk",
    "1,3,5": "At Risk",
    "1,3,4": "At Risk",
    "1,3,3": "At Risk",
    "1,2,5": "At Risk",
    "1,2,4": "At Risk",
    "3,3,2": "Hibernating",
    "3,2,2": "Hibernating",
    "2,5,1": "Hibernating",
    "2,4,1": "Hibernating",
    "2,3,3": "Hibernating",
    "2,3,2": "Hibernating",
    "2,3,1": "Hibernating",
    "2,2,3": "Hibernating",
    "2,2,2": "Hibernating",
    "2,1,2": "Hibernating",
    "2,1,1": "Hibernating",
    "1,3,2": "Hibernating",
    "1,2,3": "Hibernating",
    "1,2,2": "Hibernating",
    "1,5,1": "Lost",
    "1,4,1": "Lost",
    "1,3,1": "Lost",
    "1,2,1": "Lost",
    "1,1,2": "Lost",
    "1,1,1": "Lost",
}


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


def label_customers(df: pd.DataFrame, segments_dict: dict[str, str]) -> pd.DataFrame:
    return df.assign(Segment=df["RFMCell"].map(segments_dict).astype("category"))


def customers_by_segment(df: pd.DataFrame, segments_dict: dict[str, str]) -> pd.DataFrame:
    return (
        df["Segment"]
        .value_counts()
        .to_frame()
        .rename(columns={"count": "CustomerCount"})
        .assign(CustomerPercentage=(100.0 * df["Segment"].value_counts(normalize=True)).round(2))
        .reindex(list(unique_everseen(segments_dict.values())))
    )


def revenue_by_segment(df: pd.DataFrame, segments_dict: dict[str, str]) -> pd.DataFrame:
    seg_groups = df.groupby(by="Segment", observed=True)
    total_rev = np.round(df["Monetary"].sum(), 2)
    return (
        seg_groups.Monetary.sum()
        .round(2)
        .to_frame()
        .rename(columns={"Monetary": "Revenue"})
        .assign(RevenuePercentage=(100.0 * seg_groups.Monetary.agg(lambda c: c.sum() / total_rev)).round(2))
        .reindex(list(unique_everseen(segments_dict.values())))
    )


def get_segment_data(df: pd.DataFrame, segments_dict: dict[str, str]) -> pd.DataFrame:
    return pd.concat(
        [
            customers_by_segment(df, segments_dict),
            revenue_by_segment(df, segments_dict),
        ],
        axis=1,
    )


def plot_customers_by_segment(
    df_segment: pd.DataFrame,
    *,
    save: bool = False,
    figsize: tuple[float, float] = (13.0, 6.5),
) -> None:
    fig, axs = plt.subplots(1, 2, sharey=True, figsize=figsize, layout="constrained")

    axs_list = axs.flatten().tolist()
    axs_list = cast(list[Axes], axs_list)

    axs_list[0].barh(df_segment.index, df_segment["CustomerCount"])
    axs_list[0].set_title("Number of Customers by Segment")
    axs_list[0].set_xlabel("Number of Customers")

    axs_list[1].barh(df_segment.index, df_segment["CustomerPercentage"])
    axs_list[1].set_title("Percentage of Customers in Each Segment")
    axs_list[1].set_xlabel("Percentage (%)")

    axs_list[1].invert_yaxis()
    fig.suptitle("Customers by Segment", fontsize="xx-large")

    if save:
        fig.savefig(IMG_DIR / "customers_by_segment.png")
        plt.close(fig)
    else:
        plt.show()


def plot_revenue_by_segment(
    df_segment: pd.DataFrame,
    *,
    save: bool = False,
    figsize: tuple[float, float] = (13.0, 6.5),
) -> None:
    fig, axs = plt.subplots(1, 2, sharey=True, figsize=figsize, layout="constrained")

    axs_list = axs.flatten().tolist()
    axs_list = cast(list[Axes], axs_list)

    axs_list[0].barh(df_segment.index, df_segment["Revenue"] / 1e6)
    axs_list[0].xaxis.set_minor_locator(AutoMinorLocator(5))
    axs_list[0].set_title("Revenue by Segment")
    axs_list[0].set_xlabel("Revenue (Million £)")

    axs_list[1].barh(df_segment.index, df_segment["RevenuePercentage"])
    axs_list[1].xaxis.set_minor_locator(AutoMinorLocator(5))
    axs_list[1].set_title("Percentage of Revenue from Each Segment")
    axs_list[1].set_xlabel("Percentage (%)")

    axs_list[1].invert_yaxis()
    fig.suptitle("Revenue", fontsize="xx-large")

    if save:
        fig.savefig(IMG_DIR / "revenue_by_segment.png")
        plt.close(fig)
    else:
        plt.show()


def compute_and_plot_segment_data(
    df: pd.DataFrame,
    segments_dict: dict[str, str],
    figsize: tuple[float, float] = (13.0, 6.5),
) -> None:
    df_segment = get_segment_data(df, segments_dict)
    out_file = DATA_DIR / "segment_data.csv"
    df_segment.to_csv(out_file, index=True)

    plot_customers_by_segment(df_segment, save=True, figsize=figsize)
    plot_revenue_by_segment(df_segment, save=True, figsize=figsize)
