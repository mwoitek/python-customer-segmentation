# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Online Retail Dataset: RFM Scores
# In this notebook, I'll use the customer data to compute the RFM scores.
# ## Imports

# %%
from pathlib import Path
from typing import Literal, cast, get_args

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import AutoMinorLocator
from pandas.testing import assert_frame_equal, assert_index_equal, assert_series_equal

from utils.rfm import RFMAttribute, add_rfm_scores

# %% [markdown]
# ## Read customer data

# %%
# Path to dataset
file_path = Path.cwd().parents[1] / "data" / "online_retail.csv"
assert file_path.exists(), f"file doesn't exist: {file_path}"

# %%
df = pd.read_csv(
    file_path,
    dtype={
        "InvoiceNo": "category",
        "CustomerID": "category",
        "TotalPrice": np.float_,
    },
    parse_dates=["InvoiceDate"],
)
df.head(15)

# %%
df.info()


# %%
# This may seem redundant, but it allows this code to be reused later
def read_customer_data(file_path: Path) -> pd.DataFrame:
    return pd.read_csv(
        file_path,
        dtype={
            "InvoiceNo": "category",
            "CustomerID": "category",
            "TotalPrice": np.float_,
        },
        parse_dates=["InvoiceDate"],
    )


# %% [markdown]
# ## Compute RFM attributes
# ### Recency
# **To calculate the recency, I'll pretend that I'm performing this analysis 1
# day after the last piece of data was collected.**

# %%
today = df["InvoiceDate"].max() + pd.Timedelta(days=1)
today

# %% [markdown]
# Recall that, in this case, recency corresponds to the number of days since
# the last purchase. Figuring out the best way to compute this metric:

# %%
# Recency for a particular customer
(today - df.loc[df["CustomerID"] == "14688", "InvoiceDate"].max()).days

# %%
# Recency for all customers
recency = (today - df.groupby(by="CustomerID", observed=True).InvoiceDate.max()).dt.days.rename("Recency")
recency = cast(pd.Series, recency)
recency.head()

# %%
recency.loc["14688"]

# %% [markdown]
# This seems OK. Then I'll start building the DataFrame that will store the RFM
# scores:

# %%
df_rfm = (
    df.groupby(by="CustomerID", observed=True)
    .agg(LastPurchaseDate=pd.NamedAgg(column="InvoiceDate", aggfunc="max"))
    .assign(Recency=lambda x: (today - x.LastPurchaseDate).dt.days)
    .drop(columns="LastPurchaseDate")
)
df_rfm.head(15)

# %% [markdown]
# ### Frequency
# Here I'll use the following definition of frequency: for a given customer,
# frequency is the total number of purchases he/she made. Figuring out the best
# way to evaluate this metric:

# %%
# Frequency for a particular customer
df[df["CustomerID"] == "14688"].shape[0]

# %%
# Alternative solution
df.loc[df["CustomerID"] == "14688", "InvoiceNo"].nunique()

# %%
# Frequency for all customers
freq_1 = df.groupby(by="CustomerID", observed=True).InvoiceNo.count().rename("Frequency")
freq_1 = cast(pd.Series, freq_1)
freq_1.head()

# %%
freq_1.loc["14688"]

# %%
# Alternative solution
freq_2 = df.groupby(by="CustomerID", observed=True).InvoiceNo.nunique().rename("Frequency")
freq_2 = cast(pd.Series, freq_2)
freq_2.head()

# %%
assert_series_equal(freq_1, freq_2)

# %% [markdown]
# The two methods used above are equivalent. Then I'll calculate the frequency
# with the aid of the simplest one:

# %%
assert_index_equal(df_rfm.index, freq_1.index)

# %%
df_rfm["Frequency"] = df.groupby(by="CustomerID", observed=True).InvoiceNo.count()
df_rfm.head(15)

# %% [markdown]
# ### Monetary
# Here I'll use the following definition of monetary (value): for a given
# customer, monetary corresponds to the total amount spent by him/her. Figuring
# out the best way to evaluate this quantity:

# %%
# Monetary for a particular customer
df.loc[df["CustomerID"] == "14688", "TotalPrice"].sum()

# %%
# Monetary for all customers
monetary = df.groupby(by="CustomerID", observed=True).TotalPrice.sum().rename("Monetary")
monetary = cast(pd.Series, monetary)
monetary.head()

# %%
monetary.loc["14688"]

# %% [markdown]
# Now that I understand how to perform the desired calculation, I'll add
# monetary to my DataFrame:

# %%
assert_index_equal(df_rfm.index, monetary.index)

# %%
df_rfm["Monetary"] = df.groupby(by="CustomerID", observed=True).TotalPrice.sum()
df_rfm.head(15)

# %%
df_rfm.info()

# %%
# Quick consistency checks
assert (df_rfm["Recency"] > 0).all()
assert (df_rfm["Frequency"] > 0).all()
assert (df_rfm["Monetary"] > 0.0).all()

# %% [markdown]
# At this point, all RFM attributes have been evaluated. Before proceeding,
# let's collect the essential parts of the above code, and transform them into
# a function:


# %%
def compute_rfm_attributes(df: pd.DataFrame) -> pd.DataFrame:
    today = df.InvoiceDate.max() + pd.Timedelta(days=1)
    customer_groups = df.groupby(by="CustomerID", observed=True)
    return (
        customer_groups.agg(LastPurchaseDate=pd.NamedAgg(column="InvoiceDate", aggfunc="max"))
        .assign(
            Recency=lambda x: (today - x.LastPurchaseDate).dt.days,
            Frequency=customer_groups.InvoiceNo.count(),
            Monetary=customer_groups.TotalPrice.sum(),
        )
        .drop(columns="LastPurchaseDate")
    )


# %%
# Quick tests
customer_data = read_customer_data(file_path)
assert_frame_equal(customer_data, df)

# %%
rfm_attrs = compute_rfm_attributes(customer_data)
assert_frame_equal(rfm_attrs, df_rfm)

# %%
del customer_data
del rfm_attrs

# %% [markdown]
# ## Visualizing RFM attributes
# ### Boxplots

# %%
RFM_UNITS = {
    "Recency": "days",
    "Frequency": "purchases",
    "Monetary": "£",
}


def boxplot_rfm(
    df: pd.DataFrame,
    attr: RFMAttribute,
    figsize: tuple[float, float] = (6.0, 6.0),
) -> None:
    fig = plt.figure(figsize=figsize, layout="tight")
    ax = fig.add_subplot()
    ax.boxplot(df[attr])
    ax.set_title(f"Boxplot of {attr}")
    ax.set_xticks([])
    ax.set_ylabel(f"{attr} ({RFM_UNITS[attr]})")
    ax.set_ylim(bottom=0)
    plt.show()


# %%
# Recency
boxplot_rfm(df_rfm, "Recency")

# %%
# Frequency
boxplot_rfm(df_rfm, "Frequency")

# %%
# Monetary
boxplot_rfm(df_rfm, "Monetary")

# %% [markdown]
# Clearly, all RFM attributes have outliers. But the situation is "worse" for
# `Frequency` and `Monetary`. There may be an explanation for this. Some of the
# online retailer's customers are wholesale stores. It's reasonable to assume
# that such customers make purchases very frequently, and spend a good amount
# of money. This explains at least a portion of the outliers for `Frequency`
# and `Monetary`.
# ### KDE plots


# %%
def kde_rfm(
    df: pd.DataFrame,
    attr: RFMAttribute,
    figsize: tuple[float, float] = (8.0, 6.0),
) -> None:
    fig = plt.figure(figsize=figsize, layout="tight")
    ax = fig.add_subplot()
    sns.kdeplot(data=df, x=attr, ax=ax)
    ax.set_title(f"KDE for {attr}")
    ax.set_xlabel(f"{attr} ({RFM_UNITS[attr]})")
    plt.show()


# %%
# Recency
kde_rfm(df_rfm, "Recency")

# %%
# Frequency
kde_rfm(df_rfm, "Frequency")

# %%
# Monetary
kde_rfm(df_rfm, "Monetary")

# %% [markdown]
# ### Correlation


# %%
def correlation_heatmap(
    df: pd.DataFrame,
    columns: list[str] = ["Recency", "Frequency", "Monetary"],
    figsize: tuple[float, float] = (6.0, 6.0),
) -> None:
    fig = plt.figure(figsize=figsize, layout="tight")
    ax = fig.add_subplot()
    sns.heatmap(df[columns].corr(), annot=True, cmap=mpl.colormaps["coolwarm"], ax=ax)
    ax.set_title("Heatmap of Correlations")
    plt.show()


# %%
# Correlation between RFM attributes
correlation_heatmap(df_rfm)

# %% [markdown]
# ## Dealing with outliers
# Later, I'll use the RFM attributes to do customer segmentation with the aid
# of clustering algorithms. In these cases, the presence of outliers in the
# dataset may lead to poor results. For this reason, I'll implement a few
# functions for removing outliers.
#
# I won't use these functions right away. Except for testing. I'm including
# them here for the sake of organization. It makes sense to add the code for
# removing outliers after the one for computing the features.
# ### IQR method
# To remove outliers, I'll use the interquartile range (IQR) method. A very
# simple explanation of this method can be found
# [here](https://online.stat.psu.edu/stat200/lesson/3/3.2). It's important to
# note that, when we adopt this approach, the notion of an outlier is the same
# as when creating a boxplot. So I'm going to remove the same outliers that
# were shown above.
#
# Figuring out the best way to do this:

# %%
# Compute quartiles and bounds
cols = ["Frequency", "Monetary"]  # just an example
quartiles_and_bounds = (
    pd.concat(
        [
            df_rfm[cols].quantile(q=0.25).rename("Q1"),
            df_rfm[cols].quantile(q=0.75).rename("Q3"),
        ],
        axis=1,
    )
    .assign(
        IQR=lambda x: x.Q3 - x.Q1,
        LowerBound=lambda x: x.Q1 - 1.5 * x.IQR,
        UpperBound=lambda x: x.Q3 + 1.5 * x.IQR,
    )
    .transpose()
)
quartiles_and_bounds

# %%
bounds = quartiles_and_bounds.loc[["LowerBound", "UpperBound"], :]
bounds

# %%
# Select indexes I want to KEEP
idxs = df_rfm.index

for col in cols:
    mask_lower = df_rfm[col] < bounds.loc["LowerBound", col]
    idxs = idxs.drop(df_rfm[mask_lower].index, errors="ignore")

    mask_upper = df_rfm[col] > bounds.loc["UpperBound", col]
    idxs = idxs.drop(df_rfm[mask_upper].index, errors="ignore")

# %%
# Remove outliers
df_no_out = df_rfm.loc[idxs, :]

# %% [markdown]
# Check by recreating the boxplots:

# %%
# Frequency
boxplot_rfm(df_no_out, "Frequency")

# %%
# Monetary
boxplot_rfm(df_no_out, "Monetary")

# %% [markdown]
# Using the above code to define a couple of functions that allow us to remove
# outliers:


# %%
def compute_outlier_bounds(df: pd.DataFrame, columns: str | list[str], scale: float = 1.5) -> pd.DataFrame:
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
            LowerBound=lambda x: x.Q1 - scale * x.IQR,
            UpperBound=lambda x: x.Q3 + scale * x.IQR,
        )
        .transpose()
        .loc[["LowerBound", "UpperBound"], :]
    )


# %%
def remove_outliers(df: pd.DataFrame, columns: str | list[str], scale: float = 1.5) -> pd.DataFrame:
    bounds = compute_outlier_bounds(df, columns, scale)
    idxs = df.index

    if isinstance(columns, str):
        columns = [columns]

    for column in columns:
        mask_lower = df[column] < bounds.loc["LowerBound", column]
        idxs = idxs.drop(df[mask_lower].index, errors="ignore")

        mask_upper = df[column] > bounds.loc["UpperBound", column]
        idxs = idxs.drop(df[mask_upper].index, errors="ignore")

    return df.loc[idxs, :]


# %%
# At this point, this step isn't necessary. This command is here just for
# testing.

# Remove outliers from `Frequency` and `Monetary`
df_func = remove_outliers(df_rfm, columns=["Frequency", "Monetary"])

# %%
assert_frame_equal(df_func, df_no_out)
del df_func

# %% [markdown]
# ## Compute RFM scores
# Finally, it's time to compute the RFM scores. To do so, I'll use the
# `add_rfm_scores` function that I defined in the `utils` package:

# %%
df_rfm = add_rfm_scores(df_rfm)
df_rfm.head()

# %%
df_rfm.info()

# %% [markdown]
# ## Save computed scores

# %%
# Path to output CSV
out_file = file_path.parent / "rfm_scores_5.csv"

df_rfm.to_csv(out_file)

# %% [markdown]
# ## Visualization
# ### Recency and R score
# Check that every bin contains approximately the same number of customers:

# %%
fig = plt.figure(figsize=(8.0, 6.0), layout="tight")
ax = fig.add_subplot()
sns.countplot(data=df_rfm, x="RScore", ax=ax)
ax.bar_label(ax.containers[0])  # pyright: ignore [reportArgumentType]
ax.set_title("R Score: # Customers in each bin")
ax.set_xlabel("R Score")
ax.set_ylabel("Count")
ax.yaxis.set_minor_locator(AutoMinorLocator(4))
plt.show()

# %% [markdown]
# Distribution of `Recency` for each R score:

# %%
fig = plt.figure(figsize=(8.0, 6.0), layout="tight")
ax = fig.add_subplot()
sns.boxplot(data=df_rfm, x="RScore", y="Recency", ax=ax)
ax.set_title("Recency: Distribution for each R score")
ax.set_xlabel("R Score")
ax.set_ylabel("Recency (days)")
ax.set_ylim(bottom=0)
ax.yaxis.set_minor_locator(AutoMinorLocator(5))
plt.show()

# %%
# Quick check
df_rfm.loc[df_rfm["RScore"] == 1, "Recency"].describe()

# %% [markdown]
# ### Frequency and F score
# Next, we're going to do something very similar to what we've done above. Then
# it's convenient to use this code to define a couple of functions:

# %%
# Path to images directory
IMG_DIR = Path.cwd().parents[1] / "img"
assert IMG_DIR.exists(), f"directory doesn't exist: {IMG_DIR}"


# %%
def plot_bin_count(
    df: pd.DataFrame,
    score: Literal["R", "F", "M"],
    *,
    save: bool = False,
    figsize: tuple[float, float] = (8.0, 6.0),
) -> None:
    fig = plt.figure(figsize=figsize, layout="tight")
    ax = fig.add_subplot()

    sns.countplot(data=df, x=f"{score}Score", ax=ax)
    ax.bar_label(ax.containers[0])  # pyright: ignore [reportArgumentType]

    ax.set_title(f"{score} Score: # Customers in each bin")
    ax.set_xlabel(f"{score} Score")
    ax.set_ylabel("Count")

    if save:
        out_img = IMG_DIR / f"bin_count_{score.lower()}score.png"
        fig.savefig(out_img)
        plt.close(fig)
    else:
        plt.show()


# %%
def plot_distribution_by_score(
    df: pd.DataFrame,
    attr: RFMAttribute,
    *,
    save: bool = False,
    figsize: tuple[float, float] = (8.0, 6.0),
) -> None:
    fig = plt.figure(figsize=figsize, layout="tight")
    ax = fig.add_subplot()

    first_letter = attr[0]
    sns.boxplot(
        data=df,
        x=f"{first_letter}Score",
        y=attr,
        ax=ax,
        # I just want to know if the RFM attributes are distributed in a way
        # that makes sense. For this reason, I'll hide the outliers.
        showfliers=False,
    )

    ax.set_title(f"{attr}: Distribution for each {first_letter} score")
    ax.set_xlabel(f"{first_letter} Score")
    ax.set_ylabel(f"{attr} ({RFM_UNITS[attr]})")
    ax.set_ylim(bottom=0)

    if save:
        out_img = IMG_DIR / f"{attr.lower()}_distribution.png"
        fig.savefig(out_img)
        plt.close(fig)
    else:
        plt.show()


# %% [markdown]
# Using these functions to generate the desired plots:

# %%
plot_bin_count(df_rfm, "F")

# %%
plot_distribution_by_score(df_rfm, "Frequency")

# %% [markdown]
# In the case of the F score, the number of customers in each bin varies
# considerably. Basically, this imbalance is a consequence of the following
# fact: **more than a third of customers only made a single purchase**. This
# fact can be verified as follows:

# %%
df_rfm.Frequency.transform(lambda f: "1 purchase" if f == 1 else "2 or more purchases").value_counts(
    normalize=True, sort=False
).mul(100.0).transform(lambda p: f"{p:.2f}%").sort_index()

# %% [markdown]
# In fact, all customers who purchased only once were assigned an F score of 1:

# %%
df_rfm.loc[df_rfm.FScore == 1, "Frequency"].unique()

# %% [markdown]
# Something similar happened for F scores 2 and 3:

# %%
df_rfm.loc[df_rfm.FScore == 2, "Frequency"].unique()

# %%
df_rfm.loc[df_rfm.FScore == 3, "Frequency"].unique()

# %% [markdown]
# This is the reason why we see no variation in the first 3 boxplots above.
#
# The `FScore = 4` case is the first that corresponds to more than one
# `Frequency` value:

# %%
df_rfm.loc[df_rfm.FScore == 4, "Frequency"].value_counts()

# %% [markdown]
# When `FScore = 5`, there's a lot more variation:

# %%
np.sort(df_rfm.loc[df_rfm.FScore == 5, "Frequency"].unique())

# %% [markdown]
# So it makes sense to recreate the above boxplot just for this case. This
# time, we won't hide the outliers. The desired plot can be created as follows:

# %%
fig = plt.figure(figsize=(6.0, 6.0), layout="tight")
ax = fig.add_subplot()
ax.boxplot(df_rfm.loc[df_rfm.FScore == 5, "Frequency"])
ax.set_title("Frequency: Distribution for F Score = 5")
ax.set_xticks([])
ax.set_ylabel("Frequency (purchases)")
ax.set_ylim(bottom=0)
plt.show()

# %% [markdown]
# ### Monetary value and M score
# In this case, each bin has approximately the same number of customers:

# %%
plot_bin_count(df_rfm, "M")

# %%
plot_distribution_by_score(df_rfm, "Monetary")

# %% [markdown]
# ## Summarizing through functions
# The RFM analysis isn't complete yet. But I've already achieved my goal for
# this notebook. As usual, I'll conclude by writing a couple of functions that
# summarize what I just did.


# %%
def compute_and_save_rfm_scores(
    file_path: Path,
    *,
    outlier_cols: str | list[str] | None = None,
    outlier_scale: float = 1.5,
    num_bins: int = 5,
) -> None:
    df_rfm = read_customer_data(file_path).pipe(compute_rfm_attributes)
    if outlier_cols is not None:
        df_rfm = remove_outliers(df_rfm, outlier_cols, outlier_scale)
    add_rfm_scores(df_rfm, num_bins).to_csv(file_path.parent / f"rfm_scores_{num_bins}.csv")


# %%
compute_and_save_rfm_scores(Path.cwd().parents[1] / "data" / "online_retail.csv")


# %%
def plot_rfm_attributes_and_scores(df: pd.DataFrame, figsize: tuple[float, float] = (8.0, 6.0)) -> None:
    attrs = get_args(RFMAttribute)
    for score in (attr[0] for attr in attrs):
        plot_bin_count(df, score, save=True, figsize=figsize)
    for attr in attrs:
        plot_distribution_by_score(df, attr, save=True, figsize=figsize)


# %%
plot_rfm_attributes_and_scores(df_rfm)
