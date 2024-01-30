# %% [markdown]
# # Online Retail Dataset: RFM Scores
#
# In this notebook, I'll use the prepared dataset to compute the RFM scores.
#
# ## Imports

# %%
from pathlib import Path
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.ticker import AutoMinorLocator
from pandas.testing import assert_frame_equal, assert_index_equal, assert_series_equal

from utils.rfm import add_rfm_scores

# %% [markdown]
# ## Read prepared dataset

# %%
# File path for dataset
file_path = Path.cwd().parents[1] / "data" / "online_retail.csv"
assert file_path.exists(), f"file doesn't exist: {file_path}"
assert file_path.is_file(), f"not a file: {file_path}"

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

# %% [markdown]
# ## Compute RFM attributes
# ### Recency
#
# To calculate the recency, I'll pretend that I'm performing this analysis 1
# day after the last piece of data was collected.

# %%
today = df["InvoiceDate"].max() + pd.Timedelta(days=1)
today = cast(pd.Timestamp, today)
today

# %% [markdown]
# Recall that, in this case, recency corresponds to the number of days since
# the last purchase. Figuring out the best way to compute this metric:

# %%
# Recency for a particular customer
(today - df.loc[df["CustomerID"] == "14688", "InvoiceDate"].max()).days

# %%
# Recency for all customers
recency = (today - df.groupby(by="CustomerID", observed=True).InvoiceDate.max()).dt.days
recency = cast(pd.Series, recency)

assert (recency > 0).all()

recency.head()

# %%
recency.loc["14688"]

# %% [markdown]
# This seems OK. Then I'll start building the DataFrame that will store the RFM
# scores:

# %%
df_rfm = (
    df.groupby(by="CustomerID", observed=True)
    .InvoiceDate.max()
    .to_frame()
    .rename(columns={"InvoiceDate": "LastPurchaseDate"})
)
df_rfm = cast(pd.DataFrame, df_rfm)

df_rfm["Recency"] = (today - df_rfm["LastPurchaseDate"]).dt.days
df_rfm = df_rfm.drop(columns="LastPurchaseDate")

df_rfm.head()

# %% [markdown]
# ### Frequency
#
# Here I'll use the following definition of frequency: for a given customer,
# frequency is the total number of purchases he/she made. Figuring out the best
# way to evaluate this metric:

# %%
# Frequency for a particular customer
df[df["CustomerID"] == "14688"].shape[0]

# %%
df.loc[df["CustomerID"] == "14688", "InvoiceNo"].nunique()

# %%
# Frequency for all customers
freq_1 = df.groupby(by="CustomerID", observed=True).InvoiceNo.count()
freq_1 = cast(pd.Series, freq_1)
freq_1.name = "Frequency"
freq_1.head()

# %%
freq_1.loc["14688"]

# %%
freq_2 = df.groupby(by="CustomerID", observed=True).InvoiceNo.nunique()
freq_2 = cast(pd.Series, freq_2)
freq_2.name = "Frequency"
freq_2.head()

# %%
assert_series_equal(freq_1, freq_2)

# %% [markdown]
# The two methods used above are equivalent. Then I'll calculate the frequency
# with the aid of the simplest one:

# %%
assert_index_equal(df_rfm.index, freq_1.index)

df_rfm["Frequency"] = df.groupby(by="CustomerID", observed=True).InvoiceNo.count()
df_rfm.head()

# %% [markdown]
# ### Monetary
#
# Here I'll use the following definition of monetary: for a given customer,
# monetary corresponds to the total amount spent by him/her. Figuring out the
# best way to evaluate this quantity:

# %%
# Monetary for a particular customer
df.loc[df["CustomerID"] == "14688", "TotalPrice"].sum()

# %%
# Monetary for all customers
monetary = df.groupby(by="CustomerID", observed=True).TotalPrice.sum()
monetary = cast(pd.Series, monetary)
monetary.head()

# %%
monetary.loc["14688"]

# %% [markdown]
# Now that I understand how to perform the desired calculation, I'll add
# monetary to my DataFrame:

# %%
assert_index_equal(df_rfm.index, monetary.index)

df_rfm["Monetary"] = df.groupby(by="CustomerID", observed=True).TotalPrice.sum()
df_rfm.head()

# %%
df_rfm.info()

# %%
# Quick consistency checks
assert (df_rfm["Recency"] > 0).all()
assert (df_rfm["Frequency"] > 0).all()
assert (df_rfm["Monetary"] > 0.0).all()

# %% [markdown]
# ### Summarizing through functions
#
# At this point, all RFM attributes have been evaluated. Before proceeding,
# let's collect the essential parts of the above code, and transform them into
# a couple of functions.


# %%
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


# %%
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


# %%
# Quick tests
prepared_data = read_prepared_data(file_path)
assert_frame_equal(prepared_data, df)

# %%
rfm_attrs = compute_rfm_attributes(prepared_data)
assert_frame_equal(rfm_attrs, df_rfm)

# %%
del prepared_data
del rfm_attrs

# %% [markdown]
# ## Compute RFM scores
#
# Finally, it's possible to compute the RFM scores. To do so, I'll use the
# `add_rfm_scores` function that I defined in the `utils` package:

# %%
df_rfm = add_rfm_scores(df_rfm)
df_rfm.head()

# %%
df_rfm.info()

# %% [markdown]
# ## Save computed scores

# %%
# File path for output CSV
out_file = file_path.parent / "rfm_scores.csv"

df_rfm.to_csv(out_file, index=True)

# %% [markdown]
# ## Visualization
# ### Recency and R score
#
# Check that every bin contains approximately the same number of customers:

# %%
fig, ax = plt.subplots(figsize=(8.0, 6.0), layout="tight")
ax = cast(Axes, ax)
sns.countplot(data=df_rfm, x="RScore", ax=ax)
ax.set_title("R Score: # Customers in each bin")
ax.set_xlabel("R Score")
ax.set_ylabel("Count")
ax.yaxis.set_minor_locator(AutoMinorLocator(4))
plt.show()

# %%
# These are the values shown above:
df_rfm["RScore"].value_counts(sort=False)

# %% [markdown]
# Distribution of `Recency` for each R score:

# %%
fig, ax = plt.subplots(figsize=(8.0, 6.0), layout="tight")
ax = cast(Axes, ax)
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
# ## Summarizing through a function
#
# The RFM analysis isn't complete yet. But I've already achieved my goal for
# this notebook. As usual, I'll conclude by writing a function that summarizes
# what I just did.


# %%
def compute_and_save_rfm_scores(file_path: Path, num_bins: int = 5) -> None:
    prepared_data = read_prepared_data(file_path)
    df_rfm = compute_rfm_attributes(prepared_data)
    df_rfm = add_rfm_scores(df_rfm, num_bins)
    out_file = file_path.parent / f"rfm_scores_{num_bins}.csv"
    df_rfm.to_csv(out_file, index=True)


# %%
# compute_and_save_rfm_scores(Path.cwd().parents[1] / "data" / "online_retail.csv", num_bins=5)
