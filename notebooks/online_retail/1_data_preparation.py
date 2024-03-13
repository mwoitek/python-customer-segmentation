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
# # Online Retail Dataset: Data Preparation
# In this notebook, I'll prepare the dataset for analysis.
# ## Imports

# %%
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

# %% [markdown]
# ## Read dataset

# %%
# Path to dataset
file_path = Path.cwd().parents[1] / "data" / "online_retail.xlsx"
assert file_path.exists(), f"file doesn't exist: {file_path}"

# %%
# Columns I'll actually use
cols = ["InvoiceNo", "InvoiceDate", "CustomerID", "Quantity", "UnitPrice"]

df = pd.read_excel(
    file_path,
    usecols=cols,
    dtype={
        "InvoiceNo": object,
        "CustomerID": object,
        "Quantity": np.int_,
        "UnitPrice": np.float_,
    },
    parse_dates=["InvoiceDate"],
).loc[:, cols]
df.head(10)

# %%
df.info()

# %% [markdown]
# ## Data cleaning

# %%
# Number of missing values
df.isna().sum()

# %% [markdown]
# The only column that has missing values is `CustomerID`. But we really need
# to know who bought what. Then rows with missing `CustomerID` have to go.

# %%
df = df[df.CustomerID.notna()]
df.info()

# %% [markdown]
# `CustomerID` is supposed to be a 5-digit integer number. Let's confirm that
# this is the case:

# %%
ids = df["CustomerID"].astype(str)
assert ids.str.len().eq(5).all()
assert ids.str.isdigit().all()
del ids

# %% [markdown]
# Looking for invalid quantities:

# %%
# `Quantity` should be strictly positive
# So this should equal 0
df.Quantity.le(0).sum()

# %% [markdown]
# Not every row corresponds to a sale. When the invoice number starts with "C",
# that transaction was canceled. That explains the observations with
# non-positive quantities.

# %%
df["InvoiceNo"].astype(str).str.startswith("C").sum()

# %%
tmp_df = df.loc[df["InvoiceNo"].astype(str).str.startswith("C"), ["InvoiceNo", "Quantity"]]
tmp_df.head(10)

# %%
assert tmp_df.Quantity.le(0).all()
del tmp_df

# %% [markdown]
# I chose to remove those rows:

# %%
df = df[df.Quantity > 0]

# %% [markdown]
# `InvoiceNo` is supposed to be a 6-digit integer number. Let's check that this
# column is OK:

# %%
invoice_nums = df.InvoiceNo.astype(str)
assert invoice_nums.str.startswith("C").sum() == 0
assert invoice_nums.str.len().eq(6).all()
assert invoice_nums.str.isdigit().all()
del invoice_nums

# %% [markdown]
# Looking for invalid prices:

# %%
# Negative prices
df.UnitPrice.lt(0.0).sum()  # OK

# %%
# Products that cost nothing
df.UnitPrice.eq(0.0).sum()

# %% [markdown]
# I don't know how to explain such values. They should make no difference. Then
# I chose to drop them:

# %%
df = df[df.UnitPrice > 0.0]

# %% [markdown]
# ## Finish fixing dataset

# %%
# Use appropriate data types
df.InvoiceNo = df.InvoiceNo.astype("category")
assert not df.InvoiceNo.cat.ordered

df.CustomerID = df.CustomerID.astype("category")
assert not df.CustomerID.cat.ordered

# %%
df = df.reset_index(drop=True)
df.info()

# %% [markdown]
# The only part of `InvoiceDate` that matters is the date. The following
# command sets all the times to midnight:

# %%
df["InvoiceDate"] = df["InvoiceDate"].dt.normalize()
df["InvoiceDate"].head()

# %% [markdown]
# For convenience, I'll collect the essential parts of the above code, and
# create a function:


# %%
def get_clean_data(file_path: Path) -> pd.DataFrame:
    cols = ["InvoiceNo", "InvoiceDate", "CustomerID", "Quantity", "UnitPrice"]
    df = pd.read_excel(
        file_path,
        usecols=cols,
        dtype={
            "InvoiceNo": object,
            "CustomerID": object,
            "Quantity": np.int_,
            "UnitPrice": np.float_,
        },
        parse_dates=["InvoiceDate"],
    )
    return (
        df.loc[df.CustomerID.notna() & df.Quantity.gt(0) & df.UnitPrice.gt(0.0), cols]
        .assign(
            InvoiceNo=lambda x: x.InvoiceNo.astype("category"),
            InvoiceDate=lambda x: x.InvoiceDate.dt.normalize(),
            CustomerID=lambda x: x.CustomerID.astype("category"),
        )
        .reset_index(drop=True)
    )


# %%
# Quick check
df_func = get_clean_data(file_path)
assert_frame_equal(df_func, df)
del df_func

# %% [markdown]
# ## Aggregate data
# Before aggregating the data, I'll do some more consistency tests. Rows with
# the same `InvoiceNo` must also have the same `InvoiceDate`. For a specific
# value of `InvoiceNo`, this can be tested as follows:

# %%
assert df.loc[df["InvoiceNo"] == 536365, "InvoiceDate"].nunique() == 1

# %% [markdown]
# To test all values of `InvoiceNo`, one can do the following:

# %%
assert df.groupby(by="InvoiceNo", observed=True).InvoiceDate.nunique().eq(1).all()

# %% [markdown]
# Similarly, rows with the same `InvoiceNo` must also have the same
# `CustomerID`. Checking if this is true:

# %%
# Single value
assert df.loc[df["InvoiceNo"] == 536365, "CustomerID"].nunique() == 1

# %%
# All values
assert df.groupby(by="InvoiceNo", observed=True).CustomerID.nunique().eq(1).all()

# %% [markdown]
# Everything is OK. Then I'll compute the total amount spent for each
# `InvoiceNo`.

# %%
df_total = (
    df.assign(Price=lambda x: x.Quantity * x.UnitPrice)
    .groupby(by="InvoiceNo", observed=True)
    .agg({"InvoiceDate": "first", "CustomerID": "first", "Price": "sum"})
    .rename(columns={"Price": "TotalPrice"})
    .reset_index()
)
df_total.head()

# %%
df_total.info()

# %% [markdown]
# I'll use this code in other notebooks. For this reason, I'll define a
# function that returns the aggregated data. This function doesn't do much, but
# it is convenient.


# %%
def compute_total_price(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.assign(Price=lambda x: x.Quantity * x.UnitPrice)
        .groupby(by="InvoiceNo", observed=True)
        .agg({"InvoiceDate": "first", "CustomerID": "first", "Price": "sum"})
        .rename(columns={"Price": "TotalPrice"})
        .reset_index()
    )


# %%
def get_aggregated_data(file_path: Path) -> pd.DataFrame:
    return get_clean_data(file_path).pipe(compute_total_price)


# %%
# Quick check
df_func = get_aggregated_data(file_path)
assert_frame_equal(df_func, df_total)
del df_func

# %% [markdown]
# ## Save prepared data
# Clearly, I've ended up with a much smaller dataset than the original. To
# avoid having to repeat the above steps, I'll save the new `DataFrame` to a
# CSV file.

# %%
# Path to output CSV
out_file = file_path.with_suffix(".csv")

df_total.to_csv(out_file, index=False)
