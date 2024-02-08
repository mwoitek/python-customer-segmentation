# %% [markdown]
# # Online Retail Dataset: Data Preparation
#
# In this notebook, I'll prepare the dataset for analysis.
#
# ## Imports

# %%
from pathlib import Path
from typing import cast

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
df = cast(pd.DataFrame, df)
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
df = cast(pd.DataFrame, df)
df.info()

# %% [markdown]
# `CustomerID` is supposed to be a 5-digit integer number. Let's confirm that
# this is the case:

# %%
ids = df.CustomerID.astype(str)
ids = cast(pd.Series, ids)
assert ids.str.len().eq(5).all()
assert ids.str.isdigit().all()
del ids

# %%
# Use appropriate data type
df.CustomerID = df.CustomerID.astype("category")

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

# %% [markdown]
# I chose to remove those rows:

# %%
df = df[df.Quantity > 0]
df = cast(pd.DataFrame, df)

# %% [markdown]
# `InvoiceNo` is supposed to be a 6-digit integer number. Let's check that this
# column is OK:

# %%
invoice_nums = df.InvoiceNo.astype(str)
assert invoice_nums.str.startswith("C").sum() == 0
assert invoice_nums.str.len().eq(6).all()
assert invoice_nums.str.isdigit().all()
del invoice_nums

# %%
# Use appropriate data type
df.InvoiceNo = df.InvoiceNo.astype("category")

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
df = cast(pd.DataFrame, df)

# %% [markdown]
# ## Finish fixing dataset

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
    ).loc[:, cols]
    df = df[df.CustomerID.notna()]
    df.CustomerID = df.CustomerID.astype("category")
    df = df[df.Quantity > 0]
    df.InvoiceNo = df.InvoiceNo.astype("category")
    df = df[df.UnitPrice > 0.0]
    df = df.reset_index(drop=True)
    df.InvoiceDate = df.InvoiceDate.dt.normalize()
    return df


# %%
# Quick check
df_func = get_clean_data(file_path)
assert_frame_equal(df_func, df)
del df_func

# %% [markdown]
# ## Aggregate data
#
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
# Figuring out how to do it
tmp_df = df[df.InvoiceNo == 536365]
tmp_df = cast(pd.DataFrame, tmp_df)
tmp_df

# %%
tmp_row = pd.Series(
    data={
        "InvoiceDate": tmp_df["InvoiceDate"].iloc[0],
        "CustomerID": tmp_df["CustomerID"].iloc[0],
        "TotalPrice": (tmp_df["Quantity"] * tmp_df["UnitPrice"]).sum(),
    }
)
tmp_row

# %%
del tmp_df
del tmp_row


# %%
# Actual calculation
def compute_total_price(df: pd.DataFrame) -> pd.Series:
    return pd.Series(
        data={
            "InvoiceDate": df["InvoiceDate"].iloc[0],
            "CustomerID": df["CustomerID"].iloc[0],
            "TotalPrice": (df["Quantity"] * df["UnitPrice"]).sum(),
        }
    )


# %%
df_total = (
    df.groupby(by="InvoiceNo", observed=True).apply(compute_total_price, include_groups=False).reset_index()
)
df_total.head()

# %%
df_total.info()

# %% [markdown]
# I'll use this code in other notebooks. For this reason, I'll define a
# function that returns the aggregated data. This function doesn't do much, but
# it is convenient.


# %%
def get_aggregated_data(file_path: Path) -> pd.DataFrame:
    return (
        get_clean_data(file_path)
        .groupby(by="InvoiceNo", observed=True)
        .apply(compute_total_price, include_groups=False)
        .reset_index()
    )


# %%
# Quick check
df_func = get_aggregated_data(file_path)
assert_frame_equal(df_func, df_total)
del df_func

# %% [markdown]
# ## Save prepared data
#
# Clearly, I've ended up with a much smaller dataset than the original. To
# avoid having to repeat the above steps, I'll save the new `DataFrame` to a
# CSV file.

# %%
# Path to output CSV
out_file = file_path.with_suffix(".csv")

df_total.to_csv(out_file, index=False)
