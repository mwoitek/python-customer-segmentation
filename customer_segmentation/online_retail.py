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
# File path for dataset
file_path = Path.cwd() / "data" / "online_retail.xlsx"
assert file_path.exists(), "file doesn't exist"
assert file_path.is_file(), "not a file"

# %%
# Columns I'll actually use
cols = ["InvoiceNo", "InvoiceDate", "CustomerID", "Quantity", "UnitPrice"]

df = pd.read_excel(
    file_path,
    usecols=cols,
    dtype={col: object for col in cols},
).loc[:, cols]
df = cast(pd.DataFrame, df)

# %%
df.head(10)

# %%
df.info()

# %% [markdown]
# ## Data cleaning

# %%
# Missing values
df.isna().sum()

# %% [markdown]
# I really need to know who bought what. In other words, rows with missing
# `CustomerID` have to go.

# %%
df = df.dropna()
df.info()

# %%
# Look for invalid quantities
(df["Quantity"] <= 0).sum()

# %% [markdown]
# Not every row corresponds to a sale. When the invoice number starts with "C",
# that transaction was canceled. That explains the observations with
# non-positive quantities.

# %%
df["InvoiceNo"].astype(str).str.startswith("C").sum()

# %% [markdown]
# I chose to remove those rows:

# %%
df = df.loc[df["Quantity"] > 0, :]
df = cast(pd.DataFrame, df)

# Quick check
assert df["InvoiceNo"].astype(str).str.startswith("C").sum() == 0, "there are remaining canceled transactions"

# %%
# Look for invalid prices
(df["UnitPrice"] == 0.0).sum()

# %% [markdown]
# I don't know how to explain such values. They should make no difference. Then
# I chose to drop them:

# %%
df = df.loc[df["UnitPrice"] != 0.0, :]
df = cast(pd.DataFrame, df)

# %% [markdown]
# ## Finish fixing dataset

# %%
df = df.reset_index(drop=True)
df.info()

# %%
# Use appropriate data types
df["InvoiceNo"] = df["InvoiceNo"].astype("category")
df["CustomerID"] = df["CustomerID"].astype("category")
df["Quantity"] = df["Quantity"].astype(np.int_)
df["UnitPrice"] = df["UnitPrice"].astype(np.float_)

df.dtypes

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
df.loc[df["InvoiceNo"] == 536365, "InvoiceDate"].nunique() == 1

# %% [markdown]
# To test all values of `InvoiceNo`, one can do the following:

# %%
df.groupby(by="InvoiceNo", observed=True).InvoiceDate.nunique().eq(1).all()

# %% [markdown]
# Similarly, rows with the same `InvoiceNo` must also have the same
# `CustomerID`. Checking if this is true:

# %%
# Single value
df.loc[df["InvoiceNo"] == 536365, "CustomerID"].nunique() == 1

# %%
# All values
df.groupby(by="InvoiceNo", observed=True).CustomerID.nunique().eq(1).all()

# %% [markdown]
# Everything is OK. Then I'll compute the total amount spent for each
# `InvoiceNo`.

# %%
# Figuring out how to do it
tmp_df = df.loc[df["InvoiceNo"] == 536365, :]
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
def compute_total_price(df_group: pd.DataFrame) -> pd.Series:
    return pd.Series(
        data={
            "InvoiceDate": df_group["InvoiceDate"].iloc[0],
            "CustomerID": df_group["CustomerID"].iloc[0],
            "TotalPrice": (df_group["Quantity"] * df_group["UnitPrice"]).sum(),
        }
    )


df_total = df.groupby(by="InvoiceNo", observed=True).apply(compute_total_price).reset_index()

# %%
df_total.head()

# %%
df_total.info()

# %% [markdown]
# ## Save prepared data
#
# Clearly, I've ended up with a much smaller dataset than the original. To
# avoid having to repeat the above steps, I'll save the new `DataFrame` to a
# CSV file.

# %%
# File path for output CSV
out_file = file_path.parent / "online_retail.csv"

df_total.to_csv(out_file, index=False)
