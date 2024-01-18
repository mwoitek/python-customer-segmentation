# %% [markdown]
# # Online Retail Dataset

# %% [markdown]
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
