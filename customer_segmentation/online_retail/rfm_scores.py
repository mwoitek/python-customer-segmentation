# %% [markdown]
# # Online Retail Dataset: RFM Scores
#
# In this notebook, I'll use the prepared dataset to compute the RFM scores.
#
# ## Imports

# %%
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd

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
