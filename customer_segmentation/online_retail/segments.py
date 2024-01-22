# %% [markdown]
# # Online Retail Dataset: Customer Segmentation
#
# In this notebook, we'll perform the actual customer segmentation. More
# precisely, we'll use our results for the RFM scores to divide the customers
# into meaningful segments.
#
# ## Imports

# %%
from pathlib import Path

import numpy as np
import pandas as pd

# %% [markdown]
# ## Read RFM scores

# %%
# File path for dataset
file_path = Path.cwd().parents[1] / "data" / "rfm_scores.csv"
assert file_path.exists(), f"file doesn't exist: {file_path}"
assert file_path.is_file(), f"not a file: {file_path}"

# %%
df_rfm = pd.read_csv(
    file_path,
    dtype={
        "CustomerID": "category",
        "Recency": np.int_,
        "Frequency": np.int_,
        "Monetary": np.float_,
        "RScore": "category",
        "FScore": "category",
        "MScore": "category",
        "RFMCell": object,
        "RFMScore": np.float_,
    },
)
df_rfm.head(15)

# %%
df_rfm.info()

# %%
# Categorical columns aren't ordered
df_rfm["RScore"].cat.ordered

# %%
# Fix categorical columns
score_cols = ["RScore", "FScore", "MScore"]
df_rfm.loc[:, score_cols] = df_rfm.loc[:, score_cols].transform(lambda col: col.cat.as_ordered())
del score_cols

# %%
# Quick check
df_rfm["RScore"].head()
