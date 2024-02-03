# %% [markdown]
# # Online Retail Dataset: k-means Clustering
#
# In this notebook, I'll perform customer segmentation using the k-means
# clustering algorithm.
#
# ## Imports

# %%
from pathlib import Path

import numpy as np
import pandas as pd

# %% [markdown]
# ## Read features
#
# In another notebook, we obtained the features we'll use here. Next, we'll
# read those results.

# %%
# Path to features CSV
file_path = Path.cwd().parents[1] / "data" / "clustering_features.csv"
assert file_path.exists(), f"file doesn't exist: {file_path}"
assert file_path.is_file(), f"not a file: {file_path}"

# %%
# Read features CSV
df = pd.read_csv(
    file_path,
    dtype={
        "CustomerID": "category",
        "Recency": np.int_,
        "Frequency": np.int_,
        "AvgSpent": np.float_,
        "PTRecency": np.float_,
        "PTFrequency": np.float_,
        "PTAvgSpent": np.float_,
        "QTRecency": np.float_,
        "QTFrequency": np.float_,
        "QTAvgSpent": np.float_,
    },
    index_col="CustomerID",
)
df.head()

# %%
df.info()
