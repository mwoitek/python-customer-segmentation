# %% [markdown]
# # Online Retail Dataset: RFM Scores
#
# In this notebook, I'll use the prepared dataset to compute the RFM scores.
#
# ## Imports

# %%
from pathlib import Path

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
