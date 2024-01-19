# %% [markdown]
# # RFM Analysis: Example
#
# To figure out how to implement RFM analysis, I'll first consider a fake
# dataset. This data was taken from this [blog post](https://clevertap.com/blog/rfm-analysis/).
# Here the goal is to reproduce the results presented in that post.
#
# ## Imports

# %%
from typing import cast

import numpy as np
import pandas as pd

# %% [markdown]
# ## Fake dataset
#
# For comparison, here's the original dataset:
#
# ![Fake dataset](./rfm_example_1.png)
#
# Recreating this dataset using pandas:

# %%
df = pd.DataFrame(
    data={
        "Recency": [4, 6, 46, 23, 15, 32, 7, 50, 34, 10, 3, 1, 27, 18, 5],
        "Frequency": [6, 11, 1, 3, 4, 2, 3, 1, 15, 5, 8, 10, 3, 2, 1],
        "Monetary": [540, 940, 35, 65, 179, 56, 140, 950, 2630, 191, 845, 1510, 54, 40, 25],
    },
    index=list(range(1, 16)),
)
df.index.name = "CustomerId"
df

# %% [markdown]
# ## R score
#
# We begin by calculating the R score. For comparison, these are the results
# we're trying to reproduce:
#
# ![R scores](./rfm_example_2.png)
#
# Notice that the rank on the fourth row is wrong. Recreating the above table
# using pandas:

# %%
df_r = df.loc[:, ["Recency"]]
df_r = cast(pd.DataFrame, df_r)

# %%
# Use `Recency` to compute the rank
df_r["Rank"] = df_r["Recency"].rank().astype(np.int_)
df_r = df_r.sort_values(by="Rank")
df_r

# %%
# Use `Rank` to compute the R score
NUM_BINS = 5
SCORE_LABELS = list(range(NUM_BINS, 0, -1))

df_r["RScore"] = pd.cut(df_r["Rank"], NUM_BINS, labels=SCORE_LABELS)
df_r
