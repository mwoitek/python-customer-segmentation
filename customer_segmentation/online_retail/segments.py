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
from typing import cast

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

# %% [markdown]
# ## Customer segmentation
#
# Next, we'll divide the customers into segments. Unfortunately, there's no
# single way to do this. More precisely, there are several ways to assign
# meaning to RFM scores. In this notebook, we'll adopt the approach described
# in this [blog post](https://nealanalytics.com/blog/customer-segmentation-using-rfm-analysis/).
#
# Specifically, we'll consider the following 11 segments:
# - Champions
# - Loyal
# - Potential Loyalists
# - New Customers
# - Promising
# - Need Attention
# - About to Sleep
# - Cannot Lose Them
# - At Risk
# - Hibernating
# - Lost
#
# For a description of these segments, see the blog post mentioned above.
#
# Before continuing, we need to make a few comments. **These segments are far
# from perfect.** For instance, we believe some of them are too broad. But
# they're the closest thing to a standard that we could find. They're also
# discussed in these articles:
# - [https://documentation.bloomreach.com/engagement/docs/rfm-segmentation](https://documentation.bloomreach.com/engagement/docs/rfm-segmentation)
# - [https://shopup.me/model/rfm-segmentation/](https://shopup.me/model/rfm-segmentation/)
# - [https://www.dase-analytics.com/blog/en/rfm-analysis/](https://www.dase-analytics.com/blog/en/rfm-analysis/)
#
# ### Champions
#
# Each segment is characterized by a set of values for `RFMCell`. In the case
# of "Champions", these values are as follows:

# %%
cells_1 = [
    "5,5,5",
    "5,5,4",
    "5,4,5",
    "5,4,4",
    "4,5,5",
    "4,5,4",
    "4,4,5",
]

# %%
# TODO: Label customers

# %% [markdown]
# ### Loyal
#
# This segment is characterized by the following `RFMCell` values:

# %%
cells_2 = [
    "5,4,3",
    "4,4,4",
    "4,3,5",
    "3,5,5",
    "3,5,4",
    "3,4,5",
    "3,4,4",
    "3,3,5",
]

# %%
# TODO: Label customers

# %% [markdown]
# ### Potential Loyalists
#
# This segment corresponds to the following values of `RFMCell`:

# %%
cells_3 = [
    "5,5,3",
    "5,5,2",
    "5,5,1",
    "5,4,2",
    "5,4,1",
    "5,3,3",
    "5,3,2",
    "5,3,1",
    "4,5,3",
    "4,5,2",
    "4,5,1",
    "4,4,2",
    "4,4,1",
    "4,3,3",
    "4,3,2",
    "4,3,1",
    "4,2,3",
    "3,5,3",
    "3,5,2",
    "3,5,1",
    "3,4,2",
    "3,4,1",
    "3,3,3",
    "3,2,3",
]

# %%
# TODO: Label customers

# %% [markdown]
# ### New Customers
#
# For this segment, the `RFMCell` values are

# %%
cells_4 = [
    "5,1,2",
    "5,1,1",
    "4,2,2",
    "4,2,1",
    "4,1,2",
    "4,1,1",
    "3,1,1",
]

# %%
# TODO: Label customers

# %% [markdown]
# ### Promising
#
# "Promising" customers are characterized by the following values of `RFMCell`:

# %%
cells_5 = [
    "5,2,5",
    "5,2,4",
    "5,2,3",
    "5,2,2",
    "5,2,1",
    "5,1,5",
    "5,1,4",
    "5,1,3",
    "4,2,5",
    "4,2,4",
    "4,1,5",
    "4,1,4",
    "4,1,3",
    "3,1,5",
    "3,1,4",
    "3,1,3",
]

# %%
# TODO: Label customers

# %% [markdown]
# ### Need Attention
#
# In this case, the values of `RFMCell` are the following:

# %%
cells_6 = [
    "5,3,5",
    "5,3,4",
    "4,4,3",
    "4,3,4",
    "3,4,3",
    "3,3,4",
    "3,2,5",
    "3,2,4",
]

# %%
# TODO: Label customers

# %% [markdown]
# ### About to Sleep
#
# For this segment, the list of characteristic values is

# %%
cells_7 = [
    "3,3,1",
    "3,2,1",
    "3,1,2",
    "2,2,1",
    "2,1,3",
]

# %%
# TODO: Label customers

# %% [markdown]
# ### Cannot Lose Them
#
# The `RFMCell` values for the next segment are

# %%
cells_8 = [
    "2,1,5",
    "2,1,4",
    "1,5,5",
    "1,5,4",
    "1,4,4",
    "1,1,5",
    "1,1,4",
    "1,1,3",
]

# %%
# TODO: Label customers

# %% [markdown]
# ### At Risk
#
# This segment is characterized by the values below:

# %%
cells_9 = [
    "2,5,5",
    "2,5,4",
    "2,5,3",
    "2,5,2",
    "2,4,5",
    "2,4,4",
    "2,4,3",
    "2,4,2",
    "2,3,5",
    "2,3,4",
    "2,2,5",
    "2,2,4",
    "1,5,3",
    "1,5,2",
    "1,4,5",
    "1,4,3",
    "1,4,2",
    "1,3,5",
    "1,3,4",
    "1,3,3",
    "1,2,5",
    "1,2,4",
]

# %%
# TODO: Label customers

# %% [markdown]
# ### Hibernating
#
# For this segment, the values of `RFMCell` are

# %%
cells_10 = [
    "3,3,2",
    "3,2,2",
    "2,5,1",
    "2,4,1",
    "2,3,3",
    "2,3,2",
    "2,3,1",
    "2,2,3",
    "2,2,2",
    "2,1,2",
    "2,1,1",
    "1,3,2",
    "1,2,3",
    "1,2,2",
]

# %%
# TODO: Label customers

# %% [markdown]
# ### Lost
#
# Finally, the last segment is characterized by

# %%
cells_11 = [
    "1,5,1",
    "1,4,1",
    "1,3,1",
    "1,2,1",
    "1,1,2",
    "1,1,1",
]

# %%
# TODO: Label customers

# %% [markdown]
# ### Consistency check
#
# Checking if all values of `RFMCell` have been used:

# %%
# Join all lists I've defined
all_cells: list[str] = []
gl = globals()

for i in range(1, 12):
    cells = gl[f"cells_{i}"]
    cells = cast(list[str], cells)
    all_cells.extend(cells)

del gl

all_cells.sort(reverse=True)
all_cells[:20]

# %%
# Generate all possible values
vals = list(range(1, 6))
df_1 = pd.DataFrame(data={"R": vals})
df_2 = pd.DataFrame(data={"F": vals})
df_3 = pd.DataFrame(data={"M": vals})

df_tmp = (
    df_1.merge(df_2, how="cross")
    .merge(df_3, how="cross")
    .sort_values(by=["R", "F", "M"], ascending=False)
    .reset_index(drop=True)
)
df_tmp["RFMCell"] = df_tmp.agg(lambda r: f"{r.iloc[0]},{r.iloc[1]},{r.iloc[2]}", axis="columns")

del vals
del df_1
del df_2
del df_3

df_tmp.head(20)

# %%
# Finally check
assert all_cells == df_tmp["RFMCell"].to_list(), "not all values of RFMCell have been used"

del all_cells
del df_tmp
