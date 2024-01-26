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
from pandas.testing import assert_series_equal

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

# %% [markdown]
# Labeling customers:

# %%
df_rfm["Segment"] = ""

mask = df_rfm["RFMCell"].isin(cells_1)
df_rfm.loc[mask, "Segment"] = "Champions"
df_rfm.loc[mask, :].head()

# %% [markdown]
# **NOTE**: This isn't the best way to label customers. After introducing all
# segments, we'll solve this problem more efficiently.

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

# %% [markdown]
# Labeling customers:

# %%
mask = df_rfm["RFMCell"].isin(cells_2)
df_rfm.loc[mask, "Segment"] = "Loyal"
df_rfm.loc[mask, :].head()

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

# %% [markdown]
# Labeling customers:

# %%
mask = df_rfm["RFMCell"].isin(cells_3)
df_rfm.loc[mask, "Segment"] = "Potential Loyalists"
df_rfm.loc[mask, :].head()

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

# %% [markdown]
# Labeling customers:

# %%
mask = df_rfm["RFMCell"].isin(cells_4)
df_rfm.loc[mask, "Segment"] = "New Customers"
df_rfm.loc[mask, :].head()

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

# %% [markdown]
# Labeling customers:

# %%
mask = df_rfm["RFMCell"].isin(cells_5)
df_rfm.loc[mask, "Segment"] = "Promising"
df_rfm.loc[mask, :].head()

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

# %% [markdown]
# Labeling customers:

# %%
mask = df_rfm["RFMCell"].isin(cells_6)
df_rfm.loc[mask, "Segment"] = "Need Attention"
df_rfm.loc[mask, :].head()

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

# %% [markdown]
# Labeling customers:

# %%
mask = df_rfm["RFMCell"].isin(cells_7)
df_rfm.loc[mask, "Segment"] = "About to Sleep"
df_rfm.loc[mask, :].head()

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

# %% [markdown]
# Labeling customers:

# %%
mask = df_rfm["RFMCell"].isin(cells_8)
df_rfm.loc[mask, "Segment"] = "Cannot Lose Them"
df_rfm.loc[mask, :].head()

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

# %% [markdown]
# Labeling customers:

# %%
mask = df_rfm["RFMCell"].isin(cells_9)
df_rfm.loc[mask, "Segment"] = "At Risk"
df_rfm.loc[mask, :].head()

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

# %% [markdown]
# Labeling customers:

# %%
mask = df_rfm["RFMCell"].isin(cells_10)
df_rfm.loc[mask, "Segment"] = "Hibernating"
df_rfm.loc[mask, :].head()

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

# %% [markdown]
# Labeling customers:

# %%
mask = df_rfm["RFMCell"].isin(cells_11)
df_rfm.loc[mask, "Segment"] = "Lost"
df_rfm.loc[mask, :].head()

# %%
del mask

# %% [markdown]
# ### Consistency checks
#
# Confirm that every customer has been labeled:

# %%
assert (df_rfm["Segment"].str.len() > 0).all(), "there are unlabeled customers"

# %% [markdown]
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

# %% [markdown]
# ### Better way to label customers

# %%
# Define all segments at once
SEGMENTS_5 = {
    "5,5,5": "Champions",
    "5,5,4": "Champions",
    "5,4,5": "Champions",
    "5,4,4": "Champions",
    "4,5,5": "Champions",
    "4,5,4": "Champions",
    "4,4,5": "Champions",
    "5,4,3": "Loyal",
    "4,4,4": "Loyal",
    "4,3,5": "Loyal",
    "3,5,5": "Loyal",
    "3,5,4": "Loyal",
    "3,4,5": "Loyal",
    "3,4,4": "Loyal",
    "3,3,5": "Loyal",
    "5,5,3": "Potential Loyalists",
    "5,5,2": "Potential Loyalists",
    "5,5,1": "Potential Loyalists",
    "5,4,2": "Potential Loyalists",
    "5,4,1": "Potential Loyalists",
    "5,3,3": "Potential Loyalists",
    "5,3,2": "Potential Loyalists",
    "5,3,1": "Potential Loyalists",
    "4,5,3": "Potential Loyalists",
    "4,5,2": "Potential Loyalists",
    "4,5,1": "Potential Loyalists",
    "4,4,2": "Potential Loyalists",
    "4,4,1": "Potential Loyalists",
    "4,3,3": "Potential Loyalists",
    "4,3,2": "Potential Loyalists",
    "4,3,1": "Potential Loyalists",
    "4,2,3": "Potential Loyalists",
    "3,5,3": "Potential Loyalists",
    "3,5,2": "Potential Loyalists",
    "3,5,1": "Potential Loyalists",
    "3,4,2": "Potential Loyalists",
    "3,4,1": "Potential Loyalists",
    "3,3,3": "Potential Loyalists",
    "3,2,3": "Potential Loyalists",
    "5,1,2": "New Customers",
    "5,1,1": "New Customers",
    "4,2,2": "New Customers",
    "4,2,1": "New Customers",
    "4,1,2": "New Customers",
    "4,1,1": "New Customers",
    "3,1,1": "New Customers",
    "5,2,5": "Promising",
    "5,2,4": "Promising",
    "5,2,3": "Promising",
    "5,2,2": "Promising",
    "5,2,1": "Promising",
    "5,1,5": "Promising",
    "5,1,4": "Promising",
    "5,1,3": "Promising",
    "4,2,5": "Promising",
    "4,2,4": "Promising",
    "4,1,5": "Promising",
    "4,1,4": "Promising",
    "4,1,3": "Promising",
    "3,1,5": "Promising",
    "3,1,4": "Promising",
    "3,1,3": "Promising",
    "5,3,5": "Need Attention",
    "5,3,4": "Need Attention",
    "4,4,3": "Need Attention",
    "4,3,4": "Need Attention",
    "3,4,3": "Need Attention",
    "3,3,4": "Need Attention",
    "3,2,5": "Need Attention",
    "3,2,4": "Need Attention",
    "3,3,1": "About to Sleep",
    "3,2,1": "About to Sleep",
    "3,1,2": "About to Sleep",
    "2,2,1": "About to Sleep",
    "2,1,3": "About to Sleep",
    "2,1,5": "Cannot Lose Them",
    "2,1,4": "Cannot Lose Them",
    "1,5,5": "Cannot Lose Them",
    "1,5,4": "Cannot Lose Them",
    "1,4,4": "Cannot Lose Them",
    "1,1,5": "Cannot Lose Them",
    "1,1,4": "Cannot Lose Them",
    "1,1,3": "Cannot Lose Them",
    "2,5,5": "At Risk",
    "2,5,4": "At Risk",
    "2,5,3": "At Risk",
    "2,5,2": "At Risk",
    "2,4,5": "At Risk",
    "2,4,4": "At Risk",
    "2,4,3": "At Risk",
    "2,4,2": "At Risk",
    "2,3,5": "At Risk",
    "2,3,4": "At Risk",
    "2,2,5": "At Risk",
    "2,2,4": "At Risk",
    "1,5,3": "At Risk",
    "1,5,2": "At Risk",
    "1,4,5": "At Risk",
    "1,4,3": "At Risk",
    "1,4,2": "At Risk",
    "1,3,5": "At Risk",
    "1,3,4": "At Risk",
    "1,3,3": "At Risk",
    "1,2,5": "At Risk",
    "1,2,4": "At Risk",
    "3,3,2": "Hibernating",
    "3,2,2": "Hibernating",
    "2,5,1": "Hibernating",
    "2,4,1": "Hibernating",
    "2,3,3": "Hibernating",
    "2,3,2": "Hibernating",
    "2,3,1": "Hibernating",
    "2,2,3": "Hibernating",
    "2,2,2": "Hibernating",
    "2,1,2": "Hibernating",
    "2,1,1": "Hibernating",
    "1,3,2": "Hibernating",
    "1,2,3": "Hibernating",
    "1,2,2": "Hibernating",
    "1,5,1": "Lost",
    "1,4,1": "Lost",
    "1,3,1": "Lost",
    "1,2,1": "Lost",
    "1,1,2": "Lost",
    "1,1,1": "Lost",
}

# %%
# Remove "old" segments
old_segments = df_rfm.pop("Segment")

df_rfm.info()

# %%
# Label customers
df_rfm["Segment"] = df_rfm["RFMCell"].map(SEGMENTS_5)

# %%
# Quick check
assert_series_equal(df_rfm["Segment"], old_segments)

del old_segments
