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

# %% [markdown]
# ## Customer segmentation
#
# Next, we'll divide the customers into segments. Unfortunately, there's no
# single way to do this. More precisely, there are several ways to assign
# meaning to RFM scores. In this notebook, we'll **adapt** the approach described
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
# We'll describe each of these segments shortly. Before doing so, allow us to
# make a few comments. **The above segments are far from perfect.** For
# instance, we believe some of them are too broad. But these segments are the
# closest thing to a standard that we could find. They are also discussed
# [here](https://documentation.bloomreach.com/engagement/docs/rfm-segmentation).
# However, we've taken the liberty to redefine some segments. In our opinion,
# not all original definitions make complete sense. This doesn't mean our
# definitions are flawless. But we believe they are an improvement.
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
# These correspond to the best customers. According to any of the individual
# scores, these customers are in the top tiers. In other words, **Champions
# bought very recently, order very often, and spend the most**.

# %%
# TODO: Label customers

# %% [markdown]
# ### Loyal
#
# This segment is characterized by the following `RFMCell` values:

# %%
cells_2 = [
    "5,5,3",
    "5,4,3",
    "5,3,5",
    "4,5,3",
    "4,4,4",
    "4,4,3",
    "4,3,5",
    "3,5,5",
    "3,5,4",
    "3,4,5",
    "3,4,4",
    "3,3,5",
]

# %% [markdown]
# Notice that, in all cases, each score takes values in the range 3-5. In other
# words, we have medium to high values. Then "Loyal" customers can be described
# as follows:
# - It hasn't been that long since their last purchase.
# - They order regularly.
# - Often they spend good money.

# %%
# TODO: Label customers

# %% [markdown]
# ### Potential Loyalists
#
# This is one of the segments that we consider too broad. It corresponds to the
# following values of `RFMCell`:

# %%
cells_3 = [
    "5,5,2",
    "5,5,1",
    "5,4,2",
    "5,4,1",
    "5,3,3",
    "5,3,2",
    "5,3,1",
    "4,5,2",
    "4,5,1",
    "4,4,2",
    "4,4,1",
    "4,3,3",
    "4,3,2",
    "4,3,1",
    "3,5,3",
    "3,5,2",
    "3,5,1",
    # FIXME: Move to another segment
    # "3,4,2",
    # "3,4,1",
]

# %% [markdown]
# This segment has many cases. This is problematic, because describing it is
# complicated. But let's try. First, notice that the R scores are in the range
# 3-5. Then it's fair to say that these are recent customers. Furthermore,
# frequency values range from medium to high. We're talking about frequent
# customers. So far, we have a similar situation to that of "Loyal" customers.
# The difference is in the values for the M scores. For this segment, these
# scores are in the 1-3 range. Therefore, "Potential Loyalists" can be
# described as follows: **recent, frequent customers with medium to low
# monetary value**.

# %%
# TODO: Label customers

# %% [markdown]
# ### New Customers
#
# For this segment, the `RFMCell` values are

# %%
cells_4 = [
    "5,2,2",
    "5,2,1",
    "5,1,2",
    "5,1,1",
    "4,2,2",
    "4,2,1",
    "4,1,2",
    "4,1,1",
    # FIXME: Move to another segment
    # "3,1,1",
]

# %% [markdown]
# By inspecting the above list, one can see that basically we have high
# recency, and low frequency and monetary values. Therefore, the name of this
# segment is appropriate. No further explanation is required.

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
    "5,1,5",
    "5,1,4",
    "5,1,3",
    "4,2,5",
    "4,2,4",
    "4,2,3",
    "4,1,5",
    "4,1,4",
    "4,1,3",
    "3,1,5",
    "3,1,4",
    "3,1,3",
]

# %% [markdown]
# In this case, we have medium to high values for the R and M scores. So these
# are recent customers who spent a good amount. However, the frequency values
# are low. These customers didn't buy many times. Therefore, they're
# "Promising" in the following sense: **if they make more purchases, they'll be
# among the best customers**.

# %%
# TODO: Label customers

# %% [markdown]
# ### Need Attention
#
# TODO

# %%
cells_6 = [
    "3,3,4",
    "3,3,3",
    "3,2,5",
    "3,2,4",
    "3,2,3",
    # FIXME: Move to another segment
    # "5,3,4",
    # "4,3,4",
    # "3,4,3",
]
