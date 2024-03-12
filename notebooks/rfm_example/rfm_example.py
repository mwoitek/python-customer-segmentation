# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # RFM Analysis: Example
# To figure out how to implement RFM analysis, I'll first consider a fake
# dataset. This data was taken from this [blog post](https://clevertap.com/blog/rfm-analysis/).
# Here the goal is to reproduce the results presented in that post.
# ## Imports

# %%
from typing import Literal, get_args

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

# %% [markdown]
# ## Fake dataset
# For comparison, here's the original dataset:
#
# <img src="rfm_example_1.png" style="width: 60%;"/>
#
# Recreating this dataset using pandas:

# %%
df = pd.DataFrame(
    data={
        "Recency": [4, 6, 46, 23, 15, 32, 7, 50, 34, 10, 3, 1, 27, 18, 5],
        "Frequency": [6, 11, 1, 3, 4, 2, 3, 1, 15, 5, 8, 10, 3, 2, 1],
        "Monetary": [540, 940, 35, 65, 179, 56, 140, 950, 2630, 191, 845, 1510, 54, 40, 25],
    },
    index=pd.Index(data=list(range(1, 16)), name="CustomerId"),
)
df

# %% [markdown]
# ## R score
# We begin by calculating the R score. For comparison, these are the results
# we're trying to reproduce:
#
# <img src="rfm_example_2.png" style="width: 60%;"/>
#
# Notice that **the rank on the fourth row is wrong**. Recreating the above
# table using pandas:

# %%
NUM_BINS = 5
SCORE_LABELS = list(range(NUM_BINS, 0, -1))

# %%
# Use `Recency` to compute the rank
df_r = df[["Recency"]].assign(Rank=lambda x: x.Recency.rank().astype(np.int_)).sort_values(by="Rank")
df_r

# %%
# Use `Rank` to compute the R score
df_r = df_r.assign(
    RScore=pd.cut(df_r["Rank"], NUM_BINS, labels=SCORE_LABELS).cat.reorder_categories(
        SCORE_LABELS[::-1], ordered=True
    )
)
df_r

# %% [markdown]
# Notice that these results agree with the reference values.
# ## F score and M score
# The next step is to calculate the F and M scores. To do so, I'll generalize
# the code above. The following function can be used to compute all scores:

# %%
RFMAttribute = Literal["Recency", "Frequency", "Monetary"]


def compute_score(df: pd.DataFrame, attr: RFMAttribute, num_bins: int = 5) -> pd.DataFrame:
    score_name = f"{attr[0]}Score"
    score_labels = list(range(num_bins, 0, -1)) if attr == "Recency" else list(range(1, num_bins + 1))

    df_score = (
        df[[attr]]
        .assign(Rank=lambda x: x[attr].rank(method="min").astype(np.int_))
        .sort_values(by="Rank", ascending=attr == "Recency")
    )

    df_score[score_name] = pd.cut(df_score["Rank"], num_bins, labels=score_labels)
    if attr == "Recency":
        df_score[score_name] = df_score[score_name].cat.reorder_categories(score_labels[::-1], ordered=True)

    return df_score.drop(columns="Rank")


# %%
# As a test, we'll re-calculate the R scores:
df_r_func = compute_score(df, "Recency")
assert_frame_equal(df_r_func, df_r.drop(columns="Rank"))

# %%
# Checking that categories are ordered correctly
df_r_func["RScore"]

# %%
del df_r_func

# %% [markdown]
# Comparing this function with the previous code, one should notice that the
# most important change is related to the score labels. Basically, this has to
# do with the desirable values for the different RFM attributes.
#
# In the case of recency, we want small values (customer purchased recently).
# These values are assigned the best R scores. We have the opposite for
# frequency and monetary. For these attributes, we want high values (customer
# buys frequently/customer spends a lot of money). These cases are assigned the
# best F and M scores. The above function takes this difference into account.
#
# Finally, let's compute the F and M scores. For comparison, these are the
# values we want to obtain:
#
# <img src="rfm_example_3.png" style="width: 80%;"/>
#
# Calculating the F score:

# %%
df_f = compute_score(df, "Frequency")
df_f

# %% [markdown]
# Note that the results above are different from the reference values.
# Specifically, there's a difference when the frequency value isn't unique. It
# makes more sense that in such cases customers get the same F score. Our
# results obey this rule. However, those in the original post don't. Then our
# approach isn't wrong. It's better.

# %%
# Checking that categories are ordered correctly
df_f["FScore"]

# %% [markdown]
# Calculating the M score:

# %%
df_m = compute_score(df, "Monetary")
df_m

# %% [markdown]
# Notice that these results agree with the reference values.
# ## RFM score
# Next, we'll combine the results obtained above to calculate the RFM scores.
# More precisely, we'll reproduce the following table:
#
# <img src="rfm_example_4.png" style="width: 50%;"/>
#
# As already explained, our results for the F score are different. For this
# reason, two rows in this table won't be reproduced exactly. But our values
# will be very close.

# %%
# Concatenate scores from different DataFrames
df_rfm = pd.concat([df_r["RScore"], df_f["FScore"], df_m["MScore"]], axis=1).sort_index()
df_rfm

# %%
# Compute RFM cells
df_rfm["RFMCell"] = df_rfm.agg(lambda r: f"{r.iloc[0]},{r.iloc[1]},{r.iloc[2]}", axis="columns")
df_rfm

# %%
# Compute RFM scores
df_rfm["RFMScore"] = df_rfm.iloc[:, :3].astype(np.int_).agg("mean", axis="columns")
df_rfm[["RFMCell", "RFMScore"]].round(decimals={"RFMScore": 1})

# %% [markdown]
# Notice that these results agree with the reference values, except for the
# rows with `CustomerId` 7 and 13. As expected, the difference is in the F
# score values.
#
# Now I'm confident that my implementation is correct. So I'll collect the
# essential parts of the code above, and write a function that adds the RFM
# scores to the original DataFrame.


# %%
def add_score_column(df: pd.DataFrame, attr: RFMAttribute, num_bins: int = 5) -> pd.DataFrame:
    score_name = f"{attr[0]}Score"
    score_labels = list(range(num_bins, 0, -1)) if attr == "Recency" else list(range(1, num_bins + 1))

    rank = df[attr].rank(method="min").astype(np.int_)
    df[score_name] = pd.cut(rank, num_bins, labels=score_labels)
    if attr == "Recency":
        df[score_name] = df[score_name].cat.reorder_categories(score_labels[::-1], ordered=True)

    return df


# %%
def add_rfm_scores(df: pd.DataFrame, num_bins: int = 5) -> pd.DataFrame:
    for attr in get_args(RFMAttribute):
        df = add_score_column(df, attr, num_bins)

    score_cols = [f"{attr[0]}Score" for attr in get_args(RFMAttribute)]
    df["RFMCell"] = df[score_cols].agg(lambda r: f"{r.iloc[0]},{r.iloc[1]},{r.iloc[2]}", axis="columns")
    df["RFMScore"] = df[score_cols].astype(np.int_).agg("mean", axis="columns")

    return df


# %%
# Quick test
df = add_rfm_scores(df)
assert_frame_equal(df[["RFMCell", "RFMScore"]], df_rfm[["RFMCell", "RFMScore"]])

# %%
df
