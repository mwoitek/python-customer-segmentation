# %% [markdown]
# # RFM Analysis: Example
#
# To figure out how to implement RFM analysis, I'll first consider a fake
# dataset. This data was taken from this [blog post](https://clevertap.com/blog/rfm-analysis/).
# Here the goal is to reproduce the results presented in that post.
#
# ## Imports

# %%
from typing import Literal, cast

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

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
df_r["RScore"] = df_r["RScore"].cat.reorder_categories(SCORE_LABELS[::-1], ordered=True)
df_r

# %% [markdown]
# Notice that these results agree with the reference values.

# %% [markdown]
# ## F score and M score
#
# The next step is to calculate the F and M scores. To do so, I'll generalize
# the code above. The following function can be used to compute all scores:

# %%
RFMAttribute = Literal["Recency", "Frequency", "Monetary"]


def compute_score(
    df: pd.DataFrame,
    attr: RFMAttribute,
    num_bins: int = 5,
) -> pd.DataFrame:
    df_score = df.loc[:, [attr]]
    df_score = cast(pd.DataFrame, df_score)

    df_score["Rank"] = df_score[attr].rank(method="min").astype(np.int_)
    df_score = df_score.sort_values(by="Rank", ascending=attr == "Recency")

    score_name = f"{attr[0]}Score"
    score_labels = list(range(num_bins, 0, -1)) if attr == "Recency" else list(range(1, num_bins + 1))
    df_score[score_name] = pd.cut(df_score["Rank"], num_bins, labels=score_labels)
    if attr == "Recency":
        df_score[score_name] = df_score[score_name].cat.reorder_categories(score_labels[::-1], ordered=True)

    df_score = df_score.drop(columns="Rank")
    return df_score


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
# As a test, we'll re-calculate the R scores:

# %%
df_r_func = compute_score(df, "Recency")
assert_frame_equal(df_r_func, df_r.drop(columns="Rank"))
df_r_func

# %%
# Checking that categories are ordered correctly
df_r_func["RScore"]

# %%
del df_r_func

# %% [markdown]
# Finally, let's compute the F and M scores. For comparison, these are the
# values we want to obtain:
#
# ![F and M scores](./rfm_example_3.png)
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
