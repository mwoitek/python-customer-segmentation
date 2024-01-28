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

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from more_itertools import unique_everseen
from pandas.testing import assert_frame_equal, assert_series_equal

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
    index_col="CustomerID",
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
df_rfm[score_cols] = df_rfm[score_cols].transform(lambda col: col.cat.as_ordered())
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

# %%
# Make it categorical
df_rfm["Segment"] = df_rfm["Segment"].astype("category")
df_rfm["Segment"].cat.categories

# %%
# Quick checks
assert not df_rfm["Segment"].cat.ordered

num_categories_1: int = df_rfm["Segment"].cat.categories.shape[0]
num_categories_2: int = len(set(SEGMENTS_5.values()))
print(num_categories_1 == num_categories_2)
del num_categories_1
del num_categories_2

# %% [markdown]
# ### Save segment data

# %%
# File path for output CSV
out_file = file_path.parent / "rfm_segments.csv"

df_rfm.to_csv(out_file, index=True)

# %% [markdown]
# ## Summarizing through functions
#
# The above code seems to be doing what it's supposed to. So, as usual, we'll
# collect the essential parts of this code, and define a couple of functions.


# %%
def read_rfm_scores(file_path: Path) -> pd.DataFrame:
    df = pd.read_csv(
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
        index_col="CustomerID",
    )

    score_cols = ["RScore", "FScore", "MScore"]
    df[score_cols] = df[score_cols].transform(lambda col: col.cat.as_ordered())

    return df


# %%
def label_customers(df: pd.DataFrame, segments_dict: dict[str, str]) -> pd.DataFrame:
    return df.assign(Segment=df["RFMCell"].map(segments_dict).astype("category"))


# %%
# Quick check
df_func = read_rfm_scores(file_path)
assert_frame_equal(df_func, df_rfm.iloc[:, :-1])

# %%
df_func = label_customers(df_func, SEGMENTS_5)
assert_frame_equal(df_func, df_rfm)

# %%
del df_func

# %% [markdown]
# ## Customers by segment
# ### Calculations

# %%
# Number of customers
num_customers = df_rfm["Segment"].value_counts()
num_customers = cast(pd.Series, num_customers)
num_customers

# %%
# Proportion
prop = df_rfm["Segment"].value_counts(normalize=True)
prop = cast(pd.Series, prop)
prop

# %%
# Quick check
assert prop.loc["Champions"] == num_customers.loc["Champions"] / num_customers.sum()

# %%
# Percentage
perc = 100.0 * prop
perc = cast(pd.Series, perc)
perc.name = "percentage"
perc.transform(lambda p: f"{p:.2f}%")

# %%
# A few checks
assert (perc >= 0.0).all()
assert (perc <= 100.0).all()
assert np.isclose(perc.sum(), 100.0)

# %%
del num_customers
del prop
del perc

# %%
# Combine the above results into a DataFrame
cus_by_seg = (
    df_rfm["Segment"]
    .value_counts()
    .to_frame()
    .rename(columns={"count": "Count"})
    .assign(
        Percentage=(100.0 * df_rfm["Segment"].value_counts(normalize=True)).transform(lambda p: f"{p:.2f}%")
    )
    .reindex(list(unique_everseen(SEGMENTS_5.values())))
)
cus_by_seg = cast(pd.DataFrame, cus_by_seg)
cus_by_seg

# %% [markdown]
# Note the following: customers in the "Champions" segment correspond to almost
# 20% of all customers. In other words: **roughly speaking, the top 20% of
# customers can be associated with the "Champions" segment**. The relevance of
# this fact will become clear when we discuss the Pareto principle.
#
# Use the essential parts of this code to implement a function:


# %%
def customers_by_segment(df: pd.DataFrame, segments_dict: dict[str, str]) -> pd.DataFrame:
    return (
        df["Segment"]
        .value_counts()
        .to_frame()
        .rename(columns={"count": "CustomerCount"})
        .assign(CustomerPercentage=(100.0 * df["Segment"].value_counts(normalize=True)).round(2))
        .reindex(list(unique_everseen(segments_dict.values())))
    )


# %%
cus_by_seg = customers_by_segment(df_rfm, SEGMENTS_5)
cus_by_seg

# %% [markdown]
# ### Visualization

# %%
# Number of customers
fig, ax = plt.subplots(figsize=(8.0, 6.0), layout="tight")
ax = cast(Axes, ax)
ax.barh(cus_by_seg.index, cus_by_seg["CustomerCount"])
ax.invert_yaxis()
ax.set_title("Number of Customers by Segment")
ax.set_xlabel("Number of Customers")
plt.show()

# %%
# Percentage
fig, ax = plt.subplots(figsize=(8.0, 6.0), layout="tight")
ax = cast(Axes, ax)
ax.barh(cus_by_seg.index, cus_by_seg["CustomerPercentage"])
ax.invert_yaxis()
ax.set_title("Percentage of Customers in Each Segment")
ax.set_xlabel("Percentage (%)")
plt.show()

# %%
# Combine these plots into a single figure
fig, axs = plt.subplots(1, 2, sharey=True, figsize=(13.0, 6.5), layout="constrained")

axs_list = axs.flatten().tolist()
axs_list = cast(list[Axes], axs_list)

axs_list[0].barh(cus_by_seg.index, cus_by_seg["CustomerCount"])
axs_list[0].set_title("Number of Customers by Segment")
axs_list[0].set_xlabel("Number of Customers")

axs_list[1].barh(cus_by_seg.index, cus_by_seg["CustomerPercentage"])
axs_list[1].set_title("Percentage of Customers in Each Segment")
axs_list[1].set_xlabel("Percentage (%)")

axs_list[1].invert_yaxis()
fig.suptitle("Customers by Segment", fontsize="xx-large")

plt.show()

# %% [markdown]
# ## Revenue by segment

# %%
# Total revenue by segment
total_by_seg = df_rfm.groupby(by="Segment", observed=True).Monetary.sum()
total_by_seg = cast(pd.Series, total_by_seg)
total_by_seg.name = "revenue"
total_by_seg.round(2)

# %%
# Quick check
rev_new_1 = total_by_seg.loc["New Customers"]
rev_new_2 = df_rfm.loc[df_rfm["Segment"] == "New Customers", "Monetary"].sum()
assert rev_new_1 == rev_new_2
del rev_new_1
del rev_new_2

# %%
# Total revenue
total_rev = df_rfm["Monetary"].sum()
total_rev = cast(np.float_, total_rev)
assert np.isclose(total_rev, total_by_seg.sum())
np.round(total_rev, 2)

# %%
# Proportion
rev_prop = df_rfm.groupby(by="Segment", observed=True).Monetary.agg(lambda col: col.sum() / total_rev)
rev_prop = cast(pd.Series, rev_prop)
rev_prop.name = "proportion"
rev_prop

# %%
# Quick check
prop_1 = rev_prop.loc["New Customers"]
prop_2 = total_by_seg.loc["New Customers"] / df_rfm["Monetary"].sum()
assert prop_1 == prop_2
del prop_1
del prop_2

# %%
# Percentage
rev_perc = 100.0 * rev_prop
rev_perc = cast(pd.Series, rev_perc)
rev_perc.name = "percentage"
rev_perc.transform(lambda p: f"{p:.2f}%")

# %%
# A few checks
assert (rev_perc >= 0.0).all()
assert (rev_perc <= 100.0).all()
assert np.isclose(rev_perc.sum(), 100.0)

# %%
# Combine the above results into a DataFrame
seg_groups = df_rfm.groupby(by="Segment", observed=True)
total_rev = np.round(df_rfm["Monetary"].sum(), 2)

rev_by_seg = (
    seg_groups.Monetary.sum()
    .round(2)
    .to_frame()
    .rename(columns={"Monetary": "Revenue"})
    .assign(
        Percentage=(100.0 * seg_groups.Monetary.agg(lambda col: col.sum() / total_rev)).transform(
            lambda p: f"{p:.2f}%"
        )
    )
    .reindex(list(unique_everseen(SEGMENTS_5.values())))
)
rev_by_seg = cast(pd.DataFrame, rev_by_seg)
rev_by_seg

# %% [markdown]
# Use the essential parts of this code to implement a function:


# %%
def revenue_by_segment(df: pd.DataFrame, segments_dict: dict[str, str]) -> pd.DataFrame:
    seg_groups = df.groupby(by="Segment", observed=True)
    total_rev = np.round(df["Monetary"].sum(), 2)
    return (
        seg_groups.Monetary.sum()
        .round(2)
        .to_frame()
        .rename(columns={"Monetary": "Revenue"})
        .assign(RevenuePercentage=(100.0 * seg_groups.Monetary.agg(lambda c: c.sum() / total_rev)).round(2))
        .reindex(list(unique_everseen(segments_dict.values())))
    )


# %%
revenue_by_segment(df_rfm, SEGMENTS_5)

# %%
# Combine segment data
cus_by_seg = cus_by_seg.rename(columns={"Percentage": "CustomerPercentage"})
rev_by_seg = rev_by_seg.rename(columns={"Percentage": "RevenuePercentage"})
pd.concat([cus_by_seg, rev_by_seg], axis=1)


# %%
def get_segment_data(df: pd.DataFrame, segments_dict: dict[str, str]) -> pd.DataFrame:
    return pd.concat(
        [
            customers_by_segment(df, segments_dict),
            revenue_by_segment(df, segments_dict),
        ],
        axis=1,
    )


# %%
get_segment_data(df_rfm, SEGMENTS_5)
