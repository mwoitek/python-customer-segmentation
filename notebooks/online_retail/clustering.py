# %% [markdown]
# # Online Retail Dataset: Clustering
#
# In this notebook, I'll use a clustering algorithm to perform customer
# segmentation.
#
# ## Imports

# %%
from pathlib import Path
from typing import cast

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.axes import Axes
from sklearn.preprocessing import PowerTransformer

from utils.online_retail import compute_rfm_attributes, compute_total_price, get_clean_data

# %% [markdown]
# ## Prepare the data
#
# In other notebooks, we've cleaned the original data, and computed the RFM
# attributes. We're done with the RFM analysis, but we'll use those results
# here. More precisely, to implement the clustering algorithm, we'll utilize
# recency, frequency and monetary value as features.
#
# Fortunately, we've defined a few functions that we can call to repeat the
# steps mentioned above. We can prepare the data again with the aid of the code
# that follows.

# %%
# Path to original dataset
file_path = Path.cwd().parents[1] / "data" / "online_retail.xlsx"
assert file_path.exists(), f"file doesn't exist: {file_path}"
assert file_path.is_file(), f"not a file: {file_path}"

# %%
df = (
    get_clean_data(file_path)
    .groupby("InvoiceNo", observed=True)
    .apply(compute_total_price, include_groups=False)
    .reset_index()
    .pipe(compute_rfm_attributes)
)
df.head()

# %%
df.info()

# %% [markdown]
# **NOTE**: I'm not sure where this is going. To have a clearer idea, I need to
# play around with the data. So for now this is the only thing I'm going to do.
# Later I'll come back here and build a narrative.

# %% [markdown]
# ## Correlations

# %%
fig, ax = plt.subplots(figsize=(6.0, 6.0), layout="tight")
ax = cast(Axes, ax)
sns.heatmap(
    df[["Recency", "Frequency", "Monetary"]].corr(),
    annot=True,
    cmap=mpl.colormaps["coolwarm"],
    ax=ax,
)
plt.show()

# %%
# New feature
df["AvgSpent"] = df["Monetary"] / df["Frequency"]
df["AvgSpent"].isna().sum()

# %%
fig, ax = plt.subplots(figsize=(6.0, 6.0), layout="tight")
ax = cast(Axes, ax)
sns.heatmap(
    df[["Recency", "Frequency", "AvgSpent"]].corr(),
    annot=True,
    cmap=mpl.colormaps["coolwarm"],
    ax=ax,
)
plt.show()

# %% [markdown]
# ## Distributions

# %%
# Recency
fig, ax = plt.subplots(figsize=(8.0, 6.0), layout="tight")
ax = cast(Axes, ax)
sns.kdeplot(data=df, x="Recency", ax=ax)
plt.show()

# %%
# Frequency
fig, ax = plt.subplots(figsize=(8.0, 6.0), layout="tight")
ax = cast(Axes, ax)
sns.kdeplot(data=df, x="Frequency", ax=ax)
plt.show()

# %%
# AvgSpent
fig, ax = plt.subplots(figsize=(8.0, 6.0), layout="tight")
ax = cast(Axes, ax)
sns.kdeplot(data=df, x="AvgSpent", ax=ax)
plt.show()

# %% [markdown]
# ## Power transforms

# %%
assert (df["Recency"] > 0).all()
assert (df["Frequency"] > 0).all()
assert (df["AvgSpent"] > 0).all()

# %%
X = df[["Recency", "Frequency", "AvgSpent"]].to_numpy()
X

# %%
pt = PowerTransformer(method="box-cox")
X_new = pt.fit_transform(X)
X_new

# %%
df = df.assign(
    PTRecency=X_new[:, 0],
    PTFrequency=X_new[:, 1],
    PTAvgSpent=X_new[:, 2],
)
df.head()

# %%
df[["PTRecency", "PTFrequency", "PTAvgSpent"]].agg(["mean", "var"])

# %%
# NOT Gaussian-like
fig, ax = plt.subplots(figsize=(8.0, 6.0), layout="tight")
ax = cast(Axes, ax)
sns.kdeplot(data=df, x="PTRecency", ax=ax)
plt.show()

# %%
# NOT Gaussian-like
fig, ax = plt.subplots(figsize=(8.0, 6.0), layout="tight")
ax = cast(Axes, ax)
sns.kdeplot(data=df, x="PTFrequency", ax=ax)
plt.show()

# %%
# Gaussian-like
fig, ax = plt.subplots(figsize=(8.0, 6.0), layout="tight")
ax = cast(Axes, ax)
sns.kdeplot(data=df, x="PTAvgSpent", ax=ax)
plt.show()
