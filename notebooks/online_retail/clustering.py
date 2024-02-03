# %% [markdown]
# # Online Retail Dataset: Feature Engineering for Clustering
#
# In this notebook, I'll create some features that will be used to perform
# customer segmentation with clustering algorithms.
#
# ## Imports

# %%
from pathlib import Path
from typing import cast

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.axes import Axes
from mpl_toolkits.mplot3d.axes3d import Axes3D
from sklearn.preprocessing import PowerTransformer, QuantileTransformer, StandardScaler

from utils.online_retail import compute_rfm_attributes, compute_total_price, get_clean_data

# %% [markdown]
# ## Prepare the data
#
# In other notebooks, we've cleaned the original data, and computed the RFM
# attributes. We're done with the RFM analysis, but we'll use those results
# here. Specifically, we'll utilize recency, frequency and monetary value to
# create new features.
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
# Get RFM attributes
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
# Correlation between RFM attributes
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
# Introduce feature to replace `Monetary`
df["AvgSpent"] = df["Monetary"] / df["Frequency"]
assert df["AvgSpent"].isna().sum() == 0

# %%
# Correlations with `Monetary` replaced
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
# ## Create features by using transforms
# ### Power transforms

# %%
# Box-Cox transform requires strictly positive input
assert (df["Recency"] > 0).all()
assert (df["Frequency"] > 0).all()
assert (df["AvgSpent"] > 0).all()

# %%
# Actual transformation
X = df[["Recency", "Frequency", "AvgSpent"]].to_numpy()
X_new = PowerTransformer(method="box-cox").fit_transform(X)
df = df.assign(
    PTRecency=X_new[:, 0],
    PTFrequency=X_new[:, 1],
    PTAvgSpent=X_new[:, 2],
)
del X_new
df.head()

# %%
# New features have mean 0 and variance 1
df[["PTRecency", "PTFrequency", "PTAvgSpent"]].agg(["mean", "var"])

# %%
# Distribution of PTRecency: NOT Gaussian-like
fig, ax = plt.subplots(figsize=(8.0, 6.0), layout="tight")
ax = cast(Axes, ax)
sns.kdeplot(data=df, x="PTRecency", ax=ax)
plt.show()

# %%
# Distribution of PTFrequency: NOT Gaussian-like
fig, ax = plt.subplots(figsize=(8.0, 6.0), layout="tight")
ax = cast(Axes, ax)
sns.kdeplot(data=df, x="PTFrequency", ax=ax)
plt.show()

# %%
# Distribution of PTAvgSpent: Gaussian-like!!!
fig, ax = plt.subplots(figsize=(8.0, 6.0), layout="tight")
ax = cast(Axes, ax)
sns.kdeplot(data=df, x="PTAvgSpent", ax=ax)
plt.show()

# %% [markdown]
# ### Quantile transform

# %%
qt = QuantileTransformer(n_quantiles=100, output_distribution="normal", random_state=333)
X_new = qt.fit_transform(X)
df = df.assign(
    QTRecency=X_new[:, 0],
    QTFrequency=X_new[:, 1],
    QTAvgSpent=X_new[:, 2],
)
del qt
del X_new
df.head()

# %%
# New features need to be standardized
df[["QTRecency", "QTFrequency", "QTAvgSpent"]].agg(["mean", "var"])

# %%
# Standardize new features
X = df[["QTRecency", "QTFrequency", "QTAvgSpent"]].to_numpy()
X_new = StandardScaler().fit_transform(X)
df = df.assign(
    QTRecency=X_new[:, 0],
    QTFrequency=X_new[:, 1],
    QTAvgSpent=X_new[:, 2],
)
del X
del X_new
df.head()

# %%
# Distribution of QTRecency: Gaussian-like!!!
fig, ax = plt.subplots(figsize=(8.0, 6.0), layout="tight")
ax = cast(Axes, ax)
sns.kdeplot(data=df, x="QTRecency", ax=ax)
plt.show()

# %%
# Distribution of QTFrequency: NOT Gaussian-like
fig, ax = plt.subplots(figsize=(8.0, 6.0), layout="tight")
ax = cast(Axes, ax)
sns.kdeplot(data=df, x="QTFrequency", ax=ax)
plt.show()

# %%
# Distribution of QTAvgSpent: Gaussian-like!!!
fig, ax = plt.subplots(figsize=(8.0, 6.0), layout="tight")
ax = cast(Axes, ax)
sns.kdeplot(data=df, x="QTAvgSpent", ax=ax)
plt.show()

# %% [markdown]
# ## 3D Plots

# %%
# For comparison, plot "original" features
fig = plt.figure(figsize=(8.0, 6.5), layout="tight")
ax = fig.add_subplot(projection="3d")
ax = cast(Axes3D, ax)
ax.scatter(df.Recency, df.Frequency, df.AvgSpent)
ax.set_xlabel("Recency")
ax.set_ylabel("Frequency")
ax.set_zlabel("AvgSpent")
plt.show()

# %%
# Features yielded by power transform
fig = plt.figure(figsize=(8.0, 6.5), layout="tight")
ax = fig.add_subplot(projection="3d")
ax = cast(Axes3D, ax)
ax.scatter(df.PTRecency, df.PTFrequency, df.PTAvgSpent)
ax.view_init(elev=55.0, azim=-10.0, roll=0.0)
ax.set_xlabel("PTRecency")
ax.set_ylabel("PTFrequency")
ax.set_zlabel("PTAvgSpent")
plt.show()

# %%
# Features yielded by quantile transform
fig = plt.figure(figsize=(8.0, 6.5), layout="tight")
ax = fig.add_subplot(projection="3d")
ax = cast(Axes3D, ax)
ax.scatter(df.QTRecency, df.QTFrequency, df.QTAvgSpent)
ax.view_init(elev=55.0, azim=3.0, roll=0.0)
ax.set_xlabel("QTRecency")
ax.set_ylabel("QTFrequency")
ax.set_zlabel("QTAvgSpent")
plt.show()

# %% [markdown]
# Based on the above figures, the power transform features seem the most
# useful. After all, in the corresponding plot, we can see a few clusters quite
# clearly.
