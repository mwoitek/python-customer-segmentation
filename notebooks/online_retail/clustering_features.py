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
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from mpl_toolkits.mplot3d.axes3d import Axes3D
from pandas.testing import assert_frame_equal
from sklearn.preprocessing import PowerTransformer, QuantileTransformer, StandardScaler

from utils.online_retail import compute_rfm_attributes, compute_total_price, get_clean_data
from utils.outliers import remove_outliers

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

# %%
# Remove outliers from Frequency and Monetary
df = remove_outliers(df, ["Frequency", "Monetary"])

# %%
df.head(10)

# %%
df.info()

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
# New feature
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
assert (df["Monetary"] > 0).all()
assert (df["AvgSpent"] > 0).all()

# %%
# Actual transformation
X = df[["Recency", "Frequency", "Monetary", "AvgSpent"]].to_numpy()
X_new = PowerTransformer(method="box-cox").fit_transform(X)
df = df.assign(
    PTRecency=X_new[:, 0],
    PTFrequency=X_new[:, 1],
    PTMonetary=X_new[:, 2],
    PTAvgSpent=X_new[:, 3],
)
del X_new
df.head()

# %%
# New features have mean 0 and variance 1
df[["PTRecency", "PTFrequency", "PTMonetary", "PTAvgSpent"]].agg(["mean", "var"])

# %%
# Correlations
fig, ax = plt.subplots(figsize=(6.0, 6.0), layout="tight")
ax = cast(Axes, ax)
sns.heatmap(
    df[["PTRecency", "PTFrequency", "PTMonetary", "PTAvgSpent"]].corr(),
    annot=True,
    cmap=mpl.colormaps["coolwarm"],
    ax=ax,
)
plt.show()

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
# Distribution of PTMonetary: Almost Gaussian-like
fig, ax = plt.subplots(figsize=(8.0, 6.0), layout="tight")
ax = cast(Axes, ax)
sns.kdeplot(data=df, x="PTMonetary", ax=ax)
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
    QTMonetary=X_new[:, 2],
    QTAvgSpent=X_new[:, 3],
)
del qt
del X_new
df.head()

# %%
# New features need to be standardized
df[["QTRecency", "QTFrequency", "QTMonetary", "QTAvgSpent"]].agg(["mean", "var"])

# %%
# Standardize new features
X = df[["QTRecency", "QTFrequency", "QTMonetary", "QTAvgSpent"]].to_numpy()
X_new = StandardScaler().fit_transform(X)
df = df.assign(
    QTRecency=X_new[:, 0],
    QTFrequency=X_new[:, 1],
    QTMonetary=X_new[:, 2],
    QTAvgSpent=X_new[:, 3],
)
del X
del X_new
df.head()

# %%
# Correlations
fig, ax = plt.subplots(figsize=(6.0, 6.0), layout="tight")
ax = cast(Axes, ax)
sns.heatmap(
    df[["QTRecency", "QTFrequency", "QTMonetary", "QTAvgSpent"]].corr(),
    annot=True,
    cmap=mpl.colormaps["coolwarm"],
    ax=ax,
)
plt.show()

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
# Distribution of QTMonetary: Gaussian-like!!!
fig, ax = plt.subplots(figsize=(8.0, 6.0), layout="tight")
ax = cast(Axes, ax)
sns.kdeplot(data=df, x="QTMonetary", ax=ax)
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
ax.scatter(df.Recency, df.Frequency, df.Monetary)
ax.set_xlabel("Recency")
ax.set_ylabel("Frequency")
ax.set_zlabel("Monetary")
plt.show()

# %% [markdown]
# Features yielded by power transform:

# %%
fig = plt.figure(figsize=(8.0, 6.5), layout="tight")
ax = fig.add_subplot(projection="3d")
ax = cast(Axes3D, ax)
ax.scatter(df.PTRecency, df.PTFrequency, df.PTMonetary)
ax.view_init(elev=55.0, azim=-10.0, roll=0.0)
ax.set_xlabel("PTRecency")
ax.set_ylabel("PTFrequency")
ax.set_zlabel("PTMonetary")
plt.show()

# %%
fig = plt.figure(figsize=(8.0, 6.5), layout="tight")
ax = fig.add_subplot(projection="3d")
ax = cast(Axes3D, ax)
ax.scatter(df.PTRecency, df.PTFrequency, df.PTAvgSpent)
ax.view_init(elev=55.0, azim=-10.0, roll=0.0)
ax.set_xlabel("PTRecency")
ax.set_ylabel("PTFrequency")
ax.set_zlabel("PTAvgSpent")
plt.show()

# %% [markdown]
# Features yielded by quantile transform:

# %%
fig = plt.figure(figsize=(8.0, 6.5), layout="tight")
ax = fig.add_subplot(projection="3d")
ax = cast(Axes3D, ax)
ax.scatter(df.QTRecency, df.QTFrequency, df.QTMonetary)
ax.view_init(elev=55.0, azim=3.0, roll=0.0)
ax.set_xlabel("QTRecency")
ax.set_ylabel("QTFrequency")
ax.set_zlabel("QTMonetary")
plt.show()

# %%
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
# Based on the above figures, the quantile transform features seem the most
# useful. After all, in the corresponding plot, we can see a few clusters quite
# clearly.

# %% [markdown]
# ## Save features

# %%
# Just checking everything's OK
df.head()

# %%
df.info()

# %%
# Save to CSV
out_file = file_path.parent / "clustering_features.csv"
df.to_csv(out_file, index=True)

# %% [markdown]
# ## Summarizing through functions


# %%
def get_original_features(
    file_path: Path,
    outlier_cols: str | list[str] | None = None,
) -> pd.DataFrame:
    df = (
        get_clean_data(file_path)
        .groupby("InvoiceNo", observed=True)
        .apply(compute_total_price, include_groups=False)
        .reset_index()
        .pipe(compute_rfm_attributes)
        .assign(AvgSpent=lambda x: x.Monetary / x.Frequency)
    )
    return df if outlier_cols is None else remove_outliers(df, outlier_cols)


# %%
def add_power_transform_features(df: pd.DataFrame) -> pd.DataFrame:
    X = df[["Recency", "Frequency", "Monetary", "AvgSpent"]].to_numpy()  # noqa: N806
    X_new = PowerTransformer(method="box-cox").fit_transform(X)  # noqa: N806
    return df.assign(
        PTRecency=X_new[:, 0],
        PTFrequency=X_new[:, 1],
        PTMonetary=X_new[:, 2],
        PTAvgSpent=X_new[:, 3],
    )


# %%
def add_quantile_transform_features(df: pd.DataFrame) -> pd.DataFrame:
    X = df[["Recency", "Frequency", "Monetary", "AvgSpent"]].to_numpy()  # noqa: N806
    qt = QuantileTransformer(n_quantiles=100, output_distribution="normal", random_state=333)
    X_new_1 = qt.fit_transform(X)  # noqa: N806
    X_new_2 = StandardScaler().fit_transform(X_new_1)  # noqa: N806
    return df.assign(
        QTRecency=X_new_2[:, 0],
        QTFrequency=X_new_2[:, 1],
        QTMonetary=X_new_2[:, 2],
        QTAvgSpent=X_new_2[:, 3],
    )


# %%
# Quick check
df_funcs = (
    get_original_features(
        Path.cwd().parents[1] / "data" / "online_retail.xlsx",
        outlier_cols=["Frequency", "Monetary"],
    )
    .pipe(add_power_transform_features)
    .pipe(add_quantile_transform_features)
)
assert_frame_equal(df_funcs, df)
del df_funcs


# %%
def compute_and_save_features(
    file_path: Path,
    outlier_cols: str | list[str] | None = None,
) -> None:
    df = (
        get_original_features(file_path, outlier_cols)
        .pipe(add_power_transform_features)
        .pipe(add_quantile_transform_features)
    )
    df.to_csv(file_path.parent / "clustering_features.csv", index=True)


# %%
compute_and_save_features(
    Path.cwd().parents[1] / "data" / "online_retail.xlsx",
    outlier_cols=["Frequency", "Monetary"],
)
