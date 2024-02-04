# %% [markdown]
# # Online Retail Dataset: k-means Clustering
#
# In this notebook, I'll perform customer segmentation using the k-means
# clustering algorithm.
#
# ## Imports

# %%
import random
from pathlib import Path
from typing import cast

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d.axes3d import Axes3D
from pandas.testing import assert_series_equal
from sklearn.cluster import KMeans

# %% [markdown]
# ## Read features
#
# In another notebook, we obtained the features we'll use here. Next, we'll
# read those results.

# %%
# Path to features CSV
file_path = Path.cwd().parents[1] / "data" / "clustering_features.csv"
assert file_path.exists(), f"file doesn't exist: {file_path}"
assert file_path.is_file(), f"not a file: {file_path}"

# %%
# Read features CSV
df = pd.read_csv(
    file_path,
    dtype={
        "CustomerID": "category",
        "Recency": np.int_,
        "Frequency": np.int_,
        "AvgSpent": np.float_,
        "PTRecency": np.float_,
        "PTFrequency": np.float_,
        "PTAvgSpent": np.float_,
        "QTRecency": np.float_,
        "QTFrequency": np.float_,
        "QTAvgSpent": np.float_,
    },
    index_col="CustomerID",
)
df.head()

# %%
df.info()

# %% [markdown]
# ## k-means: First attempt
#
# In this section, the goal is to get the basic code working. Improvements on
# this initial approach will be made later. For this first attempt, we'll
# **arbitrarily** choose `k = 5`.

# %%
# Use power transform features
features = ["PTRecency", "PTFrequency", "PTAvgSpent"]
X = df[features].to_numpy()
X

# %%
# k-means parameters
k = 5
max_iter = 300

# %%
# Get cluster labels
kmeans = KMeans(n_clusters=k, max_iter=max_iter, random_state=333)
cluster_labels = kmeans.fit_predict(X)
assert kmeans.n_iter_ < max_iter, f"maximum number of iterations is too low: {max_iter}"
cluster_labels

# %%
# Unique labels
unique_labels = np.unique(cluster_labels)
np.sort(unique_labels)

# %%
# Make labels start from 1
cluster_labels += 1
unique_labels = np.unique(cluster_labels)
np.sort(unique_labels)

# %%
del unique_labels

# %%
# Add column to DataFrame
column_name = f"ClusterLabel{k}"
df[column_name] = cluster_labels
df.head()

# %% [markdown]
# The above code seems to be working correctly. Then we're going to collect the
# essential parts of this code, and use them to define a function:


# %%
def add_cluster_labels(
    df: pd.DataFrame,
    features: list[str],
    k: int,
    max_iter: int = 300,
) -> pd.DataFrame:
    X = df[features].to_numpy()  # noqa: N806
    kmeans = KMeans(n_clusters=k, max_iter=max_iter, random_state=333).fit(X)
    assert kmeans.n_iter_ < max_iter, f"maximum number of iterations is too low: {max_iter}"
    df[f"ClusterLabel{k}"] = kmeans.predict(X) + 1
    return df


# %%
# Quick check
old_labels = df.pop("ClusterLabel5")
df = add_cluster_labels(
    df,
    features=["PTRecency", "PTFrequency", "PTAvgSpent"],
    k=5,
)
assert_series_equal(df.ClusterLabel5, old_labels)

# %%
del old_labels

# %% [markdown]
# ### Plot clusters

# %%
fig = plt.figure(figsize=(8.0, 6.5), layout="tight")
ax = fig.add_subplot(projection="3d")
ax = cast(Axes3D, ax)

colors = list(mpl.colormaps["Dark2"].colors)[:5]  # pyright: ignore [reportAttributeAccessIssue]
markers = ["*", "D", "P", "X", "^"]

for i in range(5):
    cluster_label = i + 1
    df_cluster = df[df["ClusterLabel5"] == cluster_label]
    ax.scatter(
        df_cluster["PTRecency"],
        df_cluster["PTFrequency"],
        df_cluster["PTAvgSpent"],
        color=colors[i],
        marker=markers[i],
        label=f"Cluster {cluster_label}",
    )

ax.view_init(elev=31.0, azim=16.0, roll=0.0)
ax.legend()
ax.set_xlabel("PTRecency")
ax.set_ylabel("PTFrequency")
ax.set_zlabel("PTAvgSpent")

plt.show()

# %% [markdown]
# These results don't look very good. But that's not the point. What matters is
# that the code above works. Next, we'll generalize this code, and create a
# function.

# %%
# Path to images directory
IMG_DIR = Path.cwd().parents[1] / "img"
assert IMG_DIR.exists(), f"directory doesn't exist: {IMG_DIR}"
assert IMG_DIR.is_dir(), f"not a directory: {IMG_DIR}"

# %%
# Define colors and markers
random.seed(a=333)

COLORS = list(mpl.colormaps["Dark2"].colors)  # pyright: ignore [reportAttributeAccessIssue]
COLORS.extend(list(mpl.colormaps["Set2"].colors))  # pyright: ignore [reportAttributeAccessIssue]
random.shuffle(COLORS)
assert len(COLORS) == 16

MARKERS = ["*", ".", "8", "<", ">", "D", "H", "P", "X", "^", "d", "h", "o", "p", "s", "v"]
random.shuffle(MARKERS)
assert len(MARKERS) == 16


# %%
def plot_clusters(
    df: pd.DataFrame,
    *,
    features: list[str],
    k: int,
    save: bool = False,
    figsize: tuple[float, float] = (8.0, 6.5),
) -> None:
    fig = plt.figure(figsize=figsize, layout="tight")
    ax = fig.add_subplot(projection="3d")
    ax = cast(Axes3D, ax)

    xlabel, ylabel, zlabel = features

    for i in range(k):
        cluster_label = i + 1
        df_cluster = df[df[f"ClusterLabel{k}"] == cluster_label]
        ax.scatter(
            df_cluster[xlabel],
            df_cluster[ylabel],
            df_cluster[zlabel],
            color=COLORS[i],
            marker=MARKERS[i],
            label=f"Cluster {cluster_label}",
        )

    ax.view_init(elev=31.0, azim=16.0, roll=0.0)
    ax.legend()
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)

    if save:
        out_img = IMG_DIR / f"clusters_{k}.png"
        fig.savefig(out_img)
        plt.close(fig)
    else:
        plt.show()


# %%
plot_clusters(
    df,
    features=["PTRecency", "PTFrequency", "PTAvgSpent"],
    k=5,
    save=True,
)
