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
from typing import Literal, cast

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d.axes3d import Axes3D
from pandas.testing import assert_series_equal
from sklearn.cluster import KMeans

# %%
# This package messes with the fonts used by matplotlib. This is my workaround:
fonts = plt.rcParams["font.sans-serif"]

from yellowbrick.cluster.elbow import KElbowVisualizer
from yellowbrick.cluster.silhouette import SilhouetteVisualizer

plt.rcParams["font.sans-serif"] = fonts
del fonts

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

# %% [markdown]
# ## Elbow plot: Find the optimal value of k
#
# This method is considered **unreliable**. But it's simple to apply it with
# the aid of the `yellowbrick` package. So, just out of curiosity, we're going
# to check out the kind of result the elbow method yields.

# %%
features = ["PTRecency", "PTFrequency", "PTAvgSpent"]
X = df[features].to_numpy()

# %%
# Distortion: Mean sum of squared distances to centers
fig = plt.figure(figsize=(8.0, 6.5), layout="tight")
ax = fig.add_subplot()

kmeans = KMeans(max_iter=1000, random_state=333)
visualizer = KElbowVisualizer(
    kmeans,
    k=16,
    metric="distortion",
    timings=False,
    ax=ax,
)
visualizer.fit(X)
visualizer.show()

ax.set_title("Distortion Score for k-means Clustering")
ax.set_ylabel("Distortion score")
ax.set_xlabel("Number of clusters k")
ax.set_xticks(list(range(2, 17)))

plt.show()

# %% [markdown]
# One could say that in the above figure there's no clear elbow. This is one of
# the situations in which this approach is considered unreliable.

# %%
# Silhouette: Mean ratio of intra-cluster and nearest-cluster distances
fig = plt.figure(figsize=(8.0, 6.5), layout="tight")
ax = fig.add_subplot()

ks = list(range(4, 17))

kmeans = KMeans(max_iter=1000, random_state=333)
visualizer = KElbowVisualizer(
    kmeans,
    k=ks,  # pyright: ignore [reportArgumentType]
    metric="silhouette",
    timings=False,
    locate_elbow=False,
    ax=ax,
)
visualizer.fit(X)
visualizer.show()

ax.set_title("Silhouette Score for k-means Clustering")
ax.set_ylabel("Silhouette score")
ax.set_xlabel("Number of clusters k")
ax.set_xticks(ks)

plt.show()

# %% [markdown]
# This plot doesn't even look like an elbow!!! Furthermore, based on what I saw
# in a couple of papers, this type of silhouette plot doesn't have a
# well-defined pattern. Then, if the goal is to find the optimal value of k,
# using this figure doesn't seem like a good idea.
#
# However, the plot above has its value. In this case, it shows that the
# silhouette coefficients aren't close to 1 (the ideal value). So this plot
# gives us an indication that our model isn't producing very good results.

# %%
# Calinski-Harabasz: Ratio of within to between cluster dispersion
fig = plt.figure(figsize=(8.0, 6.5), layout="tight")
ax = fig.add_subplot()

ks = list(range(4, 17))

kmeans = KMeans(max_iter=1000, random_state=333)
visualizer = KElbowVisualizer(
    kmeans,
    k=ks,  # pyright: ignore [reportArgumentType]
    metric="calinski_harabasz",
    timings=False,
    locate_elbow=False,
    ax=ax,
)
visualizer.fit(X)
visualizer.show()

ax.set_title("Calinski-Harabasz Index for k-means Clustering")
ax.set_ylabel("Calinski-Harabasz index")
ax.set_xlabel("Number of clusters k")
ax.set_xticks(ks)

plt.show()

# %% [markdown]
# This plot should have a peak. This indicates that there's a problem with our
# model. However, this code seems to be working correctly. So, before trying
# anything else, we'll define a function for recreating the figures above.


# %%
def plot_elbow(
    df: pd.DataFrame,
    *,
    features: list[str],
    ks: list[int],
    metric: Literal["calinski_harabasz", "distortion", "silhouette"] = "distortion",
    save: bool = False,
    figsize: tuple[float, float] = (8.0, 6.5),
) -> None:
    fig = plt.figure(figsize=figsize, layout="tight")
    ax = fig.add_subplot()

    locate_elbow = metric == "distortion"
    visualizer = KElbowVisualizer(
        KMeans(max_iter=1000, random_state=333),
        k=ks,  # pyright: ignore [reportArgumentType]
        metric=metric,
        locate_elbow=locate_elbow,
        ax=ax,
        timings=False,
    )
    visualizer.fit(df[features].to_numpy())
    visualizer.draw()

    if locate_elbow:
        handles, _ = ax.get_legend_handles_labels()
        ax.legend(
            handles=[handles[0]],
            loc="best",
            fontsize="medium",
            frameon=True,
        )

    metric_label = (
        "Calinski-Harabasz index" if metric == "calinski_harabasz" else f"{metric.capitalize()} score"
    )
    ax.set_title(f"{metric_label.title()} for k-means Clustering")
    ax.set_ylabel(metric_label)
    ax.set_xlabel("Number of clusters k")
    ax.set_xticks(ks)

    if save:
        out_img = IMG_DIR / f"elbow_{metric}.png"
        fig.savefig(out_img)
        plt.close(fig)
    else:
        plt.show()


# %% [markdown]
# Testing this function with another set of features:

# %%
features = ["QTRecency", "QTFrequency", "QTAvgSpent"]
ks = list(range(4, 17))

# %%
plot_elbow(df, metric="distortion", features=features, ks=ks)

# %%
plot_elbow(df, metric="silhouette", features=features, ks=ks)

# %%
plot_elbow(df, metric="calinski_harabasz", features=features, ks=ks)

# %% [markdown]
# ## Silhouette analysis

# %%
# Use the same set of features as above
features = ["QTRecency", "QTFrequency", "QTAvgSpent"]
X = df[features].to_numpy()

# %%
# k = 5
fig = plt.figure(figsize=(8.0, 6.5), layout="tight")
ax = fig.add_subplot()
kmeans = KMeans(n_clusters=5, max_iter=1000, random_state=333)
visualizer = SilhouetteVisualizer(kmeans, colors="Dark2", ax=ax)
visualizer.fit(X)
visualizer.show()
plt.show()

# %%
# Alternative implementation
fig = plt.figure(figsize=(8.0, 6.5), layout="tight")
ax = fig.add_subplot()

kmeans = KMeans(n_clusters=5, max_iter=1000, random_state=333)
visualizer = SilhouetteVisualizer(kmeans, colors=COLORS, ax=ax)
visualizer.fit(X)
visualizer.finalize()

ax.set_title("Silhouette Plot of k-Means Clustering for k = 5")
ax.set_xlabel("Silhouette coefficient")
ax.set_ylabel("Cluster label")
ax.set_yticklabels([str(i) for i in range(1, 6)])

plt.show()


# %%
def silhouette_plot(
    df: pd.DataFrame,
    *,
    features: list[str],
    k: int,
    save: bool = False,
    figsize: tuple[float, float] = (8.0, 6.5),
) -> None:
    fig = plt.figure(figsize=figsize, layout="tight")
    ax = fig.add_subplot()

    kmeans = KMeans(n_clusters=k, max_iter=1000, random_state=333)
    visualizer = SilhouetteVisualizer(kmeans, colors=COLORS, ax=ax)
    visualizer.fit(df[features].to_numpy())
    visualizer.finalize()

    ax.set_title(f"Silhouette Plot of k-Means Clustering for k = {k}")
    ax.set_xlabel("Silhouette coefficient")
    ax.set_ylabel("Cluster label")
    ax.set_yticklabels([str(i) for i in range(1, k + 1)])

    if save:
        out_img = IMG_DIR / f"silhouette_plot_{k}.png"
        fig.savefig(out_img)
        plt.close(fig)
    else:
        plt.show()


# %%
silhouette_plot(df, k=6, features=features)

# %%
silhouette_plot(df, k=7, features=features)

# %%
silhouette_plot(df, k=8, features=features)

# %% [markdown]
# Some of these results are decent. But none of them look very good. Assuming
# my code works correctly, either there's a problem with the features I used or
# the k-means algorithm isn't the most appropriate. I need to investigate these
# possibilities. So for now this notebook is finished.
