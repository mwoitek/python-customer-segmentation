# %% [markdown]
# # Online Retail Dataset: k-means Clustering
#
# In this notebook, I'll perform customer segmentation using the k-means
# clustering algorithm.
#
# ## Imports

# %%
from pathlib import Path

import numpy as np
import pandas as pd
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
