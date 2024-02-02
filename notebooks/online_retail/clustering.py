# %% [markdown]
# # Online Retail Dataset: Clustering
#
# In this notebook, I'll use a clustering algorithm to perform customer
# segmentation.
#
# ## Imports

# %%
from pathlib import Path

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
