# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
# ---

# %% [markdown]
# # Online Retail Dataset: Clean Database
# ## Imports

# %%
from pathlib import Path

import polars as pl

# %% [markdown]
# ## Path to DB

# %%
data_dir = Path.cwd().parents[1] / "data"
db_path = data_dir / "online_retail.db"
assert db_path.exists(), "DB doesn't exist"

# %% [markdown]
# ## Read original data

# %%
uri = f"sqlite://{db_path}"
dtypes = {
    "invoice_no": pl.String,
    "stock_code": pl.String,
    "description": pl.String,
    "quantity": pl.Int64,
    "invoice_date": pl.String,
    "unit_price": pl.Float64,
    "customer_id": pl.Int64,
    "country": pl.String,
}
query_cols = ", ".join(dtypes.keys())
query = f"SELECT {query_cols} FROM original_data;"

# %%
df = pl.read_database_uri(query, uri, schema_overrides=dtypes).with_columns(
    pl.col("invoice_date").str.to_datetime("%Y-%m-%d %H:%M:%S")
)
df.head()

# %% [markdown]
# ## Remove non-purchase data
#
# We begin by inspecting the `invoice_no` column. Specifically, we'll look
# at its non-integer values. These values can be selected as follows:

# %%
tmp_1 = df.filter(pl.col("invoice_no").str.find(r"^\d+$").is_null())
tmp_1.head()

# %% [markdown]
# The non-integer values start with A or C:

# %%
assert tmp_1.get_column("invoice_no").str.contains(r"^[AC].*").all()

# %% [markdown]
# When `invoice_no` starts with C, the corresponding order was canceled.
# Let's look at these entries more closely. They can be selected as
# follows:

# %%
tmp_2 = tmp_1.filter(pl.col("invoice_no").str.starts_with("C"))
tmp_2.head()

# %% [markdown]
# All of these entries have a negative `quantity`:

# %%
assert tmp_2.get_column("quantity").lt(0).all()

# %% [markdown]
# There are only 3 entries whose `invoice_no` starts with A. Clearly, they
# don't correspond to purchases:

# %%
tmp_1.filter(pl.col("invoice_no").str.starts_with("A"))

# %% [markdown]
# Finally, we're going to remove these rows. After all, we're only
# interested in the purchase data.

# %%
df = df.filter(pl.col("invoice_no").str.contains(r"^[AC].*").not_())

# %% [markdown]
# `invoice_no` has become an integer column. Then we also change its type:

# %%
assert df.get_column("invoice_no").str.contains(r"^\d+$").all(), "found non-integer value"
df = df.with_columns(pl.col("invoice_no").str.to_integer())

# %% [markdown]
# ## Dealing with negative values of `quantity`
#
# I would expect `quantity` to have only positive numbers. However, as we
# have seen, this column contains negative values. Above we have removed
# some of the corresponding entries. But many negative quantities still
# remain:

# %%
tmp_1 = df.filter(pl.col("quantity").lt(0))
tmp_1.head()

# %%
tmp_1.height

# %% [markdown]
# As we can see, rows with a negative `quantity` also have missing/invalid
# values in other columns. For example, all of these rows don't have a
# value for `customer_id`:

# %%
assert tmp_1.get_column("customer_id").is_null().all(), "found non-null value"

# %% [markdown]
# Furthermore, every row with a negative `quantity` has a `unit_price`
# equal to zero:

# %%
assert tmp_1.get_column("unit_price").eq(0.0).all(), "found non-zero price"

# %% [markdown]
# Most values of `description` are missing (but not all):

# %%
tmp_1.get_column("description").null_count()

# %% [markdown]
# Let's look more closely at the rows that have a description. These rows
# can be selected as follows:

# %%
tmp_2 = tmp_1.filter(pl.col("description").is_not_null())
tmp_2.head()

# %%
tmp_2.height

# %% [markdown]
# Judging by their descriptions, these rows don't correspond to purchase
# data:

# %%
rows = 20
with pl.Config(tbl_rows=rows):
    print(tmp_2.get_column("description").unique().head(rows))

# %% [markdown]
# More than half of these entries indicate some kind of stock check or
# damaged merchandise:

# %%
tmp_2.filter(pl.col("description").str.contains(r"(CHECK|DAMAGE)"))

# %% [markdown]
# Based on everything we've seen, it's OK to remove all rows with a
# negative `quantity`. We'll also change the type of this column. This can
# be done as follows:

# %%
df = df.filter(pl.col("quantity").gt(0)).with_columns(pl.col("quantity").cast(pl.UInt64))

# %% [markdown]
# ## Dealing with zero prices
#
# As we've seen, there are rows with a `unit_price` value equal to zero.
# This doesn't make a lot of sense. After all, unless the store is giving
# out stuff for free, all prices should be positive. Above some of these
# suspicious rows have been removed. But not all of them. This can be
# checked as follows:

# %%
tmp_1 = df.filter(pl.col("unit_price").eq(0.0))
tmp_1.head()

# %%
tmp_1.height

# %% [markdown]
# Notice that some of these rows also have missing values for `description`
# and `customer_id`. These entries have three problematic columns. We
# believe they're too broken to be useful. So we remove them right away:

# %%
df = df.filter(
    (pl.col("unit_price").eq(0.0) & pl.col("description").is_null() & pl.col("customer_id").is_null()).not_()
)

# %%
tmp_1 = df.filter(pl.col("unit_price").eq(0.0))
tmp_1.head()

# %% [markdown]
# Approximately 600 problematic entries still remain. Some of them are
# clearly non-purchase data:

# %%
with pl.Config(tbl_rows=20):
    print(tmp_1.filter(pl.col("description").str.contains(r"(ADJUST|FOUND|WRONG)")))

# %% [markdown]
# These rows can be removed. However, there are rows with proper product
# descriptions. These entries may be fixed by using other entries related
# to the same product. For example, consider the product described as
# "GLASS JAR KINGS CHOICE". These are a few rows associated with this
# product:

# %%
df.filter(description="GLASS JAR KINGS CHOICE").sort("unit_price").head(10)

# %% [markdown]
# In some cases, the price is zero. However, there are non-zero prices for
# the same product. These prices can be used to fix the problematic
# entries. This will be done later, since this fix is tricky. There are
# other problems with this data that need to be tackled first.
#
# For now, we simply remove what's reasonable to remove. First, we drop
# every entry we won't be able to fix. This can be done as follows:

# %%
occur_once = df.get_column("description").value_counts().filter(count=1).get_column("description")
df = df.filter((pl.col("unit_price").eq(0.0) & pl.col("description").is_in(occur_once)).not_())

# %% [markdown]
# Removing more non-purchase data:

# %%
# Just to be sure
descriptions = [
    "?",
    "AMAZON",
    "CHECK",
    "DOTCOM",
    "HAD BEEN PUT ASIDE",
    "RETURNED",
    "TEST",
]
tmp = df.filter(pl.col("description").is_in(descriptions))
assert tmp.get_column("unit_price").eq(0.0).all()
assert tmp.get_column("customer_id").is_null().all()

# %%
# Actual removal
df = df.filter(pl.col("description").is_in(descriptions).not_())

# %%
pattern = r"(ADJUST|FOUND)"
tmp = df.filter(pl.col("description").str.contains(pattern))
assert tmp.get_column("unit_price").eq(0.0).all()
assert tmp.get_column("customer_id").is_null().all()
df = df.filter(pl.col("description").str.contains(pattern).not_())

# %% [markdown]
# We've dropped all rows that can be removed for now. This is the remaining
# number of entries with a price of zero:

# %%
df.filter(unit_price=0.0).height

# %% [markdown]
# ## Fixing descriptions
#
# First, notice that we no longer have missing values in this column:

# %%
assert df.get_column("description").is_not_null().all()

# %% [markdown]
# Fix comma placement in descriptions:

# %%
df = df.with_columns(pl.col("description").str.replace_all(r"(\w+)\s+,(?:\s+)?(\w+)", r"${1}, ${2}"))

# %%
# HERE

# %%

# %%

# %%

# %%
