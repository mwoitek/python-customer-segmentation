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
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Online Retail Dataset: Create Database
# ## Imports

# %%
import datetime
import re
import sqlite3
from pathlib import Path

import polars as pl

# %% [markdown]
# ## Check if DB already exists

# %%
data_dir = Path.cwd().parents[1] / "data"
db_path = data_dir / "online_retail.db"

# %%
assert not db_path.exists(), "DB already exists"
db_path.touch(mode=0o644)

# %% [markdown]
# ## Check if Excel file exists

# %%
excel_file_path = db_path.with_suffix(".xlsx")
assert excel_file_path.exists(), "Excel data file doesn't exist"

# %% [markdown]
# ## Create DB connection

# %%
con = sqlite3.connect(db_path)
cur = con.cursor()

# %% [markdown]
# ## Create table to store the original data

# %%
sql = """
CREATE TABLE original_data (
    id INTEGER PRIMARY KEY,
    invoice_no TEXT NOT NULL,
    stock_code TEXT NOT NULL,
    description TEXT,
    quantity INTEGER NOT NULL,
    invoice_date TEXT NOT NULL,
    unit_price REAL NOT NULL,
    customer_id INTEGER,
    country TEXT
);
"""
cur = cur.execute(sql)

# %%
# Check if table was created
res = cur.execute("SELECT name FROM sqlite_master;")
assert res.fetchone() == ("original_data",), "failed to create table"

# %% [markdown]
# ## Read original data file

# %%
# Read every column as a set of strings
cols = [
    "InvoiceNo",
    "StockCode",
    "Description",
    "Quantity",
    "InvoiceDate",
    "UnitPrice",
    "CustomerID",
    "Country",
]
df = pl.read_excel(
    excel_file_path,
    schema_overrides={col: pl.String for col in cols},
)

# %% [markdown]
# ## Missing values

# %%
df.null_count()

# %% [markdown]
# ## Checking and fixing the data
# ### `InvoiceNo`
#
# `InvoiceNo` cannot be converted into an integer column. Some values are
# not numbers:

# %%
tmp_df = df.filter(pl.col("InvoiceNo").str.find(r"^\d+$").is_null())
tmp_df.head()

# %%
tmp_df.height

# %%
new_col = df.get_column("InvoiceNo").str.strip_chars()
assert new_col.str.len_chars().gt(0).all(), "found empty string"
df = df.replace_column(df.get_column_index("InvoiceNo"), new_col)

# %% [markdown]
# ### `StockCode`
#
# `StockCode` will also remain as a string column. The corresponding values
# contain numbers and letters. Making sure all of these strings are
# non-empty:

# %%
new_col = df.get_column("StockCode").str.strip_chars()
assert new_col.str.len_chars().gt(0).all(), "found empty string"
df = df.replace_column(df.get_column_index("StockCode"), new_col)

# %% [markdown]
# ### `Description`
#
# `Description` is obviously a string column. Then I'll just normalize the
# corresponding data.

# %%
# Normalization
new_col = df.get_column("Description").str.strip_chars()
new_col = new_col.str.strip_chars_end(".")
new_col = new_col.str.replace_all(r" {2,}", " ")
new_col = new_col.str.to_uppercase()

# %%
# Non-null values are non-empty strings
assert new_col.drop_nulls().str.len_chars().gt(0).all(), "found empty string"

# %%
df = df.replace_column(df.get_column_index("Description"), new_col)

# %% [markdown]
# ### `Quantity`
#
# `Quantity` should be an integer column. Its values can be positive or
# negative. First, I'll check that this is actually the case:

# %%
new_col = df.get_column("Quantity").str.strip_chars()
assert new_col.str.contains(r"^-?\d+$").all(), "found non-integer value"

# %% [markdown]
# Next, we convert `Quantity` into an integer column:

# %%
new_col = new_col.str.to_integer()
df = df.replace_column(df.get_column_index("Quantity"), new_col)

# %% [markdown]
# ### `InvoiceDate`
#
# `InvoiceDate` should contain a date and a time. We begin by checking if
# this is true:

# %%
new_col = df.get_column("InvoiceDate").str.strip_chars()
assert new_col.str.contains(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$").all(), "found invalid date"

# %% [markdown]
# Then we can convert `InvoiceDate` to a more appropriate type:

# %%
new_col = new_col.str.to_datetime("%Y-%m-%d %H:%M:%S")
df = df.replace_column(df.get_column_index("InvoiceDate"), new_col)

# %% [markdown]
# ### `UnitPrice`
#
# `UnitPrice` is supposed to be a column with real numbers. First, we check
# if this is the case:

# %%
new_col = df.get_column("UnitPrice").str.strip_chars()
assert new_col.str.contains(r"^-?\d+(\.\d+)?$").all(), "found invalid price"

# %% [markdown]
# So it's safe to convert these prices into real numbers:

# %%
new_col = new_col.cast(pl.Float64)
df = df.replace_column(df.get_column_index("UnitPrice"), new_col)

# %% [markdown]
# ### `CustomerID`
#
# `CustomerID` has missing values. However, the values that are *not*
# missing should be integers. Let's confirm that:

# %%
new_col = df.get_column("CustomerID").str.strip_chars()
assert new_col.drop_nulls().str.contains(r"^\d+$").all(), "found non-integer ID"

# %% [markdown]
# Next, we convert `CustomerID` into an integer column:

# %%
new_col = new_col.str.to_integer()
df = df.replace_column(df.get_column_index("CustomerID"), new_col)

# %% [markdown]
# ### `Country`
#
# `Country` is definitely a string column. So it's not necessary to change its
# type. However, there's a change I want to make. As it is, this column has no
# missing value. But that's not really true. Some values are "Unspecified":

# %%
new_col = df.get_column("Country").str.strip_chars()
tmp = new_col.filter(new_col.eq("Unspecified"))
tmp.head()

# %%
tmp.len()

# %% [markdown]
# I want to make it clear that these are missing values. This can be done as
# follows:

# %%
new_col = new_col.set(new_col.eq("Unspecified"), None)
assert new_col.null_count() == tmp.len(), "wrong transformation"

# %% [markdown]
# Finally, we can replace the `Country` column:

# %%
assert new_col.drop_nulls().str.len_chars().gt(0).all(), "found empty string"
df = df.replace_column(df.get_column_index("Country"), new_col)

# %% [markdown]
# ### Rename DataFrame columns


# %%
def pascal_to_snake(s: str) -> str:
    """Convert from Pascal case to snake case."""
    return re.sub(r"(?<!^)(?=[A-Z])", "_", s).lower()


# %%
# New column names to match the SQL table
snake_cols = [pascal_to_snake(col) for col in cols]
snake_cols[cols.index("CustomerID")] = "customer_id"

# %%
df = df.rename({old_name: new_name for old_name, new_name in zip(cols, snake_cols)})
assert df.columns == snake_cols, "failed to rename columns"

# %%
df.head()

# %% [markdown]
# ## Adapter for `datetime.datetime`


# %%
def adapt_datetime_iso(val: datetime.datetime) -> str:
    """Adapt datetime.datetime to timezone-naive ISO 8601 date."""
    return val.isoformat(sep=" ")


sqlite3.register_adapter(datetime.datetime, adapt_datetime_iso)

# %% [markdown]
# ## Insert data into DB
#
# Finally, we can insert the data into the database. Probably, this is not
# the most efficient way to do so. But we're going to do this only once. So
# possible inefficiency is not a big deal.

# %%
# Build SQL statement
col_names = ", ".join(snake_cols)
placeholders = ", ".join(f":{col}" for col in snake_cols)
sql = f"INSERT INTO original_data ({col_names}) VALUES ({placeholders});"

# %%
# Execute statement
cur.executemany(sql, df.to_dicts())
con.commit()

# %%
res = cur.execute("SELECT COUNT(*) FROM original_data;")
assert res.fetchone() == (df.height,), "failed to insert all rows"

# %% [markdown]
# ## Close DB connection

# %%
con.commit()  # Just to be sure
con.close()
