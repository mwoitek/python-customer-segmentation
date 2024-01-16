# %% [markdown]
# # Online Retail Dataset

# %% [markdown]
# ## Imports

# %%
from pathlib import Path

import pandas as pd

# %% [markdown]
# ## Read dataset

# %%
# File path for dataset
file_path = Path.cwd() / "data" / "online_retail.xlsx"
assert file_path.exists(), "file doesn't exist"
assert file_path.is_file(), "not a file"

# %%
df = pd.read_excel(file_path)
df.info()
