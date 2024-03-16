# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Online Retail Dataset: Download Dataset
# ## Imports

# %%
import zipfile
from pathlib import Path

import requests

# %% [markdown]
# ## Download zip file

# %%
data_dir = Path.cwd().parents[1] / "data"
if not data_dir.exists():
    data_dir.mkdir()

# %%
url = "https://archive.ics.uci.edu/static/public/352/online+retail.zip"

zip_name = url.split("/")[-1].replace("+", "_")
zip_path = data_dir / zip_name

# %%
assert not zip_path.with_suffix(".xlsx").exists(), "dataset already exists"

# %%
response = requests.get(url, stream=True)  # noqa: S113
assert response.status_code == 200, "failed to download zip file"

# %%
with zip_path.open("wb") as file:
    for chunk in response.iter_content(chunk_size=1024):
        file.write(chunk)

# %% [markdown]
# ## Extract zip file

# %%
old_name = "Online Retail.xlsx"
with zipfile.ZipFile(zip_path, "r") as zip_file:
    zip_file.extract(old_name, path=data_dir)

# %%
# Delete zip file
zip_path.unlink()

# %%
# Rename extracted file
old_file_path = data_dir / old_name

new_name = old_name.lower().replace(" ", "_")
new_file_path = old_file_path.with_name(new_name)

_ = old_file_path.rename(new_file_path)
