import functools
from collections.abc import Callable

import pandas as pd


def count_dropped(func: Callable) -> Callable:
    """Decorator that prints the number of rows dropped"""

    @functools.wraps(func)
    def wrapper(df: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
        size_before = df.shape[0]
        df = func(df, *args, **kwargs)
        num_dropped = size_before - df.shape[0]
        print(f"Number of observations dropped: {num_dropped}")
        return df

    return wrapper
