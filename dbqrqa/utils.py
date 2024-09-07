from typing import Any, Callable, List, Union

import numpy as np
import pandas as pd
from tqdm import tqdm

DATA = Union[str, list, pd.DataFrame, pd.Series]


def todf(
    object: Union[float, pd.DataFrame, pd.Series],
    index: List[str]) -> DATA:

    if isinstance(object, float):
        return pd.Series([object], index=index[0])
    
    elif isinstance(object, pd.DataFrame) or isinstance(object, pd.Series):
        if len(object.index) == 0:
            raise RuntimeError('Empty data frame')
    
    return object


def function(
    _func: Callable = None,
    *,
    input_df: bool = True,
    output_df: bool = True):

    def _function(func):
        def wrapper(data: Any, *args, spec: bool = False, **kwargs):
            if spec:
                return func
            
            else:
                if isinstance(data, pd.DataFrame) \
                    or isinstance(data, pd.Series):
                    data = data.copy()

                if output_df:
                    output, idx = func(data, *args, **kwargs)
                    return todf(output, idx)
            
                else:
                    if input_df:
                        return func(data, *args, **kwargs)

                    else:
                        return func(*args, **kwargs)
        
        return wrapper
    
    if _func is None:
        return _function

    else:
        return _function(_func)


def remove_nan(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    if not isinstance(df, pd.DataFrame):
        return df

    include, idx = [], []
    nulls = df.isnull().sum(axis=1)

    for r in tqdm(range(
        len(df.index)), desc='Cleaning', disable=not verbose):

        if nulls[r] < len(df.columns):
            include.append(r)
            idx.append(df.index[r])
    
    dt = pd.DataFrame(
        data=np.empty((len(include), len(df.columns))) * np.nan,
        index=pd.MultiIndex.from_tuples(idx),
        columns=df.columns)

    for r in include:
        dt.loc[df.index[r]] = df.iloc[r]

    return dt
