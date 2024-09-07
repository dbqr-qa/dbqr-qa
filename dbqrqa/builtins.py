import math
import re
from typing import Any, List, Tuple, Union

import numpy as np
import pandas as pd

from dbqrqa.utils import DATA, function, remove_nan as clean_nan


def _const_val(source: str, val: str, step: DATA = None) -> float:
    if source == 'custom':
        val = val.strip().replace(' ', '').lower()
        
        if len(val) == 0 or val == '-' or val == 'nan':
            val = np.nan
        
        elif val[-1] == '%':
            val = float(val[:-1])
        
        elif val[-1] in 'kmbt':
            mul = 1e3 ** ('kmbt'.index(val[-1]) + 1)
            val = float(val[:-1]) * mul
        
        else:
            val = float(val)
    
    else:
        if (isinstance(step, pd.Series)):
            val = step.iloc[0]
        
        else:
            val = step.iloc[0, 0]
    
    return val


def _const_loc(df: DATA, idx: str, columns: str) -> DATA:
    idxs = pd.IndexSlice
    if df.index.nlevels > 1: idx = eval(f'idxs[{idx}]')

    if isinstance(df, pd.Series):
        return eval(f'df.loc[{idx}]')
    
    else:
        if df.columns.nlevels > 1: 
            columns = eval(f'idxs[{columns}]')

        return eval(f'df.loc[{idx}, {columns}]')


@function(output_df=False)
def loc(
    df: DATA,
    idx: str = ':',
    columns: str = ':') -> DATA:
    """Select"""
    idxs = pd.IndexSlice
    if df.index.nlevels > 1: idx = eval(f'idxs[{idx}]')

    if isinstance(df, pd.Series):
        do = eval(f'df.loc[{idx}]')

    else:
        if df.columns.nlevels > 1: 
            columns = eval(f'idxs[{columns}]')

        do = eval(f'df.loc[{idx}, {columns}]')

    return clean_nan(do, verbose=False)


@function(output_df=False)
def iloc(
    df: DATA,
    idx: str = ':',
    columns: str = ':') -> DATA:
    """Select"""
    if isinstance(df, pd.Series):
        do = eval(f'df.iloc[{idx}]')

    else:
        do = eval(f'df.iloc[{idx}, {columns}]')

    return clean_nan(do, verbose=False)
    

@function
def loc_filter(
    df: DATA,
    axis: str = 'rows',
    operation: str = 'all',
    condition: str = 'equal',
    source: str = 'custom',
    val: str = None,
    step: DATA = None,
    idx: str = ':',
    columns: str = ':') -> DATA:
    """Filter the data by a condition"""

    val = _const_val(source, val, step)
    axis = 'index' if axis == 'rows' else 'columns'

    idxs = pd.IndexSlice
    if df.index.nlevels > 1: idx = eval(f'idxs[{idx}]')

    if isinstance(df, pd.Series):
        ds = eval(f'df.loc[{idx}]')

    else:
        if df.columns.nlevels > 1: 
            columns = eval(f'idxs[{columns}]')

        ds = eval(f'df.loc[{idx}, {columns}]')

    if condition == 'equal': dt = ds == val
    elif condition == 'gt': dt = ds > val
    elif condition == 'lt': dt = ds < val
    elif condition == 'ge': dt = ds >= val
    elif condition == 'le': dt = ds <= val

    if (isinstance(df, pd.Series)):
        return df[dt], df.index[dt]

    else:
        dts = getattr(dt, operation)(axis=axis)
        do = df.T[dts].T if axis == 'index' else df[dts]
        return do, do.index


@function(output_df=False)
def merge(
    df: DATA,
    dfs: List[DATA],
    axis: str = 'rows',
    ignore_index: bool = False,
    replace_columns: bool = True) -> DATA:
    """Concatenate data frames"""

    if isinstance(df, pd.Series) or \
        any(isinstance(d, pd.Series) for d in dfs):

        if isinstance(df, pd.DataFrame):
            df = df.iloc[:,0]

        for i in range(len(dfs)):
            if isinstance(dfs[i], pd.DataFrame):
                dfs[i] = dfs[i].iloc[:,0]

        return pd.concat([df] + dfs, ignore_index=ignore_index)

    else:
        if axis == 'rows':
            if replace_columns:
                for d in dfs:
                    d.columns = df.columns

            return pd.concat([df] + dfs, axis=0, ignore_index=ignore_index)
        
        else: 
            return pd.concat([df] + dfs, axis=1, ignore_index=ignore_index)


@function
def intersect(
    df: DATA,
    dfs: List[DATA],
    axis: str = 'rows',
    level: int = None) -> DATA:
    """Intersect data frames"""
    idx = df.index

    if level is not None:
        didx = df.index.to_list()

        for d in dfs:
            if isinstance(d, list) or isinstance(d, tuple):
                refs = d

            elif isinstance(d.index[0], str):
                refs = d.index.to_list()
            
            else:
                refs = [e[level] for e in d.index.to_list()]

            fidx = []

            for drx in didx:
                if drx[level] in refs:
                    fidx.append(drx)
            
            didx = fidx

        idx = pd.MultiIndex.from_tuples(didx)
    
    else:
        for d in dfs:
            idx = idx.intersection(d.index)
    
    return df.loc[idx], idx


@function
def union(
    df: DATA,
    dfs: List[DATA]) -> DATA:
    """Union data frames"""
    idx = df.index
    do = pd.concat([df] + dfs)

    for d in dfs:
        idx = idx.union(d.index)
    
    return do.loc[idx], idx


@function
def exclude(
    df: DATA,
    dfs: List[DATA]) -> DATA:
    """Exclude rows from a dataframe"""
    idx = df.index

    for d in dfs:
        idx = idx.drop(d.index)
    
    return df.loc[idx], idx


@function(output_df=False)
def headers(
    df: DATA,
    axis: str = 'rows',
    level: int = None) -> list:
    """Return the headers"""

    def group(idx: list) -> list:
        output = []

        for l in range(len(idx[0])):
            items = set()

            for i in range(len(idx)):
                items.add(idx[i][l])
            
            output.append(list(items))
        
        return output

    if isinstance(df, pd.Series) or axis == 'rows':
        if df.index.nlevels == 1:
            return df.index.tolist()

        else:
            if level is None:
                return group(df.index.tolist())

            else:
                return group(df.index.tolist())[level]
    
    else:
        if df.columns.nlevels == 1:
            return [h[0] for h in df.columns.tolist()]

        else:
            if level is None:
                return group(df.columns.tolist())

            else:
                return group(df.columns.tolist())[level]
            

@function(output_df=False)
def tolist(df: DATA) -> list:
    """Convert the data frame into a list"""
    return df.values.tolist()


@function
def abs(df: DATA) -> DATA:
    """Calculate the absolute values"""
    return df.abs(), df.index


@function
def sum(
    df: DATA, 
    axis: str = 'rows',
    level: int = None,
    keep_col: bool = False) -> DATA:
    """Sum"""
    if isinstance(df, pd.Series): 
        if level is None:
            ds = pd.Series([df.sum(),], index=(('Sum',),))
            return ds, ds.index

        else:
            ds = df.groupby(level=level).sum()
            return ds, ds.index
    
    elif level is None:
        if axis == 'rows': do, idx = df.sum(axis=0), df.columns
        else: do, idx = df.sum(axis=1), df.index
    
    else:
        if axis == 'rows': 
            do, idx = df.groupby(level=level, axis=0).sum(), df.columns
        
        else: 
            do, idx = df.groupby(level=level, axis=1).sum(), df.index

    if keep_col and isinstance(do, pd.Series) or isinstance(do, tuple):
        return pd.DataFrame(do, columns=(('Sum',),)), idx

    else:
        return do, idx


@function
def diff(df: DATA, axis: str = 'rows') -> DATA:
    """Subtract"""
    if isinstance(df, pd.Series): 
        return df.diff().iloc[1:], df.index[1:]
    
    elif axis == 'rows': 
        return df.diff(axis=0).iloc[1:, :], df.columns[1:]
    
    else: 
        return df.diff(axis=1).iloc[:, 1:], df.index[1:]


@function
def div(df: DATA, axis: str = 'rows', keep_col: bool = False) -> DATA:
    """Divide"""
    if keep_col:
        if isinstance(df, pd.Series): 
            dt = df.copy()

            for i in range(1, len(df.index)):
                dt.iloc[i] = df.iloc[i] / df.iloc[i - 1]

            return dt[1:], dt.index[1:]

        elif axis == 'rows': 
            dt = df.copy()

            for i in range(1, len(df.columns)):
                dt.iloc[:, i] = df.iloc[:, i] / df.iloc[:, i - 1]
            
            return dt.iloc[:, 1:], dt.columns[1:]

        else: 
            dt = df.copy()

            for i in range(1, len(df.index)):
                dt.iloc[i, :] = df.iloc[i, :] / df.iloc[i - 1, :]
            
            return dt.iloc[1:, :], dt.index[1:]
        
    else:
        if isinstance(df, pd.Series): 
            return df.iloc[1] / df.iloc[0], (('Divide',),)

        elif axis == 'rows': 
            return df.iloc[:, 1] / df.iloc[:, 0], df.columns[1:]

        else: 
            return df.iloc[1, :] / df.iloc[0, :], df.index[1:]


@function
def mean(
    df: DATA, 
    axis: str = 'rows',
    level: int = None,
    keep_col: bool = False,
    skipna: bool = True) -> DATA:
    """Average"""
    if level is None:
        if isinstance(df, pd.Series): 
            return df.mean(skipna=skipna), (('Average',),)

        else:
            if axis == 'rows': 
                do, idx = df.mean(axis=0, skipna=skipna), df.columns

            else: 
                do, idx = df.mean(axis=1, skipna=skipna), df.index
    
    else:
        if isinstance(df, pd.Series): 
            return df.groupby(level=level).mean(), (('Average',),)
    
        else:
            if axis == 'rows': 
                do = df.groupby(level=level, axis=0).mean()
                idx = df.columns
            
            else: 
                do = df.groupby(level=level, axis=1).mean()
                idx = df.index

    if keep_col and isinstance(do, pd.Series) or isinstance(do, tuple):
        return pd.DataFrame(do, columns=(('Average',),)), idx

    else:
        return do, idx


@function
def sort(
    df: DATA, 
    by: Union[str, List[Union[str, Tuple[str]]]] = None,
    axis: str = 'rows',
    ascending: bool = False) -> DATA:
    """Sort"""
    if isinstance(df, pd.Series) or by is None:
        df = df.sort_values(ascending=ascending)
        return df, df.index

    if len(by) == 0: return df, df.index
    elif isinstance(by[0], list): by = [tuple(x) for x in by]

    by = eval(by) if isinstance(by, str) else list(by)
    if axis == 'rows': df = df.sort_values(by=by, axis=0, ascending=ascending)
    else: df = df.sort_values(by=by, axis=1, ascending=ascending)
    
    return df, df.index


@function
def k_end(
    df: DATA,
    k: Union[int, str],
    direction: str = 'first',
    axis: str = 'rows',
    reverse: bool = False) -> DATA:
    """Select first results"""
    if isinstance(k, str): 
        k = k.strip().replace(' ', '').lower()
        
        if re.match(r'^[0-9]+$', k):
            k = int(k)
        
        elif re.match(r'^[0-9]+[kmbt]$', k):
            mul = 1e3 ** ('kmbt'.index(k[-1]) + 1)
            k = int(int(k[:-1]) * mul)
        
        elif re.match(r'^[0-9]+\.*[0-9]*%$', k):
            if axis == 'rows': size = df.shape[1]
            else: size = df.shape[0]
            k = int(size * float(k[:-1]) / 100)

        else:
            raise RuntimeError('Invalid input')

    if reverse:
        k = -k
        direction = 'last' if direction == 'first' else 'first'

    if direction == 'first':
        if isinstance(df, pd.Series): return df.iloc[:k], df.index
        elif axis == 'rows': return df.iloc[:, :k], df.index
        else: return df.iloc[:k, :], df.index

    elif direction == 'last':
        if isinstance(df, pd.Series): return df.iloc[-k:], df.index
        elif axis == 'rows': return df.iloc[:, -k:], df.index
        else: return df.iloc[-k:, :], df.index


@function
def nth(
    df: DATA,
    n: int,
    direction: str = 'first',
    axis: str = 'rows') -> DATA:
    """Select the nth item"""
    mul = 1 if direction == 'first' else -1
    adj = 0 if direction == 'first' else -1

    if isinstance(df, pd.Series):
        val = df.iloc[(n - 1) * mul + adj]
        df = pd.Series([val,], index=[df.index[(n - 1) * mul + adj],])
        return df, df.index
    
    elif axis == 'rows': 
        dt = df.iloc[:, (n - 1) * mul + adj]
        df = pd.DataFrame(dt)
        return df, df.index

    else: 
        dt = df.iloc[(n - 1) * mul + adj, :]
        df = pd.DataFrame(dt).T
        return df, df.index
    
@function
def compare(
    df: DATA,
    axis: str = 'rows',
    greater: str = 'greater', 
    lower: str = 'lower', 
    equal: str = 'equal') -> str:
    """Compare the values, answer "greater", "lower" or "equal\""""

    def check(value: float) -> str:
        if value > 0: return greater
        elif value < 0: return lower
        elif value == 0: return equal
        else: return np.nan

    if isinstance(df, pd.Series):
        ds = df.diff().iloc[1:]
        return ds.map(check), df.index
    
    else:
        if axis == 'rows': 
            ds = df.diff(axis=0).iloc[1:, :]
            return ds.applymap(check), df.columns

        else: 
            ds = df.diff(axis=1).iloc[:, 1:]
            return ds.applymap(check), df.index
        

@function
def count(
    df: DATA,
    axis: str = 'rows') -> pd.Series:
    """Return the dimension of the data frame"""

    if isinstance(df, list): size = len(df)
    elif isinstance(df, pd.Series): size = len(df)
    elif axis == 'rows': size = len(df.columns)
    elif axis == 'columns': size = len(df.index)

    ds = pd.Series([size], index=(('Count',),))
    return ds, ds.index
        

@function(output_df=False)
def equal(
    df: DATA,
    axis: str = 'rows',
    yes: str = 'yes', 
    no: str = 'no') -> str:
    """Check whether the two values are equal, answer "yes" or "no\""""
    
    def check(value: float) -> str:
        if value == 0: return yes
        elif math.isnan(value): return np.nan
        else: return no

    if isinstance(df, pd.Series):
        ds = df.diff().iloc[1:]
        return ds.map(check), df.index
    
    else:
        if axis == 'rows': 
            ds = df.diff(axis=0).iloc[:, 1:]
            return ds.applymap(check), df.columns

        else: 
            ds = df.diff(axis=1).iloc[1:, :]
            return ds.applymap(check), df.index
        

@function
def stack(
    df: DATA,
    level: int = -1) -> DATA:
    """Stack the data frame"""
    if isinstance(df, pd.Series): return df, df.index

    if isinstance(df.columns.values[0], tuple) or \
        isinstance(df.columns.values[0], list):
        columns = ['_'.join(str(c) for c in col) \
            for col in df.columns.values]
    
    else:
        columns = df.columns.values
    
    df = df.T.reset_index(drop=True).T
    df.columns = columns
    df = df.stack(level=level)
    return df, df.index


@function
def replace(
    df: DATA,
    value: Any,
    target: Any):

    if value == 'nan':
        df = df.fillna(target)

    else:
        df = df.fillna(float('inf'))
        df[df != float('inf')] = target
        df[df == float('inf')] = np.nan
    
    return df, df.index


@function
def cond(
    df: DATA,
    condition: str = 'equal',
    cond_source: str = 'custom',
    cond_val: str = None,
    cond_step: DATA = None,
    target_source: str = 'custom',
    target_val: str = None,
    target_step: DATA = None) -> DATA:
    """Replace the values by a condition"""
    
    cond_val = _const_val(cond_source, cond_val, cond_step)
    target_val = _const_val(target_source, target_val, target_step)

    if condition == 'equal': df[df == cond_val] = target_val
    elif condition == 'gt': df[df > cond_val] = target_val
    elif condition == 'lt': df[df < cond_val] = target_val
    elif condition == 'ge': df[df >= cond_val] = target_val
    elif condition == 'le': df[df <= cond_val] = target_val
    elif condition == 'ne': df[df != cond_val] = target_val

    return df, df.index


@function
def transpose(
    df: DATA) -> DATA:
    """Transpose the data frame"""

    if isinstance(df, pd.Series):
        ds = pd.DataFrame([df.values,], columns=df.index)

    else:
        ds = df.T

    return ds, ds.index


@function
def const_add(
    df: DATA,
    idx: str = ':',
    columns: str = ':',
    source: str = 'custom',
    val: str = None,
    step: DATA = None,
    table: DATA = None) -> DATA:
    """Add a constant to the data frame"""
    if source == 'table':
        df = df.add(table)
    
    else:
        val = _const_val(source, val, step)
        dt = _const_loc(df, idx, columns) + val
        df.update(dt)

    return df, df.index


@function
def const_subtract(
    df: DATA,
    idx: str = ':',
    columns: str = ':',
    source: str = 'custom',
    val: str = '0',
    step: DATA = None) -> DATA:
    """Stract a constant from the data frame"""
    val = _const_val(source, val, step)
    dt = _const_loc(df, idx, columns) - val
    df.update(dt)
    return df, df.index


@function
def const_mul(
    df: DATA,
    idx: str = ':',
    columns: str = ':',
    source: str = 'custom',
    val: str = '1',
    step: DATA = None) -> DATA:
    """Multiply a constant to the data frame"""
    val = _const_val(source, val, step)
    dt = _const_loc(df, idx, columns) * val
    df.update(dt)
    return df, df.index


@function
def const_div(
    df: DATA,
    idx: str = ':',
    columns: str = ':',
    source: str = 'custom',
    val: str = '1',
    step: DATA = None) -> DATA:
    """Divide the data frame by a constant"""
    val = _const_val(source, val, step)
    dt = _const_loc(df, idx, columns) / val
    df.update(dt)
    return df, df.index


@function
def remove_nan(df: DATA, axis: str = 'rows') -> DATA:
    """Remove rows with all null values"""

    if axis == 'rows':
        cols = []

        for c in range(len(df.columns)):
            remove = True

            for r in range(len(df)):
                if not pd.isnull(df.iloc[r, c]):
                    remove = False
            
            if remove:
                cols.append(c)

        dt = df.copy().drop(cols, axis=1)
        
    else:
        rows = []

        for r in range(len(df)):
            remove = True

            for c in range(len(df.columns)):
                if not pd.isnull(df.iloc[r, c]):
                    remove = False
            
            if remove:
                rows.append(r)
        
        dt = df.copy().drop(rows, axis=0)
    
    return dt, dt.index


@function(input_df=False, output_df=False)
def choices(choice: str = 'yes') -> str:
    """Choose of choices"""
    return choice


@function(output_df=False)
def quantify(df: DATA, unit: str) -> str:
    """Quantify in unit"""
    if isinstance(df, pd.Series): num = df.iloc[0]
    else: num = df.iloc[0, 0]
    
    if unit == 'percent': num *= 100
    elif unit == 'thousands': num /= 1e3
    elif unit == 'millions': num /= 1e6
    elif unit == 'billions': num /= 1e9
    elif unit == 'trillions': num /= 1e12
    else: raise KeyError(f'Unknown unit: {unit}')

    return '%.2f %s' % (num, unit)


@function(output_df=False)
def reverse(df: DATA, axis: str = 'rows') -> DATA:
    if isinstance(df, pd.Series): return df.iloc[::-1]
    elif axis == 'rows': return df.iloc[:,::-1]
    else: return df.iloc[::-1]


@function(input_df=False, output_df=False)
def empty(text: str = 'None') -> str:
    """Indicate that the answer is empty"""
    return text


@function(input_df=False, output_df=False)
def ambiguous(text: str = 'terminate:ambiguous') -> str:
    """Indicate that the answer is ambiguous"""
    return text


@function(input_df=False, output_df=False)
def invalid_data(text: str = 'terminate:invalid_data') -> str:
    """Indicate that the queries are invalid"""
    return text


@function(input_df=False, output_df=False)
def other_problems(text: str = 'terminate:unsolvable') -> str:
    """Indicate that the answer is unsolvable"""
    return text
