"""Helper module for combining feature tables."""

from typing import Iterable
import pandas as pd

def combine_feature_tables(fts: Iterable[pd.DataFrame]) -> pd.DataFrame:
    """
    Take an Iterable of DataFrames and concatenate them.

    If all feature tables have an x_ROI or R column, the returned DataFrame will have those columns
    as well, where values are shifted from DataFrame to DataFrame accordingly.

    Parameters
    ----------
    fts : Iterable[pd.DataFrame]
        The iterable of DataFrames to concatenate.

    Returns
    -------
    ft: pd.DataFrame
        The combined feature table

    """
    have_x_ROI: bool = all(['x_ROI' in ft.columns for ft in fts])
    have_R: bool = all(['R' in ft.columns for ft in fts])
    x_ROI_offset = 0
    x_offset = 0
    R_max = 0
    fts_new = []
    for ft in fts:
        ft_new = ft.copy()
        ft_new.x += -ft_new.x.min() + x_offset
        x_offset = ft_new.x.max().copy()

        if have_x_ROI:
            ft_new.x_ROI += -ft_new.x_ROI.min() + x_ROI_offset
            x_ROI_offset = ft_new.x_ROI.max().copy()
        elif 'x_ROI' in ft.columns:
            ft_new.columns.drop(columns='x_ROI', inplace=True)

        if have_R:
            ft_new.R += R_max
            R_max = ft_new.R.max().copy() + 1
        elif 'R' in ft.columns:
            ft_new.drop(columns='R', inplace=True)

        fts_new.append(ft_new)

    df = pd.concat(fts_new, axis=0).reset_index(drop=True)

    return df
