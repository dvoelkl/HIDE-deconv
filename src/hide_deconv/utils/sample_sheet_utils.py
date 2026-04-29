"""
=====================================================
Utility functions for samples sheets
=====================================================
"""

import pandas as pd


def sample_ids_valid(sample_ids: list[str], bulk_names: list[str]) -> bool:
    """
    Checks if the sample ids from a sample sheet are compatible with the bulk names.

    Returns
    -------
    bool
        True, if bulk_names are subset of sample_ids.
    """

    return len(set(bulk_names).intersection(set(sample_ids))) > 0 # set(bulk_names).issubset(set(sample_ids))


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


def remove_nan_sample_ids(sample_ids: pd.Series) -> pd.Series:
    """
    Remove NaN entries from Sample Ids

    Parameters
    ----------
    sample_ids : pd.Series
        List of sample ids that should be used.

    Returns
    -------
    pd.Series
        List of sample ids without nan or empty values
    """

    sample_ids = sample_ids.dropna(inplace=False)

    return sample_ids


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


def filter_sample_sheet(
    sample_sheet: pd.DataFrame, sample_id_col: str
) -> tuple[pd.Series, pd.DataFrame]:
    """
    Filter a sample sheet, such that only non-NaN sample id's are kept.

    Parameters
    ----------
    sample_sheet : pd.DataFrame
        Sample sheet.
    sample_id_col : str
        Name of column that holds the sample ids.

    Returns
    -------
    tuple[pd.Series, pd.DataFrame]
        Valid Sample Ids, Sample Sheet only with non-NaN Sample Id Samples
    """

    ids = remove_nan_sample_ids(sample_sheet[sample_id_col])
    filtered_sample_sheet = sample_sheet[sample_sheet[sample_id_col].isin(ids.values)]

    return ids, filtered_sample_sheet
