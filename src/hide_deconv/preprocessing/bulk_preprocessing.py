"""
=====================================================
Methods for loading and preprocessing of bulk data.
=====================================================
"""

import anndata as ad
import pandas as pd

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


def get_common_genes(adata: ad.AnnData, bulk: pd.DataFrame) -> list[str]:
    """
    Return genes present in both single-cell and bulk expression data.

    The function computes the intersection between gene names in adata.var_names
    and the bulk expressions

    Parameters
    ----------
    adata : ad.AnnData
        AnnData object containing single-cell expression data.
        Gene names are taken from adata.var_names

    bulk : pd.DataFrame
        Bulk expression profiles with genes as index and samples as columns

    Returns
    -------
    list[str]
        List of shared gene names

    """

    genes_sc = set(adata.var_names)
    genes_bulk = set(bulk.index)
    common_genes = list(genes_sc.intersection(genes_bulk))

    return common_genes


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


def get_domain_transfer_factor(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.Series:
    """
    Calculate the domain transfer factor to align two bulks

    Parameters
    ----------
    df1 : pd.DataFrame
        First dataframe
    df2 : pd.DataFrame
        Second dataframe

    Returns
    -------
    pd.Series
        Series containing the conversion factors for each gene
    """
    # \sum_g df1_g = \alpha \sum_g df2_g

    mean_df1 = df1.mean(axis=1)
    mean_df2 = df2.mean(axis=1)

    alpha = mean_df1 / mean_df2

    return alpha
