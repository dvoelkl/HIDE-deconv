"""
=====================================================
Methods for loading and preprocessing of bulk data.
=====================================================
"""

import anndata as ad
import pandas as pd
import scanpy as sc

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


def get_common_genes(
    adata: ad.AnnData, bulk: pd.DataFrame, remove_zero_median=True
) -> list[str]:
    """
    Return genes present in both single-cell and bulk expression data.

    The function computes the intersection between gene names in adata.var_names
    and the bulk expressions.

    Additionally remove genes, which have a median of 0 in the bulk, as these can influence domain transfer.

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

    if remove_zero_median:
        bulk_med = bulk.median(axis=1)
        zero_genes = bulk_med[bulk_med == 0].index
        if len(zero_genes) > 0:
            bulk = bulk.drop(index=zero_genes)

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

    mean_df1 = df1.median(axis=1)
    mean_df2 = df2.median(axis=1)

    alpha = mean_df1 / mean_df2

    return alpha


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


def combine_bulk_dataframes(
    data_frames: list[pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Combines a list of bulk RNA seq dataframes and corrects for batch effects using ComBat via scanpy.

    Parameters
    ----------
    data_frames : list[pd.DataFrame]
        List of bulk RNA seq dataframes that should be combined.
    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        DataFrame, containing the bulk labels as columns and the merged genes as rows and DataFrame containing assignment to original batches.

    References
    ----------
    - Wolf, F. A., Angerer, P., & Theis, F. J. (2018). SCANPY: large-scale single-cell gene expression data analysis. Genome biology, 19(1), 15. https://link.springer.com/article/10.1186/s13059-017-1382-0.

    """
    common_genes = data_frames[0].index
    for df in data_frames[1:]:
        common_genes = common_genes.intersection(df.index)

    if len(common_genes) == 0:
        raise ValueError("No shared genes found between input bulk dataframes.")

    combined = pd.concat([df.loc[common_genes] for df in data_frames], axis=1)

    batch = []
    for i, df in enumerate(data_frames):
        batch.extend([f"batch_{i}" for _ in range(df.shape[1])])

    batch_df = pd.DataFrame(batch, columns=["batch"], index=combined.columns)

    adata = ad.AnnData(
        X=combined.values.T,
        obs=batch_df,
        var=pd.DataFrame(index=combined.index),
    )

    sc.pp.combat(adata, key="batch", inplace=True)

    combined_corrected = pd.DataFrame(
        adata.X.T,
        index=combined.index,
        columns=combined.columns,
    )

    return combined_corrected, batch_df
