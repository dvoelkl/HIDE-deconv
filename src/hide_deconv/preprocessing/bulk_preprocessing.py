"""
=====================================================
Methods for loading and preprocessing of bulk data.
=====================================================
"""

import anndata as ad
import pandas as pd
from inmoose.cohort_qc import QCReport, CohortMetric
from inmoose.pycombat import pycombat_seq

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

    mean_df1 = df1.median(axis=1)
    mean_df2 = df2.median(axis=1)

    alpha = mean_df1 / mean_df2

    return alpha


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


def combine_bulk_dataframes(
    data_frames: list[pd.DataFrame], quality_control_path: str = ""
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Combines a list of bulk RNA seq dataframes and corrects for domain transfer between them using ComBat-Seq using the python implementation of the inmoose package.

    Parameters
    ----------
    data_frames : list[pd.DataFrame]
        List of bulk RNA seq dataframes that should be combined.

    quality_control_path : str = ""
        If provided, creates a quality report using the inmoose quality control method and stores it as a html document under the given name.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        DataFrame, containing the bulk labels as columns and the merged genes as rows and DataFrame containing assignment to original batches.

    References
    ----------
    - Y. Zhang, G. Parmigiani, W. E. Johnson. 2020. ComBat-Seq: batch effect adjustment for RNASeq count data. NAR Genomics and Bioinformatics, 2(3). https://doi.org/10.1093/nargab/lqaa078.
    - Colange M, Appé G, Meunier L, Weill S, Johnson WE, Nordor A, Behdenna A. (2025) Bridging the gap between R and Python in bulk transcriptomic data analysis with InMoose. Nature Scientific Reports 15;18104. https://doi.org/10.1038/s41598-025-03376-y.

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

    combined_corrected = pycombat_seq(combined, batch=batch)

    if quality_control_path:
        cohort_metric = CohortMetric(
            batch_df,
            batch_column="batch",
            data_expression_df=combined_corrected,
            data_expression_df_before=combined,
        )
        cohort_metric.process()
        cohort_report = QCReport(cohort_metric)
        cohort_report.save_report(quality_control_path)

    return combined_corrected, batch_df
