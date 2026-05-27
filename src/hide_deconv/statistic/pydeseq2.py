"""
=====================================================
PyDESeq2-based differential expression analysis

Uses the DESeq2 implementation of scverse (https://github.com/scverse/PyDESeq2)
=====================================================
"""

from pathlib import Path

import pandas as pd

from pydeseq2.dds import DeseqDataSet
from pydeseq2.ds import DeseqStats

from ..visualization import plot_volcano


def pydeseq2_preprocess(
    bulk: pd.DataFrame,
    sample_sheet: pd.DataFrame,
    sample_id_col: str,
    condition_col: str,
    covariates: list[str] | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Prepare bulk and sample metainfo for PyDESeq2.

    Parameters
    ----------
    bulk : pd.DataFrame
        Bulk RNA-seq file (genes x samples)
    sample_sheet : pd.DataFrame
        Sample sheet containing the metainformation on samples (samples x info)
    sample_id_col : str
        Column name of the sample sheet containing the sample ids of the bulk file
    condition_col : str
        Column name of the sample sheet containing the conditions
    covariates : list[str] | None
        List of column names containing possible covariates

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        Counts, Metainformation to be used with the run_pydeseq2 function
    """

    sample_cols = [sample_id_col, condition_col]
    if covariates:
        sample_cols.extend(covariates)

    samples = sample_sheet[sample_cols].copy()
    samples = samples.dropna(subset=[sample_id_col, condition_col])

    if covariates:
        samples = samples.dropna(subset=covariates)

    samples[sample_id_col] = samples[sample_id_col].astype(str)
    samples = samples.drop_duplicates(subset=[sample_id_col], keep="first")
    bulk = bulk.copy()
    bulk.columns = bulk.columns.astype(str)
    samples = samples[samples[sample_id_col].isin(bulk.columns)]

    if len(samples) == 0:
        raise ValueError("No matching sample ids were found in the bulk file.")

    samples = samples.set_index(sample_id_col).reindex(bulk.columns)
    samples = samples.dropna(subset=[condition_col])

    if covariates:
        samples = samples.dropna(subset=covariates)

    counts = bulk.loc[:, samples.index].T.copy()
    counts.index = counts.index.astype(str)
    counts.columns = counts.columns.astype(str)

    metadata = samples.copy()
    metadata.index = metadata.index.astype(str)
    metadata[condition_col] = metadata[condition_col].astype(str)

    return counts, metadata


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


def run_pydeseq2(
    bulk: pd.DataFrame,
    metadata: pd.DataFrame,
    condition_col: str,
    tested_condition: str,
    reference_condition: str,
    covariates: list[str] | None,
    out_path: Path,
) -> pd.DataFrame:
    """
    Run PyDESeq2

    Parameters
    ----------
    bulk : pd.DataFrame
        Raw count bulk RNA-seq data (Samples x Genes)
    metadata : pd.DataFrame
        Sample metainformation
    condition_col : str
        Name of the column in metadata holding the condition
    tested_condition : str
        Name of the condition that will be tested
    reference_condition : str
        Name of the condition that is used as reference
    covariates : list[str]
        Column names of covariates to include
    out_path : Folder path, where all results and plots are stored

    Returns
    -------
    pd.DataFrame
        pydeseq2 result dataframe
    """

    design_terms = [condition_col]
    if covariates:
        design_terms.extend(covariates)

    dds = DeseqDataSet(
        counts=bulk,
        metadata=metadata,
        design="~ " + " + ".join(design_terms),
        quiet=True,
    )
    dds.deseq2()

    deseq_stats = DeseqStats(
        dds,
        contrast=[condition_col, tested_condition, reference_condition],
        quiet=True,
    )
    deseq_stats.summary()

    results = deseq_stats.results_df.copy()

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    results.to_csv(out_path.with_name(f"{out_path.name}_results.csv"))
    deseq_stats.plot_MA(save_path=str(out_path.with_name(f"{out_path.name}_ma.png")))
    plot_volcano(results, out_path.with_name(f"{out_path.name}_volcano.png"))

    return results
