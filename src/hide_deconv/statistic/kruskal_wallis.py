"""
=====================================================
Functions for Kruskal Wallis Test
=====================================================
"""

import pandas as pd
import numpy as np
from scipy.stats import kruskal, false_discovery_control

from rich.console import Console

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

console = Console()

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


def run_kruskal_wallis(
    bulks: pd.DataFrame, sample_list: pd.DataFrame, sample_id_col: str, cohort_col: str
) -> pd.DataFrame:
    """
    Performs a Kruskal Wallis Test with FDR correction.

    Parameters
    ----------
    bulks : pd.DataFrame
        bulk Samples (either Celltype x Sample or Gene x Sample)
    sample_list : pd.DataFrame
        sample list (samples x clinical variables)
    sample_id_col : str
        Name of the column, that links to the bulks
    cohort_col: str
        Name of the column containing the cohort identifiers.

    Returns
    -------
    pd.DataFrame
        DataFrame containing p-Value and adjusted p-Value
    """

    samples = sample_list[[sample_id_col, cohort_col]].set_index(sample_id_col)

    # Subset samples, as sample list might have more samples, than in deconvolution
    samples = samples.loc[bulks.columns]

    cohorts = samples[cohort_col].unique()

    fWarning = False
    for cohort in cohorts:
        if (samples[cohort_col] == cohort).sum() < 5:
            console.print(
                f"[yellow]Cohort {cohort} contains less than 5 samples.[/yellow]"
            )
            fWarning = True
    if fWarning:
        console.print(
            "[dim]It is recommended, that each cohort contains at least 5 samples.[/dim]"
        )

    pvals = []

    for ct in bulks.index:
        values = bulks.loc[ct]
        data = []

        for cohort in cohorts:
            data.append(values[samples[cohort_col] == cohort])

        _, p = kruskal(*data)
        pvals.append(p)

    pvals = np.array(pvals)

    pvals_adj = false_discovery_control(pvals)

    result = pd.DataFrame(
        {
            "celltype": bulks.index,
            "p": pvals,
            "p_adj": pvals_adj,
        }
    ).set_index("celltype")

    return result
