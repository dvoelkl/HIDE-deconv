"""
=====================================================
Functions for Mann Whitney U Test
=====================================================
"""

import pandas as pd
import numpy as np
from scipy.stats import mannwhitneyu, false_discovery_control

from rich.console import Console
from rich.table import Table

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

console = Console()

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


def run_mann_whitney_u(
    bulks: pd.DataFrame, sample_list: pd.DataFrame, sample_id_col: str, cohort_col: str
) -> pd.DataFrame:
    """
    Performs a Man Whitney U Test with FDR correction for two cohorts.

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
        DataFrame containing the Mean, Standard Deviation, p-Value and adjusted p-Value
    """

    samples = sample_list[[sample_id_col, cohort_col]].set_index(sample_id_col)

    # Subset samples, as sample list might have more samples, than in deconvolution
    samples = samples.loc[bulks.columns]

    cohorts = samples[cohort_col].unique()

    if len(cohorts) > 2 or len(cohorts) < 2:
        raise Exception(
            "MWU can only be performed between two cohorts. Consider using a Kruskal Wallis Test."
        )

    group1, group2 = cohorts[0], cohorts[1]

    pvals = []
    mean1, std1, mean2, std2 = [], [], [], []

    for ct in bulks.index:
        values = bulks.loc[ct]
        x = values[samples[cohort_col] == group1]
        y = values[samples[cohort_col] == group2]

        mean1.append(x.mean())
        mean2.append(y.mean())

        std1.append(x.std())
        std2.append(y.std())

        _, p = mannwhitneyu(x, y)
        pvals.append(p)

    pvals = np.array(pvals)

    pvals_adj = false_discovery_control(pvals)

    result = pd.DataFrame(
        {
            "celltype": bulks.index,
            f"mean[{group1}]": mean1,
            f"std[{group1}]": std1,
            f"mean[{group2}]": mean2,
            f"std[{group2}]": std2,
            "p": pvals,
            "p_adj": pvals_adj,
        }
    ).set_index("celltype")

    return result


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


def print_mwu_summary(results: pd.DataFrame, sign_level: float = 0.05) -> None:
    """
    Print signficiant results of Mann Whitney U Test in command line.

    Parameters:
    -----------
    results : pd.DataFrame
        Results from the run_mann_whitney_u() method
    sign_level:
        Float, below which the results are considered statistical significant.
    """

    significant = results[results["p_adj"] < sign_level]
    n_sign = len(significant)

    if n_sign > 0:
        table = Table(
            show_header=True,
            header_style="bold magenta",
            title=f"[bold green]Significant (p_adj < {sign_level}) cell types[/bold green]",
        )
        table.add_column("Cell Type")

        for col in results.columns:
            table.add_column(col)

        significant = significant.sort_values("p_adj")

        for idx, row in significant.iterrows():
            table.add_row(
                str(idx),
                *[
                    f"{row[col]:.4g}"
                    if isinstance(row[col], (int, float))
                    else str(row[col])
                    for col in results.columns
                ],
            )

        console.print(table)
    else:
        console.print("[bold yellow]No significant cell types found.[/bold yellow]")


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
