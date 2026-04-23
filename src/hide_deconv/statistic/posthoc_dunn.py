"""
=====================================================
Functions for posthoc dunn test
=====================================================
"""

import pandas as pd
import numpy as np
import scikit_posthocs as sp

from rich.console import Console
from rich.table import Table

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

console = Console()

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


def run_dunn(
    kruskal_results: pd.DataFrame,
    bulks: pd.DataFrame,
    sample_list: pd.DataFrame,
    sample_id_col: str,
    cohort_col: str,
    sign_level: float = 0.05,
) -> pd.DataFrame:
    """
    Performs a Posthoc Dunn Test on significant Kruskal Wallis results.

    Parameters
    ----------
    kruskal_results : pd.DataFrame
        Results of obtained by the run_kruskal_wallis() method
    bulks : pd.DataFrame
        bulk Samples (either Celltype x Sample or Gene x Sample)
    sample_list : pd.DataFrame
        sample list (samples x clinical variables)
    sample_id_col : str
        Name of the column, that links to the bulks
    cohort_col: str
        Name of the column containing the cohort identifiers.
    sign_level : float = 0.05
        Float, below which results are considered as significant.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the results of the Dunn Test.
    """

    samples = sample_list[[sample_id_col, cohort_col]].set_index(sample_id_col)
    samples = samples.loc[bulks.columns]

    # get significant results
    sig_celltypes = kruskal_results.index[kruskal_results["p_adj"] < sign_level]

    rows = []
    for ct in sig_celltypes:
        values = bulks.loc[ct]

        df = pd.DataFrame(
            {
                "value": values.values,
                "cohort": samples.loc[values.index, cohort_col].values,
            },
            index=values.index,
        ).dropna()

        dunn = sp.posthoc_dunn(
            df, val_col="value", group_col="cohort", p_adjust="fdr_bh"
        )

        dunn.index.name = "cohort_1"
        dunn.columns.name = "cohort_2"

        long_df = (
            dunn.where(np.triu(np.ones(dunn.shape, dtype=bool), k=1))
            .stack()
            .reset_index(name="p_adj")
        )

        long_df.insert(0, "celltype", ct)
        rows.append(long_df)

    return pd.concat(rows, ignore_index=True)


def print_dunn_summary(results: pd.DataFrame, sign_level: float = 0.05) -> None:
    """
    Print signficiant results of Dunn Test in command line.

    Parameters:
    -----------
    results : pd.DataFrame
        Results from the run_dunn() method
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

        for col in results.columns:
            table.add_column(col)

        significant = significant.sort_values("p_adj")

        for idx, row in significant.iterrows():
            table.add_row(
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
