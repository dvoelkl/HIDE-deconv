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
    samples = samples.reindex(bulks.columns)
    samples = samples.dropna(subset=[cohort_col])
    cohorts = samples[cohort_col].unique()
    bulks = bulks.loc[:, samples.index]

    fWarning = False
    for cohort in cohorts:
        if (samples[cohort_col] == cohort).sum() <= 1:
            console.print(f"[red]Cohort {cohort} contains only 1 sample.[/red]")
            fWarning = True
        elif (samples[cohort_col] == cohort).sum() < 5:
            console.print(
                f"[yellow]Cohort {cohort} contains less than 5 samples.[/yellow]"
            )
            fWarning = True
    if fWarning:
        console.print(
            "[dim]It is recommended, that each cohort contains at least 5 samples.[/dim]"
        )

    pvals = []
    valid_celltypes = []

    for ct in bulks.index:
        values = bulks.loc[ct]
        data = []

        for cohort in cohorts:
            cohort_values = values[samples[cohort_col] == cohort].dropna()
            data.append(cohort_values)

        if any(len(group) == 0 for group in data):
            console.print(
                f"[yellow]Celltype {ct} cannot be tested because at least one cohort is empty.[/yellow]"
            )
            continue

        all_values = pd.concat(data, ignore_index=True)
        if all_values.nunique(dropna=True) < 2:
            console.print(
                f"[yellow]Celltype {ct} cannot be tested because all cohort values are identical.[/yellow]"
            )
            continue

        try:
            _, p = kruskal(*data)
        except ValueError as exc:
            console.print(
                f"[yellow]Celltype {ct} could not be tested by Kruskal-Wallis and was skipped: {exc}[/yellow]"
            )
            continue

        if not np.isfinite(p):
            console.print(
                f"[yellow]Celltype {ct} produced an invalid p-value and was skipped.[/yellow]"
            )
            continue

        valid_celltypes.append(ct)
        pvals.append(p)

    if len(pvals) == 0:
        console.print(
            "[yellow]No testable celltypes found for Kruskal-Wallis analysis.[/yellow]"
        )
        return pd.DataFrame(columns=["p", "p_adj"]).rename_axis("celltype")

    pvals = np.array(pvals, dtype=float)
    pvals_adj = false_discovery_control(pvals)

    result = pd.DataFrame(
        {
            "celltype": valid_celltypes,
            "p": pvals,
            "p_adj": pvals_adj,
        }
    ).set_index("celltype")

    return result
