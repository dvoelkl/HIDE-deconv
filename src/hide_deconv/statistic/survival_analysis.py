"""
=====================================================
Functions for survival analysis
=====================================================
"""

import pandas as pd
import numpy as np

from rich.console import Console
from rich.table import Table

from lifelines import CoxPHFitter

import warnings
from lifelines.utils import ConvergenceWarning

from scipy.stats import false_discovery_control

# Suppress ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

console = Console()

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


def run_cox_regression(
    bulks: pd.DataFrame,
    sample_sheet: pd.DataFrame,
    sample_id_col: str,
    time_col: str,
    event_col: str,
    covariates: list[str] | None,
) -> pd.DataFrame:
    """
    Perform Cox Regression for each cell type composition.

    Parameters
    ----------
    bulks : pd.DataFrame
        Cell type compositions (either Celltype x Sample or Gene x Sample)
    sample_sheet : pd.DataFrame
        sample list (samples x clinical variables)
    sample_id_col : str
        Name of the column, that links to the bulks
    time_col : str
        Name of the column with survival times
    event_col : str
        Name of column that indicates event (0: censored, 1: event)
    covariates : list[str], optional
        List of covariate column names

    Returns
    -------
    pd.DataFrame
        DataFrame with estimated impact on survival for each cell type
    """

    samples = sample_sheet.set_index(sample_id_col)
    samples = samples.loc[bulks.columns]

    # Remove entries with missing time or event
    non_none_samples = samples[[time_col, event_col]].notna().all(axis=1)

    # Remove entries with missing covariates, if given
    if covariates:
        non_none_samples = non_none_samples & samples.loc[:, covariates].notna().all(
            axis=1
        )

    # Warning if samples have been removed
    n_removed = (~non_none_samples).sum()
    if n_removed > 0:
        console.print(
            f"[yellow]Removed {n_removed} samples due to missing values.[/yellow]"
        )

    samples = samples.loc[non_none_samples]
    bulks = bulks.loc[:, non_none_samples]

    results = []
    pvals = []

    with console.status(
        "[bold blue]Fitting CoxPH-models...[/bold blue]",
        spinner="dots",
    ):
        for ct in bulks.index:
            try:
                X = pd.DataFrame({"ct_comp": bulks.loc[ct]}, index=bulks.columns)

                X[time_col] = samples.loc[X.index, time_col].values
                X[event_col] = samples.loc[X.index, event_col].values

                if covariates:
                    for cov in covariates:
                        cov_data = samples.loc[X.index, cov].values

                        # Check if covariate is numeric or categorical
                        if pd.api.types.is_numeric_dtype(cov_data):
                            X[cov] = cov_data
                        else:
                            # One-hot encode categorical variables
                            cov_dummies = pd.get_dummies(
                                samples.loc[X.index, cov],
                                prefix=cov,
                                drop_first=True,  # Avoid multicollinearity
                            )
                            X = X.join(cov_dummies)

                X = X.loc[samples.index]

                # Train CoxPH model
                cph = CoxPHFitter()
                cph.fit(X, duration_col=time_col, event_col=event_col)

                coef = cph.params_.loc["ct_comp"]
                hr = np.exp(coef)
                ci = cph.confidence_intervals_.loc["ct_comp"]
                p_val = cph.summary.loc["ct_comp", "p"]

                concordance = cph.concordance_index_

                results.append(
                    {
                        "celltype": ct,
                        "coef": coef,
                        "hr": hr,
                        "ci_lower": np.exp(ci.iloc[0]),
                        "ci_upper": np.exp(ci.iloc[1]),
                        "p_value": p_val,
                        "concordance_index": concordance,
                    }
                )
                pvals.append(p_val)

            except Exception:
                console.print(f"[yellow]Failed to fit CoxPH model for {ct}.[/yellow]")
                results.append(
                    {
                        "celltype": ct,
                        "coef": np.nan,
                        "hr": np.nan,
                        "ci_lower": np.nan,
                        "ci_upper": np.nan,
                        "p_value": np.nan,
                        "concordance_index": np.nan,
                    }
                )
                pvals.append(np.nan)

        result_df = pd.DataFrame(results)
        p_vals = np.array(pvals)
        p_vals_valid = ~np.isnan(p_vals)

        pvals_adj = np.full_like(p_vals, np.nan, dtype=float)
        pvals_adj[p_vals_valid] = false_discovery_control(p_vals[p_vals_valid])

        result_df["p_value_adj"] = pvals_adj

        return result_df


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


def print_cox_summary(cox_results: pd.DataFrame) -> None:
    """
    Print a summary table of significant Cox regression results.

    Parameters
    ----------
    cox_results : pd.DataFrame
        Results from *run_cox_regression* function
    """

    significant = cox_results[cox_results["p_value_adj"] < 0.05].copy()

    if len(significant) > 0:
        console.print("[green]Significant Cell Types (p_adj < 0.05):[/green]")

        table = Table(show_header=True, header_style="bold")
        table.add_column("Cell Type", style="cyan")
        table.add_column("Hazard Ratio", justify="right")
        table.add_column("95% CI", justify="right")
        table.add_column("p-value", justify="right")
        table.add_column("p-adj", justify="right")

        for _, row in significant.iterrows():
            table.add_row(
                row["celltype"],
                f"{row['hr']:.3e}",
                f"[{row['ci_lower']:.2e}-{row['ci_upper']:.2e}]",
                f"{row['p_value']:.2e}",
                f"{row['p_value_adj']:.2e}",
            )

        console.print(table)
    else:
        console.print(
            "[yellow]No significant cell types found (p_adj < 0.05).[/yellow]"
        )
