"""
=====================================================
Functions for visualization of survival
=====================================================
"""

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from rich.console import Console

from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

console = Console()


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


def plot_kaplan_meier(
    bulks: pd.DataFrame,
    sample_sheet: pd.DataFrame,
    sample_id_col: str,
    time_col: str,
    event_col: str,
    celltype: str,
    out_path: str,
    stratification: str = "median",
) -> None:
    """
    Plot Kaplan-Meier survival curves by cell type composition.

    Parameters
    ----------
    bulks : pd.DataFrame
        Cell type composition (celltype x sample)
    samples_sheet : pd.DataFrame
        sample list (samples x clinical variables)
    sample_id_col : str
        Name of the column that links to bulks
    time_col : str
        Name of the column that links to survival time
    event_col : str
        Name of column that links to event (0: censored, 1: event)
    celltype : str
        Cell type to stratify by
    out_path : str
        Path, where plot will be saved
    stratification : str = "median"
        Method for stratification of patients: "median", "tertiles", "quartiles"
    """

    samples = sample_sheet[[sample_id_col, time_col, event_col]].set_index(
        sample_id_col
    )
    samples = samples.reindex(bulks.columns)

    # Remove entries with missing time or event
    non_none_samples = samples[[time_col, event_col]].notna().all(axis=1)

    samples = samples.loc[non_none_samples]
    bulks = bulks.loc[:, samples.index]

    ct_values = bulks.loc[celltype].values

    if stratification == "median":
        thresh = np.median(ct_values)
        groups = pd.Series(
            ["Low" if x <= thresh else "High" for x in ct_values], index=bulks.columns
        )
    elif stratification == "tertiles":
        thresh = np.percentile(ct_values, [33.33, 66.66])
        groups = pd.Series(
            [
                "Low" if x <= thresh[0] else ("High" if x > thresh[1] else "Medium")
                for x in ct_values
            ],
            index=bulks.columns,
        )
    elif stratification == "quartiles":
        thresh = np.percentile(ct_values, [25, 50, 75])
        groups = pd.Series(
            [
                "Q1"
                if x <= thresh[0]
                else ("Q2" if x < thresh[1] else ("Q3" if x < thresh[2] else "Q4"))
                for x in ct_values
            ],
            index=bulks.columns,
        )
    else:
        console.print("[red]Stratification Type not implemented![/red]")
        raise NotImplementedError(
            f"Stratification {stratification} is not implemented."
        )

    sns.set_style("white")
    fig, ax = plt.subplots(figsize=(8, 6))

    kmf = KaplanMeierFitter()
    group_labels = groups.unique()

    p_value = None
    colors = {
        "Low": "blue",
        "High": "red",
        "Medium": "green",
        "Q1": "blue",
        "Q2": "green",
        "Q3": "yellow",
        "Q4": "red",
    }

    for group in sorted(group_labels):
        mask = groups == group
        T = samples.loc[mask, time_col].values
        E = samples.loc[mask, event_col].values

        kmf.fit(T, E, label=f"{group} (n={mask.sum()})")
        kmf.plot_survival_function(ax=ax, color=colors.get(group, None), linewidth=1)

    # Log-rank test if two groups
    if len(group_labels) == 2:
        mask_group1 = groups == group_labels[0]
        mask_group2 = groups == group_labels[1]

        T1 = samples.loc[mask_group1, time_col].values
        E1 = samples.loc[mask_group1, event_col].values
        T2 = samples.loc[mask_group2, time_col].values
        E2 = samples.loc[mask_group2, event_col].values

        results = logrank_test(T1, T2, E1, E2)
        p_value = results.p_value

        ax.text(
            0.98,
            0.05,
            f"Log-rank p={p_value:.2e}",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=10,
        )

    ax.set_xlabel(f"Time ({time_col})")
    ax.set_ylabel("Probability of Survival")
    ax.set_title(f"Kaplan Meier: {celltype}")
    ax.legend(loc="best", fontsize=10)
    ax.grid(alpha=0.3)

    fig.savefig(out_path, bbox_inches="tight", dpi=300)

    plt.close(fig)


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


def plot_cox_forest(
    cox_results: pd.DataFrame, out_path: str, alpha: float = 0.05
) -> None:
    """
    Plot forest plot.

    Parameters
    ----------
    cox_results : pd.DataFrame
        Cox regression results from *run_cox_regression* function
    out_path : str
        Path where the plot will be saved
    alpha : float = 0.05
        Significance level for different colouring
    """

    cox_results_sorted = cox_results.sort_values("hr").reset_index(drop=True)

    sns.set_style("white")
    fig, ax = plt.subplots(figsize=(10, max(8, len(cox_results_sorted) * 0.4)))

    y_pos = np.arange(len(cox_results_sorted))

    for idx, row in cox_results_sorted.iterrows():
        color = "red" if row["p_value"] < alpha else "gray"

        ax.plot(
            [row["ci_lower"], row["ci_upper"]],
            [idx, idx],
            color=color,
            linewidth=2,
            alpha=0.7,
        )

        ax.scatter(
            row["hr"],
            idx,
            color=color,
            s=100,
            zorder=3,
            edgecolors="black",
            linewidth=0.5,
        )

    ax.axvline(x=1, color="black", linestyle="--", linewidth=1, alpha=0.5)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(cox_results_sorted["celltype"])
    ax.set_xlabel("Hazard Ratio (95% CI)")
    ax.set_title("Cox Regression - Forest Plot")
    ax.grid(axis="x", alpha=0.3)

    ax.set_xscale("log")

    fig.savefig(out_path, bbox_inches="tight", dpi=300)

    plt.close(fig)
