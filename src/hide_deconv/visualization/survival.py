"""
=====================================================
Functions for visualization of survival
=====================================================
"""

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from typing import List

from rich.console import Console

from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
from lifelines.plotting import add_at_risk_counts

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

console = Console()


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


def plot_kaplan_meier_comp(
    bulks: pd.DataFrame,
    sample_sheet: pd.DataFrame,
    sample_id_col: str,
    time_col: str,
    event_col: str,
    celltype: str,
    out_path: str,
    stratification: str = "median",
    show_censors: bool = False,
    show_risk_table: bool = False,
    show_median_lines: bool = False,
) -> None:
    """
    Plot Kaplan-Meier survival curves by cell type composition.

    Parameters
    ----------
    bulks : pd.DataFrame
        Cell type composition (celltype x sample)
    sample_sheet : pd.DataFrame
        Sample list (samples x clinical variables)
    sample_id_col : str
        Name of the column that links to bulks.
    time_col : str
        Name of the survival time column.
    event_col : str
        Name of the event column (0=censored, 1=event).
    celltype : str
        Cell type to stratify by.
    out_path : str
        Output filename.
    stratification : str, default="median"
        One of {"median", "tertiles", "quartiles"}.
    show_censors : bool, default=False
        Draw censoring marks on the KM curves.
    show_risk_table : bool, default=False
        Add an at-risk table below the plot.
    show_median_lines : bool, default=False
        Draw horizontal line at survival=0.5 and vertical median survival lines.
    """

    samples = sample_sheet[[sample_id_col, time_col, event_col]].set_index(
        sample_id_col
    )
    samples = samples.reindex(bulks.columns)

    # Remove samples with missing survival information
    samples = samples.dropna(subset=[time_col, event_col])
    bulks = bulks.loc[:, samples.index]

    ct_values = bulks.loc[celltype].values

    if stratification == "median":
        thresh = np.median(ct_values)
        groups = pd.Series(
            np.where(ct_values <= thresh, "Low", "High"),
            index=bulks.columns,
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
            f"Stratification '{stratification}' is not implemented."
        )

    sns.set_style("white")

    if show_risk_table:
        fig, ax = plt.subplots(figsize=(8, 8))
    else:
        fig, ax = plt.subplots(figsize=(8, 6))

    colors = {
        "Low": "blue",
        "Medium": "green",
        "High": "red",
        "Q1": "blue",
        "Q2": "green",
        "Q3": "orange",
        "Q4": "red",
    }

    kmfs = []
    group_labels = sorted(groups.unique())

    # Kaplan-Meier curves
    for group in group_labels:
        mask = groups == group

        T = samples.loc[mask, time_col].values
        E = samples.loc[mask, event_col].values

        kmf = KaplanMeierFitter()
        kmf.fit(
            T,
            E,
            label=f"{group} (n={mask.sum()})",
        )

        kmf.plot_survival_function(
            ax=ax,
            color=colors.get(group),
            linewidth=1.5,
            ci_show=False,
            show_censors=show_censors,
            censor_styles={
                "marker": "|",
                "ms": 8,
                "mew": 1.2,
            },
        )

        kmfs.append(kmf)

    # Log-rank test (only for two groups)
    if len(group_labels) == 2:
        mask1 = groups == group_labels[0]
        mask2 = groups == group_labels[1]

        results = logrank_test(
            samples.loc[mask1, time_col],
            samples.loc[mask2, time_col],
            event_observed_A=samples.loc[mask1, event_col],
            event_observed_B=samples.loc[mask2, event_col],
        )

        ax.text(
            0.98,
            0.05,
            f"Log-rank p = {results.p_value:.2e}",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=10,
        )

    if show_median_lines:
        ax.axhline(
            0.5,
            color="gray",
            linestyle="--",
            linewidth=1,
        )

        for kmf in kmfs:
            median = kmf.median_survival_time_
            if np.isfinite(median):
                ax.vlines(
                    median,
                    ymin=0,
                    ymax=0.5,
                    color="gray",
                    linestyle="--",
                    linewidth=1,
                    alpha=0.7,
                )

    # Risk table
    if show_risk_table:
        add_at_risk_counts(*kmfs, ax=ax)

    ax.set_xlabel(f"Time ({time_col})")
    ax.set_ylabel("Probability of Survival")
    ax.set_title(f"Kaplan-Meier: {celltype}")
    ax.legend(loc="best", fontsize=10)
    ax.grid(alpha=0.3)

    plt.tight_layout()

    fig.savefig(
        out_path,
        bbox_inches="tight",
        dpi=300,
    )

    plt.close(fig)


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


def plot_kaplan_meier_cohort(
    sample_sheet: pd.DataFrame,
    cohort_strat_col: str,
    time_col: str,
    event_col: str,
    out_path: str,
    max_time: float = -1.0,
    show_censors: bool = False,
    show_risk_table: bool = False,
    show_median_lines: bool = False,
) -> None:
    """
    Plot Kaplan-Meier survival curves stratified by cohorts.

    Parameters
    ----------
    samples_sheet : pd.DataFrame
        sample list (samples x clinical variables)
    time_col : str
        Name of the column that links to survival time
    event_col : str
        Name of column that links to event (0: censored, 1: event)
    out_path : str
        Path, where plot will be saved
    max_time : float = -1.0
        Maximum time interval to investigate, ignored set to -1.0
    show_censors : bool, default=False
        Draw censoring marks on the KM curves.
    show_risk_table : bool, default=False
        Add an "at risk" table below the plot.
    show_median_lines : bool, default=False
        Draw a horizontal line at survival=0.5 and vertical lines at the
        median survival time of each cohort.
    """

    samples = sample_sheet[[cohort_strat_col, time_col, event_col]].copy()

    # Remove entries with missing values
    samples = samples.dropna(subset=[cohort_strat_col, time_col, event_col])

    if max_time > 0:
        original_time = samples[time_col].copy()

        samples[time_col] = samples[time_col].clip(upper=max_time)

        samples[event_col] = np.where(
            original_time > max_time,
            0,
            samples[event_col],
        )

    groups = samples[cohort_strat_col]
    group_labels = sorted(samples[cohort_strat_col].unique())

    sns.set_style("white")

    if show_risk_table:
        fig, ax = plt.subplots(figsize=(8, 8))
    else:
        fig, ax = plt.subplots(figsize=(8, 6))

    kmfs: List[KaplanMeierFitter] = []

    # Plot KM curves
    for group in group_labels:
        mask = groups == group

        T = samples.loc[mask, time_col].values
        E = samples.loc[mask, event_col].values

        kmf = KaplanMeierFitter()
        kmf.fit(
            durations=T,
            event_observed=E,
            label=f"{group} (n={mask.sum()})",
        )

        kmf.plot_survival_function(
            ax=ax,
            linewidth=1.5,
            ci_show=False,
            show_censors=show_censors,
            censor_styles={
                "marker": "|",
                "ms": 8,
                "mew": 1.2,
            },
        )

        kmfs.append(kmf)

    # Log-rank test for two groups
    if len(group_labels) == 2:
        mask1 = groups == group_labels[0]
        mask2 = groups == group_labels[1]

        results = logrank_test(
            samples.loc[mask1, time_col],
            samples.loc[mask2, time_col],
            event_observed_A=samples.loc[mask1, event_col],
            event_observed_B=samples.loc[mask2, event_col],
        )

        ax.text(
            0.98,
            0.05,
            f"Log-rank p = {results.p_value:.2e}",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=10,
        )

    if show_median_lines:
        ax.axhline(
            0.5,
            color="gray",
            linestyle="--",
            linewidth=1,
        )

        for kmf in kmfs:
            median = kmf.median_survival_time_

            if np.isfinite(median):
                ax.vlines(
                    median,
                    ymin=0,
                    ymax=0.5,
                    color="gray",
                    linestyle="--",
                    linewidth=1,
                    alpha=0.7,
                )

    # Risk table
    if show_risk_table:
        add_at_risk_counts(*kmfs, ax=ax)

    if max_time > 0:
        ax.set_xlim(0, max_time)

    ax.set_xlabel(f"Time ({time_col})")
    ax.set_ylabel("Probability of Survival")
    ax.set_title(f"Kaplan-Meier: {cohort_strat_col}")

    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize=10)

    plt.tight_layout()

    fig.savefig(
        out_path,
        dpi=300,
        bbox_inches="tight",
    )

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
