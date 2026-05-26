"""
=====================================================
Functions for PLS-DA plotting
=====================================================
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def plot_plsda_score(
    scores: pd.DataFrame,
    out_path: Path,
    cohort_col: str,
) -> None:
    """
    Save a PLS-DA score plot.
    """

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.set_theme(style="whitegrid", context="paper")

    labels = scores[cohort_col].dropna().unique()
    palette = dict(zip(labels, sns.color_palette("tab10", len(labels))))

    sns.scatterplot(
        data=scores,
        x="PLS1",
        y="PLS2",
        hue=cohort_col,
        ax=ax,
        palette=palette,
        s=65,
    )

    ax.axhline(0, color="0.85", linewidth=1, zorder=0)
    ax.axvline(0, color="0.85", linewidth=1, zorder=0)
    ax.set_title("PLS-DA Score Plot")
    ax.set_xlabel("PLS1")
    ax.set_ylabel("PLS2")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(title=cohort_col, bbox_to_anchor=(1.02, 1), loc="upper left")

    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_plsda_vip(
    vip: pd.Series, out_path: Path, title: str = "PLS-DA VIP Plot"
) -> None:
    """
    Save a PLS-DA VIP plot.
    """

    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.set_theme(style="whitegrid", context="paper")

    vip = vip.sort_values(key=lambda s: np.abs(s), ascending=False)
    top_vip = vip.head(min(20, len(vip)))

    sns.barplot(x=top_vip.values, y=top_vip.index, ax=ax, color="steelblue")
    ax.axvline(1.0, color="tab:red", linestyle="--", linewidth=1)
    ax.set_title(title)
    ax.set_xlabel("VIP")
    ax.set_ylabel("Feature")

    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_plsda_loading(
    loading: pd.Series,
    out_path: Path,
    title: str = "PLS-DA Loading Plot",
) -> None:
    """
    Save a PLS-DA loading plot.
    """

    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.set_theme(style="whitegrid", context="paper")

    loading = loading.sort_values(key=lambda s: np.abs(s), ascending=False)
    top_loading = loading.head(min(20, len(loading)))

    sns.barplot(x=top_loading.values, y=top_loading.index, ax=ax, color="tab:orange")
    ax.set_title(title)
    ax.set_xlabel("Loading")
    ax.set_ylabel("Feature")

    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
