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
    palette = dict(zip(labels, sns.color_palette("hls", len(labels))))

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
    ax.set_aspect("equal", adjustable="datalim")

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
    loading, out_path: Path, title: str = "PLS-DA Loading Plot", top_n: int = 20
) -> None:
    """
    Save a PLS-DA loading plot.
    """

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.set_theme(style="whitegrid", context="paper")

    df_load = pd.DataFrame(loading).copy()
    df_load = df_load.iloc[:, :2]
    df_load.columns = ["PLS1", "PLS2"]

    rank = df_load.abs().max(axis=1).sort_values(ascending=False)
    top_idx = rank.head(min(top_n, len(rank))).index

    df_plot = df_load.loc[top_idx].copy()
    df_plot["Feature"] = df_plot.index
    comp_cols = ["PLS1", "PLS2"]
    df_melt = df_plot.reset_index(drop=True).melt(
        id_vars=["Feature"],
        value_vars=comp_cols,
        var_name="Component",
        value_name="Loading",
    )

    sns.barplot(
        data=df_melt,
        x="Loading",
        y="Feature",
        hue="Component",
        ax=ax,
        palette=["tab:orange", "tab:green"],
    )

    ax.set_title(title)
    ax.set_xlabel("Loading")
    ax.set_ylabel("Feature")
    ax.legend(title="Component", bbox_to_anchor=(1.02, 1), loc="upper left")

    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_plsda_biplot(
    scores: pd.DataFrame,
    loadings: pd.DataFrame,
    out_path: Path,
    cohort_col: str,
    top_n: int = 15,
) -> None:
    """
    Save a PLS-DA biplot.
    """

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.set_theme(style="whitegrid", context="paper")

    labels = scores[cohort_col].dropna().unique()
    palette = dict(zip(labels, sns.color_palette("hls", len(labels))))

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

    # select top n features
    loadings = loadings.copy()
    loadings["norm"] = (loadings["PLS1"] ** 2 + loadings["PLS2"] ** 2) ** 0.5
    top = loadings.sort_values("norm", ascending=False).head(min(top_n, len(loadings)))

    x_lim = ax.get_xlim()
    y_lim = ax.get_ylim()
    score_span = max(abs(x_lim[1] - x_lim[0]), abs(y_lim[1] - y_lim[0]))

    if top["norm"].max() == 0:
        scale = 1.0
    else:
        scale = 0.8 * score_span / top["norm"].max()

    for idx, row in top.iterrows():
        x = row["PLS1"] * scale
        y = row["PLS2"] * scale
        ax.arrow(
            0,
            0,
            x,
            y,
            head_width=0.02 * score_span,
            head_length=0.03 * score_span,
            linewidth=1.0,
            color="tab:red",
            length_includes_head=True,
            alpha=0.8,
        )
        ax.text(x * 1.05, y * 1.05, str(idx), fontsize=8, color="black")

    ax.set_title("PLS-DA Biplot")
    ax.set_xlabel("PLS1")
    ax.set_ylabel("PLS2")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(title=cohort_col, bbox_to_anchor=(1.02, 1), loc="upper left")

    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
