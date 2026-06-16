"""
=====================================================
Functions for visualization with heatmaps
=====================================================
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from matplotlib.patches import Rectangle
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

from scipy.cluster.hierarchy import linkage


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


def order_subtypes(mwu_sub: pd.DataFrame, A_matrix: list[pd.DataFrame]) -> list[str]:
    """
    Order subtypes according to hierarchy.
    """

    if len(A_matrix) <= 1:
        return list(mwu_sub.index)

    base_order = {ct: idx for idx, ct in enumerate(mwu_sub.index)}
    order_info = []

    for ct in mwu_sub.index:
        sort_key = []

        for proj in A_matrix[1:]:
            parent_rows = [
                row
                for row in proj.index
                if ct in proj.columns and proj.loc[row, ct] == 1
            ]

            if len(parent_rows) == 0:
                sort_key.append(len(proj.index))
            else:
                sort_key.append(proj.index.get_loc(parent_rows[0]))

        order_info.append((tuple(sort_key), base_order[ct], ct))

    order_info.sort()

    return [ct for _, _, ct in order_info]


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


def plot_hier_heat(
    mwu_sub: pd.DataFrame,
    mwu_higher: list[pd.DataFrame],
    layer_names: list[str],
    A_matrix: list[pd.DataFrame],
    cohort_1_name: str,
    cohort_2_name: str,
    out_path: Path,
) -> None:
    """
    Plots Man Whitney U results as hierarchical heatmap connected by the relationship between the celltypes.

    Parameters
    ----------
    mwu_sub : pd.DataFrame
        mwu results of finest layer
    mwu_higher : list[pd.DataFrame]
        mwu results of higher layers
    layer_names : list[str]
        Layer names, as they should appear in the plot
    A_matrix : list[pd.DataFrame]
        Projection matrices, note, that the first entry should be the identity
    cohort_1_name : str
        Name of the first cohort as it should appear in the plot
    cohort_2_name : str
        Name of the second cohort as it should appear in the plot
    out_path : Path
        Path, where the plot will be saved
    """

    sns.set_theme(style="whitegrid", context="paper")

    level_dfs = list(reversed(mwu_higher)) + [mwu_sub]
    level_names = list(reversed(layer_names[1:])) + [layer_names[0]]

    proj_dict = {name: mat for name, mat in zip(layer_names, A_matrix)}

    subtypes = order_subtypes(mwu_sub, A_matrix)
    mwu_sub = mwu_sub.reindex(subtypes)

    n_sub = len(subtypes)

    row_spacing = 1.15
    sub_y = {ct: (n_sub - 1 - i) * row_spacing for i, ct in enumerate(subtypes)}

    # determine positions of cell types
    y_positions = {}
    y_positions[layer_names[0]] = sub_y

    for level_name, proj in zip(layer_names[1:], A_matrix[1:]):
        level_pos = {}

        for parent in proj.index:
            members = proj.loc[parent]
            members = members[members == 1].index

            ys = [sub_y[m] for m in members]

            level_pos[parent] = np.mean(ys)

        y_positions[level_name] = level_pos

    ordered_level_dfs = []

    for level_name, df in zip(level_names, level_dfs):
        ordered_index = sorted(
            df.index, key=lambda ct: (y_positions[level_name][ct], df.index.get_loc(ct))
        )

        ordered_level_dfs.append(df.loc[ordered_index])

    # Collect mean expressions and normalize them
    all_means = []

    for df in ordered_level_dfs:
        all_means.extend(df.iloc[:, 0].values)
        all_means.extend(df.iloc[:, 2].values)

    all_means = np.asarray(all_means)

    log_vals = np.log10(all_means + 1e-8)

    vmin = np.quantile(log_vals, 0.02)
    vmax = np.quantile(log_vals, 0.98)

    norm = Normalize(vmin=vmin, vmax=vmax)

    cmap = sns.color_palette("viridis", as_cmap=True)

    n_levels = len(level_dfs)

    heatmap_width = 2.0
    layer_spacing = 2.5

    x_positions = {
        level_name: i * layer_spacing for i, level_name in enumerate(level_names)
    }

    fig_width = n_levels * 2.5 + 4
    fig_height = max(6, n_sub * 0.6)
    top_label_y = (n_sub - 1) * row_spacing + row_spacing * 2.35
    bottom_label_y = -row_spacing * 1.25

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    # Create connections between the various layers
    for i in range(n_levels - 1):
        parent_name = level_names[i]
        child_name = level_names[i + 1]

        parent_df = level_dfs[i]
        child_df = level_dfs[i + 1]

        parent_proj = proj_dict[parent_name]
        child_proj = proj_dict[child_name]

        x_parent = x_positions[parent_name] + heatmap_width
        x_child = x_positions[child_name]

        for parent in parent_df.index:
            parent_subs = set(
                parent_proj.loc[parent][parent_proj.loc[parent] == 1].index
            )

            y_parent = y_positions[parent_name][parent]

            for child in child_df.index:
                child_subs = set(
                    child_proj.loc[child][child_proj.loc[child] == 1].index
                )

                if len(parent_subs & child_subs) == 0:
                    continue

                y_child = y_positions[child_name][child]

                ax.plot(
                    [x_parent, x_child],
                    [y_parent, y_child],
                    color="lightgrey",
                    lw=1,
                    zorder=1,
                )

    # Draw heatmaps
    cell_height = min(0.9, row_spacing * 0.75)
    cell_width = 1.0

    for level_name, df in zip(level_names, ordered_level_dfs):
        x0 = x_positions[level_name]

        for celltype, row in df.iterrows():
            y = y_positions[level_name][celltype]

            mean_1 = row.iloc[0]
            mean_2 = row.iloc[2]

            val_1 = np.log10(mean_1 + 1e-8)
            val_2 = np.log10(mean_2 + 1e-8)

            color_1 = cmap(norm(val_1))
            color_2 = cmap(norm(val_2))

            rect1 = Rectangle(
                (x0, y - cell_height / 2),
                cell_width,
                cell_height,
                facecolor=color_1,
                edgecolor="white",
                linewidth=0.5,
                zorder=2,
            )

            rect2 = Rectangle(
                (x0 + cell_width, y - cell_height / 2),
                cell_width,
                cell_height,
                facecolor=color_2,
                edgecolor="white",
                linewidth=0.5,
                zorder=2,
            )

            ax.add_patch(rect1)
            ax.add_patch(rect2)

            marker = ""

            if row["p_adj"] < 0.05:
                marker = "**"

            elif row["p"] < 0.05:
                marker = "*"

            if marker:
                ax.text(
                    x0 + 0.5,
                    y,
                    marker,
                    ha="center",
                    va="center",
                    fontsize=8,
                    fontweight="bold",
                    c="r",
                    zorder=3,
                )

                ax.text(
                    x0 + 1.5,
                    y,
                    marker,
                    ha="center",
                    va="center",
                    fontsize=8,
                    fontweight="bold",
                    c="r",
                    zorder=3,
                )

    for level_name in level_names:
        x0 = x_positions[level_name]

        ax.text(
            x0 + 1.0,
            top_label_y,
            level_name,
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
        )

    for level_name in level_names:
        x0 = x_positions[level_name]

        ax.text(
            x0 + 0.5, bottom_label_y, cohort_1_name, ha="center", va="top", rotation=45
        )

        ax.text(
            x0 + 1.5, bottom_label_y, cohort_2_name, ha="center", va="top", rotation=45
        )

    left_level = level_names[0]

    for ct in ordered_level_dfs[0].index:
        y = y_positions[left_level][ct]
        ax.text(
            x_positions[left_level] - 0.2, y, ct, ha="right", va="center", fontsize=9
        )

    right_level = layer_names[0]

    for ct in ordered_level_dfs[-1].index:
        y = y_positions[right_level][ct]
        ax.text(
            x_positions[right_level] + 2.2, y, ct, ha="left", va="center", fontsize=9
        )

    sm = ScalarMappable(norm=norm, cmap=cmap)

    cbar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)

    cbar.set_label("log10(mean proportion)")

    ax.set_xlim(-1.5, max(x_positions.values()) + 4)

    ax.set_ylim(bottom_label_y - row_spacing * 1.3, top_label_y + row_spacing * 1.1)

    ax.set_xticks([])
    ax.set_yticks([])

    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.grid(False)

    fig.text(
        0.5, 0.02, "* nominal p < 0.05 ** adjusted p < 0.05", ha="center", fontsize=10
    )

    plt.tight_layout(rect=(0, 0.05, 1, 1))

    out_path = Path(out_path)

    plt.savefig(
        out_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


def plot_genemap(X: pd.DataFrame, gene_series, title: str, out_path: Path) -> None:
    """
    Plot subset of genes of reference profile as heatmap and cluster them.

    Parameters
    ----------
    X : pd.DataFrame
        Reference profile
    gene_series : list
        List of genes to plot
    title : str
        Title of plot
    out_path : str
        Path, where plot will be saved
    """

    # Normalize genes, such that sum over each gene = 1
    X_standardized = (X.T / X.T.sum(axis=0)).T.loc[gene_series]
    n_celltypes = X_standardized.shape[1]
    n_genes = X_standardized.shape[0]

    row_linkage = linkage(X_standardized.T, method="single")
    col_linkage = linkage(X_standardized, method="single")

    clustermap = sns.clustermap(
        X_standardized.T,
        row_cluster=True,
        col_cluster=True,
        figsize=(max(25, n_genes * 0.45), max(6, n_celltypes * 0.35)),
        row_linkage=row_linkage,
        col_linkage=col_linkage,
        # cmap="coolwarm",
        # linewidths=0.8,
        cbar_kws={"label": "Expression Level"},
        xticklabels=True,
        yticklabels=True,
        # cbar_pos=(0.05, 0.8, 0.03, 0.15),
    )
    clustermap.ax_heatmap.tick_params(axis="x", labelrotation=90, labelsize=8)
    clustermap.ax_heatmap.tick_params(axis="y", labelsize=8)
    plt.title(f"{title}")

    clustermap.savefig(out_path)
