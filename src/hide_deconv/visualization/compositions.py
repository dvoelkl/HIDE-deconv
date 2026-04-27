"""
=====================================================
Functions for visualization of compositions
=====================================================
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import umap

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from scipy.stats import spearmanr, pearsonr, kendalltau
from numpy.linalg import norm

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


def plot_eval(C_true: pd.DataFrame, C_hat: pd.DataFrame, out_path: str) -> pd.DataFrame:
    """
    Plots a scatter-box-plot of two compositions and saves it at the given directory.

    Additionally calculates various metrics between the two compositions and returns them

    Parameters
    ----------
    C_true : pd.DataFrame
        Ground truth composition (celltype x mixture)
    C_hat : pd.DataFrame
        Estimated composition (celltype x mixture)
    out_path : str
        Path, where the scatter-box-plot will be saved.

    Returns
    -------
    pd.DataFrame
        Dataframe containing various metrics for comparing the two compositions.
    """
    C_true = C_true.loc[C_hat.index, C_hat.columns]

    results = []

    celltypes = list(C_true.index)
    n = len(celltypes)

    ncols = 3
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), sharey=False)
    axes = np.array(axes).reshape(-1)

    for idx, ct in enumerate(celltypes):
        ax = axes[idx]

        c_true = C_true.loc[ct].values
        c_hat = C_hat.loc[ct].values

        c_true = C_true.loc[ct].values
        c_hat = C_hat.loc[ct].values

        min_val = np.min(c_true)
        max_val = np.max(c_true)

        bins = np.linspace(min_val, max_val, 11)

        bin_ids = np.digitize(c_true, bins) - 1
        bin_ids = np.clip(bin_ids, 0, len(bins) - 2)

        df_plot = pd.DataFrame({"true": c_true, "hat": c_hat, "bin": bin_ids})

        sns.stripplot(
            data=df_plot,
            x="bin",
            y="hat",
            ax=ax,
            color="black",
            size=2,
            alpha=0.4,
            jitter=0.2,
        )
        sns.boxplot(
            data=df_plot,
            x="bin",
            y="hat",
            ax=ax,
            color="white",
            linecolor="lightblue",
            fliersize=2,
        )

        ax.set_title(ct)
        ax.set_xlabel("True")
        ax.set_ylabel("Estimated")

        ax.set_xticks(range(len(bins) - 1))
        ax.set_xticklabels(
            [f"{bins[j]:.3f}" for j in range(len(bins) - 1)], rotation=45
        )

        # Metrics
        pcc = pearsonr(c_true, c_hat)[0]
        scc = spearmanr(c_true, c_hat)[0]
        kt = kendalltau(c_true, c_hat)[0]
        rmse = np.sqrt(np.mean((c_true - c_hat) ** 2))
        nmae = np.mean(np.abs(c_true - c_hat)) / (np.mean(c_true) + 1e-8)
        cos_sim = np.dot(c_true, c_hat) / (norm(c_true) * norm(c_hat) + 1e-8)

        annot = f"PCC: {pcc:.2f}\nSCC: {scc:.2f}\nNMAE: {nmae:.2f}"

        ax.text(
            0.98, 0.98, annot, transform=ax.transAxes, ha="right", va="top", fontsize=7
        )

        results.append(
            {
                "celltype": ct,
                "PCC": pcc,
                "SCC": scc,
                "KT": kt,
                "RMSE": rmse,
                "NMAE": nmae,
                "COS_SIM": cos_sim,
            }
        )

    for j in range(len(celltypes), len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()

    if out_path is not None:
        fig.savefig(out_path, dpi=300)

    plt.close(fig)

    return pd.DataFrame(results).set_index("celltype")


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


def plot_pca(
    C_est: pd.DataFrame,
    out_path: str,
    labeling: list = [],
    group_name: str = "Cohorts",
    title_suffix: str = "",
) -> None:
    """
    Performs a principal component analysis on the given composition and plots the result as a scatterplot.
    the labeling list can be used to add a coloring to the points.

    Parameters
    ----------
    C_est : pd.DataFrame
        Composition dataframe (celltypes x bulks)
    out_path : str
        Filename + Path, where the plot will be stored.
    labeling : list = []
        List with labels for each bulk.
    group_name : str = "Cohorts"
        Name of the legend.
    title_suffix : str = ''
        Suffix displayed after the image title.
    """

    df = C_est.T

    df_scaled = StandardScaler().fit_transform(df)

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(df_scaled)

    pca_df = pd.DataFrame(X_pca, index=df.index, columns=["PC1", "PC2"])

    fig, ax = plt.subplots(figsize=(7, 5))

    if len(labeling) > 0:
        assert len(labeling) == len(C_est.columns)
        pca_df.loc[:, "labels"] = labeling

        sns.scatterplot(x="PC1", y="PC2", data=pca_df, hue="labels", ax=ax)
        ax.legend(title=group_name, bbox_to_anchor=(1.02, 1), loc="upper left")
    else:
        sns.scatterplot(x="PC1", y="PC2", data=pca_df, ax=ax)

    ax.set_title(f"PCA{title_suffix}")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}%)")

    if out_path is not None:
        fig.savefig(out_path, dpi=300, bbox_inches="tight")

    plt.close(fig)
    # Save plot at out-path


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


def plot_umap(
    C_est: pd.DataFrame,
    out_path: str,
    labeling: list = [],
    group_name="Cohorts",
    title_suffix: str = "",
) -> None:
    """
    Performs a principal component analysis combined with an universal manifold projection on the given composition and plots the result.
    The labeling list can be used to add a coloring to the points.

    Parameters
    ----------
    C_est : pd.DataFrame
        Composition dataframe (celltypes x bulks)
    out_path : str
        Filename + Path, where the plot will be stored.
    labeling : list = []
        List with labels for each bulk.
    group_name : str = "Cohorts"
        Name of the legend.
    title_suffix : str = ''
        Suffix displayed after the image title.
    """

    df = C_est.T

    df_scaled = StandardScaler().fit_transform(df)

    pca = PCA()
    X_pca = pca.fit_transform(df_scaled)

    reducer = umap.UMAP(random_state=2304)
    embedding = reducer.fit_transform(X_pca)

    umap_df = pd.DataFrame(
        {
            "UMAP1": embedding[:, 0],
            "UMAP2": embedding[:, 1],
        },
        index=df.index,
    )

    fig, ax = plt.subplots(figsize=(7, 5))

    if len(labeling) > 0:
        assert len(labeling) == len(C_est.columns)
        umap_df.loc[:, "labels"] = labeling

        sns.scatterplot(x="UMAP1", y="UMAP2", data=umap_df, hue="labels", ax=ax)
        ax.legend(title=group_name, bbox_to_anchor=(1.02, 1), loc="upper left")
    else:
        sns.scatterplot(x="UMAP1", y="UMAP2", data=umap_df, ax=ax)

    ax.set_title(f"UMAP{title_suffix}")

    if out_path is not None:
        fig.savefig(out_path, dpi=300, bbox_inches="tight")

    plt.close(fig)
    # Save plot at out-path
