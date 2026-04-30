"""
=====================================================
Functions for visualization of AnnData objects
=====================================================
"""

import anndata as ad
import matplotlib.pyplot as plt
import scanpy as sc


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


def plot_anndata_umap(
    adata: ad.AnnData,
    out_path: str,
    obs_col: str = None,
    title_suffix: str = "",
) -> None:
    """
    Plot a UMAP of an AnnData DataFrame using scanpy.

    Parameters
    ----------
    adata : ad.AnnData
        AnnData DataFrame.
    out_path : str
        Filepath where the plot will be stored.
    obs_col : str = None
        Obs column name to use for coloring the UMAP.
    title_suffix : str = ""
        Suffix displayed after the image title.
    """
    adata = adata.copy()

    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata)
    sc.pp.scale(adata)
    sc.tl.pca(adata)
    sc.pp.neighbors(adata)
    sc.tl.umap(adata)

    title = f"UMAP{title_suffix}"

    fig = sc.pl.umap(
        adata,
        color=obs_col,
        show=False,
        title=title,
        return_fig=True,
        frameon=True,
    )

    if out_path is not None:
        fig.savefig(out_path, dpi=300, bbox_inches="tight")

    plt.close(fig)
