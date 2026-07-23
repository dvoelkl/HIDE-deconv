"""
=====================================================
Underlying pipeline for the lazy deconvolution API
=====================================================
"""

from __future__ import annotations


import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc

from ..models import HIDE
from ..preprocessing import (
    create_bulks,
    create_hierarchy,
    create_reference,
    get_common_genes,
    reduce_genes,
)
from .deconvolve_hide_pipeline import (
    normalize_bulk_to_cpm,
    predict_deconvolution_results,
)


def normalize_celltype_cols(
    celltype_cols: list[str] | tuple[str, ...] | str | None,
) -> list[str]:
    if celltype_cols is None:
        return ["cell_type"]

    if isinstance(celltype_cols, str):
        return [celltype_cols]

    cols = list(celltype_cols)
    if len(cols) == 0:
        raise ValueError("celltype_cols must contain at least one column name.")

    return cols


def validate_required_columns(adata: ad.AnnData, celltype_cols: list[str]) -> None:
    missing = [col for col in celltype_cols if col not in adata.obs.columns]
    if missing:
        raise KeyError(
            "The following cell type columns do not exists in the anndata file: "
            + ", ".join(missing)
        )


def setup_model(
    adata: ad.AnnData,
    celltype_cols: list[str],
    n_genes: int,
    n_train_bulks: int,
    n_cells_per_bulk: int,
    n_iter: int,
    seed: int,
) -> tuple[HIDE, ad.AnnData]:

    sc.pp.normalize_total(adata, target_sum=1e4)

    adata = reduce_genes(adata, n_genes, celltype_cols[0])

    X_sub = create_reference(adata, celltype_col=celltype_cols[0])
    A_sub = pd.DataFrame(
        np.eye(len(X_sub.columns), dtype=int),
        index=X_sub.columns,
        columns=X_sub.columns,
    )

    higher_cols = celltype_cols[1:]
    hierarchy = (
        create_hierarchy(adata, celltype_cols[0], higher_cols) if higher_cols else {}
    )

    X_ls = [X_sub]
    A_ls = [A_sub]
    for col in higher_cols:
        X_ls.append(create_reference(adata, celltype_col=col))
        A_ls.append(hierarchy[col])

    model = HIDE(X_ls, A_ls)

    Y_train, C_train = create_bulks(
        adata,
        n_train_bulks,
        n_cells_per_bulk,
        celltype_col=celltype_cols[0],
        seed=seed,
    )
    model.train(Y_train, C_train, iter=n_iter)

    return model, adata


def deconvolution(
    adata: ad.AnnData,
    bulk: pd.DataFrame,
    celltype_cols: list[str] | tuple[str, ...] | str | None = None,
    n_genes: int = 5000,
    n_train_bulks: int = 10000,
    n_cells_per_bulk: int = 100,
    n_iter: int = 1000,
    domain_transfer: bool = True,
    seed: int = 42,
) -> list[pd.DataFrame]:
    """
    Run preprocessing, training and deconvolution in one call.

    Parameters
    ----------
    adata : anndata.AnnData
        Annotated single-cell input data.
    bulk : pd.DataFrame
        Bulk expression with genes as rows and samples as columns.
    celltype_cols : list[str], default=None
        Cell type annotation columns in adata.obs. The first column is treated as
        the finest cell type layer. If no columns are specified, column cell_type is used.
    n_genes : int, default=5000
        Number of genes used for training and deconvolution.
    n_train_bulks : int, default=10000
        Number of training bulks generated for training.
    n_cells_per_bulk : int, default=100
        Number of cells sampled per training bulk.
    n_iter : int, default=1000
        Number of training iterations.
    domain_transfer : bool, default=True
        Correct for domain transfer between Single Cell and Bulk data.
    seed : int, default=42
        Random seed for the simulated training bulks.

    Returns
    -------
    list[pd.DataFrame]
        List of estimated composition, same order as the celltype_cols list.
    """

    celltype_cols = normalize_celltype_cols(celltype_cols)
    validate_required_columns(adata, celltype_cols)

    bulk = normalize_bulk_to_cpm(bulk)
    common_genes = get_common_genes(adata, bulk)
    if len(common_genes) == 0:
        raise ValueError("No shared genes found between adata and bulk.")

    adata = adata[:, common_genes]

    model, adata = setup_model(
        adata,
        celltype_cols,
        n_genes,
        n_train_bulks,
        n_cells_per_bulk,
        n_iter,
        seed,
    )

    bulk = bulk.loc[model.gene_labels]

    predictions, _ = predict_deconvolution_results(
        model,
        adata,
        bulk,
        celltype_cols[0],
        n_cells_per_bulk,
        domain_transfer=domain_transfer,
    )

    return predictions
