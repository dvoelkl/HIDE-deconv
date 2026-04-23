"""
=====================================================
Methods for loading and preprocessing of training
data.
=====================================================
"""

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


def reduce_genes(adata: ad.AnnData, N: int, ct_col: str = "cell_type") -> ad.AnnData:
    """
    Reduce an AnnData object to the N most informative genes.

    The function computes the mean expression per cell type and selects
    the genes with the highest variance between cell types. It returns a
    copy of the AnnData object restricted to the selected genes.

    Parameters
    ----------
    adata : anndata.AnnData
        Input expression data.
    N : int
        Number of genes to select.
    ct_col : str, default="cell_type"
        Column in adata.obs containing the cell type labels used for variance calculation

    Returns
    -------
    anndata.AnnData
        A copy of the input AnnData object containing only the selected genes.

    """
    n_genes_total = adata.n_vars

    N = min(N, n_genes_total)

    ct = adata.obs[ct_col].astype(str).values
    celltypes = np.array(sorted(pd.unique(ct)))

    X = adata.X
    if sp.issparse(X):
        X = X.tocsr()
    else:
        X = np.asarray(X)

    mean_by_ct = np.zeros((len(celltypes), n_genes_total), dtype=np.float64)

    for i, c in enumerate(celltypes):
        rows = np.where(ct == c)[0]
        if rows.size == 0:
            continue
        if sp.issparse(X):
            mean_by_ct[i, :] = np.asarray(X[rows].mean(axis=0)).ravel()
        else:
            mean_by_ct[i, :] = X[rows, :].mean(axis=0)

    var_between = mean_by_ct.var(axis=0)

    top_idx = np.argsort(-var_between)[:N]
    top_genes = adata.var_names[top_idx]

    adata_red = adata[:, top_genes].copy()
    return adata_red


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


def create_reference(
    adata: ad.AnnData, celltype_col: str = "cell_type"
) -> pd.DataFrame:
    """
    Create archetypal cell type references from a given AnnData object.

    The function creates reference profiles from a given AnnData object by
    averaging over the gene expression profiles of each cell type.
    The used cell types are determined by the given cell type observation name.

    Parameters
    ----------
    adata : anndata.AnnData
        Input expression data.
    ct_col : str, default="cell_type"
        Column in adata.obs containing the cell type labels used for averaging over the gene expressions.

    Returns
    -------
    pd.DataFrame
       A gene x cell type pandas DataFrame containing the archetypal gene expression profiles of each cell type.

    """

    celltypes = adata.obs[celltype_col].astype(str).values

    X = adata.X
    if sp.issparse(X):
        X = X.tocsr()
    else:
        X = np.asarray(X)

    unique_ct = np.array(sorted(pd.unique(celltypes)))
    ref = np.zeros((adata.n_vars, len(unique_ct)), dtype=np.float64)

    for j, ct in enumerate(unique_ct):
        mask = celltypes == ct
        if mask.sum() == 0:
            continue
        if sp.issparse(X):
            ref[:, j] = np.asarray(X[mask].mean(axis=0)).ravel()
        else:
            ref[:, j] = X[mask].mean(axis=0)

    return pd.DataFrame(ref, index=adata.var_names, columns=unique_ct)


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


def create_hierarchy(
    adata: ad.AnnData, ct_col_sub: str, ct_col_higher: list[str]
) -> dict[str, pd.DataFrame]:
    """
    Create hiearchy mapping matrices between subtypes and higher-level cell types.

    The function constructs for each column in ct_col_higher a binary projection matrix
    that maps each subtype in ct_col_sub to its corresponding higher-level cell type.
    Each returned matrix has higher-level cell types as rows and subtypes as columns.

    Parameters
    ----------
    adata : anndata.AnnData
        Input expression data containing cell annotations in adata.obs
    ct_col_sub : str
        Column in adata.obs containing the lower-level cell type labels.
    ct_col_higher : list[str]
        List of column names in adata.obs containing the higher-level
        cell type labels for which hiearchy matrices should be created.

    Returns
    -------
    dict[str, pd.DataFrame]
        Dictionary mapping each higher-level annotation column name to a
        pandas DataFrame with shape (n_higher_types, n_subtypes), where
        entries are 1 if a subtype belongs to a higher-level type and
        0 otherwise.
    """

    obs = adata.obs.copy()

    sub = obs[ct_col_sub].astype(str)
    sub_types = sorted(sub.unique())

    A_dict = {}

    for col in ct_col_higher:
        high = obs[col].astype(str)
        map_df = pd.DataFrame({"sub": sub, "high": high}).drop_duplicates()

        sub_to_high = map_df.set_index("sub")["high"].to_dict()

        high_types = sorted(high.unique())
        A = pd.DataFrame(0, index=high_types, columns=sub_types, dtype=int)

        for s in sub_types:
            h = sub_to_high.get(s, None)
            if h is not None:
                A.loc[h, s] = 1

        A_dict[col] = A

    return A_dict


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


def train_test_split_adata(
    adata: ad.AnnData,
    celltype_col: str = "cell_type",
    train_frac: float = 0.5,
    seed: int = 42,
) -> tuple[ad.AnnData, ad.AnnData]:
    """
    Split an AnnData object into a train and test subset.

    The function splits the cells of each cell type independently into a training
    and a test subset. Samples are shuffled before splitting. For cell types
    with at least two samples the split ensures that both train and test
    receive at least one sample.

    Parameters
    ----------
    adata : anndata.AnnData
        Input AnnData object to split.
    celltype_col : str, default="cell_type"
        Column in adata.obs containing the cell type labels used for splitting
    train_frac : float, default=0.5
        Fraction of samplesper cell type assigned to the training split.
    seed : int, default=42
        Random seed used for shuffling before the split.

    Returns
    -------
    tuple[ad.AnnData, ad.AnnData]
        A tuple containing the training AnnData object and the test AnnData object.

    """

    rng = np.random.default_rng(seed)
    ct = adata.obs[celltype_col].astype(str).values

    train_idx = []
    test_idx = []

    for celltype in np.unique(ct):
        idx = np.where(ct == celltype)[0]
        rng.shuffle(idx)

        n_train = int(np.floor(len(idx) * train_frac))

        if len(idx) >= 2:
            n_train = max(1, min(n_train, len(idx) - 1))

        train_idx.extend(idx[:n_train])
        test_idx.extend(idx[n_train:])

    train_idx = np.array(train_idx, dtype=int)
    test_idx = np.array(test_idx, dtype=int)

    adata_train = adata[train_idx].copy()
    adata_test = adata[test_idx].copy()

    return adata_train, adata_test


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


def create_bulks(
    adata: ad.AnnData,
    n_bulks: int,
    n_cells_per_bulk: int,
    celltype_col: str = "cell_type",
    seed: int = 42,
    norm: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate in silico bulk expression samples and their cell type compositions.

    The function draws cells with replacement from the input AnnData object to create
    n_bulks bulk samples. For each bulk, expression counts are summed across the
    sampled cells and the corresponding cell type counts are tracked. Optionally, bulk
    expression is normalized to counts per million (CPM).

    Parameters
    ----------
    adata : ad.AnnData
        Input AnnData object.
    n_bulks : int
        Number of bulks samples to simulate.
    n_cells_per_bulk : int
        Number of cells sampled per simulated bulk.
    celltype_col : str, default="cell_type"
        Column in adata.obs containing cell type labels.
    seed : int, default=42
        Random seed for reproducible simulations.
    norm : bool, default=False
        If True each bulk is normalized to CPM.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        Y : gene x bulk expression profiles.
        C : celltype x bulk composition matrix
    """

    rng = np.random.default_rng(seed)

    celltypes = adata.obs[celltype_col].astype(str).values
    if hasattr(adata.X, "toarray"):
        expr = adata.X.toarray()
    elif hasattr(adata.X, "todense"):
        expr = adata.X.todense()
    else:
        expr = np.array(adata.X[:])

    n_cells, n_genes = expr.shape

    unique_celltypes = sorted(pd.unique(celltypes))
    n_celltypes = len(unique_celltypes)
    celltype_to_idx = {ct: i for i, ct in enumerate(unique_celltypes)}

    Y = np.zeros((n_genes, n_bulks), dtype=np.float32)
    C = np.zeros((n_celltypes, n_bulks), dtype=np.float32)

    for b in range(n_bulks):
        idx = rng.choice(n_cells, size=n_cells_per_bulk, replace=True)
        sampled_expr = expr[idx]
        sampled_celltypes = celltypes[idx]

        Y[:, b] = sampled_expr.sum(axis=0)

        for ct in sampled_celltypes:
            C[celltype_to_idx[ct], b] += 1

    if norm:
        Y = (Y * 1e6) / Y.sum(axis=0)  # cpm

    Y = pd.DataFrame(
        Y, index=adata.var_names, columns=[f"bulk_{i}" for i in range(n_bulks)]
    )
    C = pd.DataFrame(
        C, index=unique_celltypes, columns=[f"bulk_{i}" for i in range(n_bulks)]
    )
    C = C / C.sum(axis=0)

    return Y, C


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


def get_adata_info(ad_file: str) -> dict[str, object]:
    """
    Returns a summary of a AnnData file.

    Parameters
    ----------
    ad_file : str
        Path to AnnData file.

    Returns
    -------
    dict[str, object]
        Dictionary containing multiple metrics.
    """

    adata = ad.read_h5ad(ad_file)

    obs = adata.obs.columns.tolist()
    var = adata.var.columns.tolist()
    n_cells, n_genes = adata.shape
    return {"obs": obs, "var": var, "n_cells": n_cells, "n_genes": n_genes}
