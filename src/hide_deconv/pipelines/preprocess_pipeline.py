"""
=====================================================
Pipeline for execution of preprocessing
=====================================================
"""

from pathlib import Path
import anndata as ad
import pandas as pd
import numpy as np
import scanpy as sc

from ..config import hidedeconv_config
from ..preprocessing import (
    get_common_genes,
    reduce_genes,
    create_reference,
    create_hierarchy,
    create_bulks,
    get_domain_transfer_factor,
)


def preprocessing_pipeline(
    hidedeconv_path: Path, f_domainTransfer: bool = True, fSave: bool = True
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, list, list]:
    """
    Run preprocessing for HIDE-Deconv by aligning genes between single-cell and bulk data,
    creating reference/hierarchy matrices, generating training bulks, and optionally
    accounting for domain transfer.

    Parameters
    ----------
    hidedeconv_path : Path
        Path to the HIDE-Deconv project.
    f_domainTransfer : bool, default=True
        If True apply gene wise domain-transfer from simulated to observed bulk.
    fSave : bool, default=True
        If True save generated matrices and training data as CSV files.

    Returns
    -------
    tuple
        (X_sub, A_sub, Y_train, C_train, X_ls, A_ls)
    """
    hconf = hidedeconv_config.load(str(hidedeconv_path) + "/config.json")

    adata = ad.read_h5ad(hconf.sc_file_name)
    bulk = pd.read_csv(hconf.bulk_file_name, index_col=0)

    common_genes = get_common_genes(adata, bulk)

    # Convert to TPM
    bulk = (bulk * 1e6) / bulk.sum(axis=0)
    sc.pp.normalize_total(adata, target_sum=1e6)

    # Subset anndata file
    adata = adata[:, common_genes]

    # Reduce genes
    adata = reduce_genes(adata, hconf.n_genes, hconf.sub_ct_col)

    # Subset bulk to reduced genes of anndata
    bulk = bulk.loc[adata.var_names]

    X_sub = create_reference(adata, celltype_col=hconf.sub_ct_col)
    A_sub = pd.DataFrame(
        np.diag(np.ones(len(X_sub.columns))), index=X_sub.columns, columns=X_sub.columns
    )

    A_ls_dict = create_hierarchy(adata, hconf.sub_ct_col, hconf.higher_ct_cols)
    X_ls = []
    A_ls = []
    for l1 in range(len(hconf.higher_ct_cols)):
        X_l = create_reference(adata, celltype_col=hconf.higher_ct_cols[l1])
        A_l = A_ls_dict[hconf.higher_ct_cols[l1]]

        X_ls.append(X_l)
        A_ls.append(A_l)

    Y_train, C_train = create_bulks(
        adata, hconf.n_train_bulks, hconf.n_cells_per_bulk, hconf.sub_ct_col
    )

    if f_domainTransfer:
        Y_domTra, _ = create_bulks(
            adata,
            hconf.n_train_bulks,
            hconf.n_cells_per_bulk,
            hconf.sub_ct_col,
            seed=2304,
        )
        alpha = get_domain_transfer_factor(Y_domTra, bulk)
        alpha_inv = 1.0 / alpha
        alpha_inv.replace([np.inf, -np.inf], 0, inplace=True)

        Y_train = Y_train.mul(alpha_inv, axis=0)
        X_sub = X_sub.mul(alpha_inv, axis=0)

        for l2 in range(len(hconf.higher_ct_cols)):
            X_ls[l2] = X_ls[l2].mul(alpha_inv, axis=0)

        if fSave:
            alpha_inv.to_csv(str(hidedeconv_path) + "/data/alpha_inv.csv")

    # Save files
    if fSave:
        Y_train.to_csv(str(hidedeconv_path) + "/processed/Y_train.csv")
        C_train.to_csv(str(hidedeconv_path) + "/processed/C_train.csv")

        X_sub.to_csv(str(hidedeconv_path) + "/data/X_sub.csv")
        A_sub.to_csv(str(hidedeconv_path) + "/data/A_sub.csv")

        bulk.to_csv(str(hidedeconv_path) + "/processed/Y_bulk.csv")

        for l3 in range(len(hconf.higher_ct_cols)):
            X_ls[l3].to_csv(
                str(hidedeconv_path) + f"/data/X_{hconf.higher_ct_cols[l3]}.csv"
            )
            A_ls[l3].to_csv(
                str(hidedeconv_path) + f"/data/A_{hconf.higher_ct_cols[l3]}.csv"
            )

    return X_sub, A_sub, Y_train, C_train, X_ls, A_ls
