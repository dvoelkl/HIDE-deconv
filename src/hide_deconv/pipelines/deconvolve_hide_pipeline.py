"""
=====================================================
Pipeline for deconvolution with HIDE
=====================================================
"""

from __future__ import annotations

from pathlib import Path
import os
import warnings

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from torch import float32, tensor

from ..config import hidedeconv_config
from ..models import HIDE
from ..preprocessing import create_bulks


DOMAIN_TRANSFER_BULK_COUNT = 1000
DOMAIN_TRANSFER_PREDS_PER_BULK = 5
DOMAIN_TRANSFER_ALPHA_WINDOW = 25
DOMAIN_TRANSFER_SEED = 2304


def normalize_bulk_to_cpm(bulk: pd.DataFrame) -> pd.DataFrame:
    bulk = bulk.copy()
    if bulk.index.has_duplicates:
        bulk = bulk.groupby(level=0).sum()

    bulk = (bulk * 1e6) / bulk.sum(axis=0)
    bulk = bulk.replace([np.inf, -np.inf], 0).fillna(0)
    return bulk


def predict_deconvolution_results(
    model: HIDE,
    adata: ad.AnnData | None,
    bulk: pd.DataFrame,
    celltype_col: str,
    n_cells_per_bulk: int,
    domain_transfer_bulk_count: int = DOMAIN_TRANSFER_BULK_COUNT,
    preds_per_bulk: int = DOMAIN_TRANSFER_PREDS_PER_BULK,
    alpha_window: int = DOMAIN_TRANSFER_ALPHA_WINDOW,
    domain_transfer: bool = True,
    seed: int = DOMAIN_TRANSFER_SEED,
    return_errors: bool = False,
) -> tuple[list[pd.DataFrame], list[pd.DataFrame]]:
    if not domain_transfer or bulk.shape[1] < alpha_window + preds_per_bulk:
        if domain_transfer and bulk.shape[1] < alpha_window + preds_per_bulk:
            warnings.warn(
                "Not enough bulk samples for the domain-transfer windowing scheme. Estimating composition without domain transfer!"
            )

        predictions = model.predict(bulk, norm=True)["prediction"]

        return predictions, []

    if adata is None:
        raise ValueError("adata is required when domain_transfer is enabled.")

    common_genes = [gene for gene in model.gene_labels if gene in adata.var_names]
    adata = adata[:, common_genes].copy()
    sc.pp.normalize_total(adata, target_sum=1e4)

    y_domain, _ = create_bulks(
        adata,
        domain_transfer_bulk_count,
        n_cells_per_bulk,
        celltype_col=celltype_col,
        seed=seed,
    )

    bulk_names = bulk.columns.tolist()
    n_bulks = len(bulk_names)
    window_step = max(
        1,
        int(round((n_bulks - alpha_window) / preds_per_bulk)),
    )
    starts = list(range(0, n_bulks, window_step))

    common_genes = y_domain.index.intersection(bulk.index).intersection(
        model.gene_labels
    )
    y_domain = y_domain.loc[common_genes]
    bulk = bulk.loc[common_genes]

    accum = [{bulk_name: [] for bulk_name in bulk_names} for _ in range(model.L)]

    for start in starts:
        alpha_idx = [(start + offset) % n_bulks for offset in range(alpha_window)]
        alpha_cols = [bulk_names[idx] for idx in alpha_idx]

        y_alpha = bulk.loc[common_genes, alpha_cols]
        alpha = y_alpha.median(axis=1) / y_domain.median(axis=1)
        alpha = alpha.replace([np.inf, -np.inf], 0).fillna(1.0).loc[common_genes]

        test_cols = [
            bulk_name for bulk_name in bulk_names if bulk_name not in alpha_cols
        ]
        if not test_cols:
            continue

        y_test = bulk.loc[common_genes, test_cols]
        y_test_adj = y_test.mul(1.0 / alpha, axis=0)

        preds = model.predict(y_test_adj, norm=True)["prediction"]

        for layer in range(model.L):
            c_pred = preds[layer]
            for col in c_pred.columns:
                accum[layer][col].append(c_pred[[col]])

    deconv_res: list[pd.DataFrame] = []
    deconv_errs: list[pd.DataFrame] = []
    for layer in range(model.L):
        means = []
        errors = []
        for bulk_name in bulk_names:
            stacked = pd.concat(accum[layer][bulk_name], axis=1)
            means.append(stacked.mean(axis=1))
            if return_errors:
                errors.append(stacked.std(axis=1, ddof=0))

        c_mean = pd.concat(means, axis=1)
        c_mean.columns = bulk_names
        deconv_res.append(c_mean)

        if return_errors:
            c_std = pd.concat(errors, axis=1)
            c_std.columns = bulk_names
            deconv_errs.append(c_std)

    if return_errors:
        return deconv_res, deconv_errs
    else:
        return deconv_res, []


def deconvolve_hide_pipeline(
    hidedeconv_path: Path, alternative_bulk_path=None
) -> list[pd.DataFrame]:
    """
    Run HIDE deconvolution pipeline.

    HIDE returns the estimated cell type proportions under the assumption,
    that the total proportion per sample sums to 1.

    **Important Remark:** If an alternative bulk is given, ensure, that the genes are the same as used in bulk used for configuration

    Parameters
    ----------
    hidedeconv_path: Path
        Path to the HIDE-deconv configuration file.

    alternative_bulk_path : Path, default = None
        Path to an alternative bulk, that should be used for deconvolution (instead of bulk from configuration).
        Bulk should be an genes x samples csv file.
    Returns
    -------
    list[pd.DataFrame]
        List containing the estimated proportions of the sample.
        Ordered in such a way, that the sub level l=0 can be accessed
        by result[0].
    """

    hconf = hidedeconv_config.load(str(hidedeconv_path) + "/config.json")

    # Load model data
    X_sub = pd.read_csv(str(hidedeconv_path) + "/data/X_sub.csv", index_col=0)
    A_sub = pd.read_csv(str(hidedeconv_path) + "/data/A_sub.csv", index_col=0)

    # Use either the predefined bulk of the configuration or if given an alternative bulk
    if alternative_bulk_path is None:
        Y_bulk = pd.read_csv(
            str(hidedeconv_path) + "/processed/Y_bulk.csv", index_col=0
        )
        # ensure bulk uses the same genes as the trained model references
        Y_bulk = Y_bulk.loc[X_sub.index, :]
        results_name = "HIDE"
    else:
        Y_bulk = pd.read_csv(alternative_bulk_path, index_col=0)
        Y_bulk = normalize_bulk_to_cpm(Y_bulk)
        Y_bulk = Y_bulk.loc[X_sub.index, :]
        results_name = f"HIDE_{Path(alternative_bulk_path).stem}"

    if not os.path.exists(str(hidedeconv_path) + f"/results/{results_name}/"):
        os.makedirs(str(hidedeconv_path) + f"/results/{results_name}/")

    X_ls = [X_sub]
    A_ls = [A_sub]

    for layer in range(len(hconf.higher_ct_cols)):
        X_l = pd.read_csv(
            str(hidedeconv_path) + f"/data/X_{hconf.higher_ct_cols[layer]}.csv",
            index_col=0,
        )
        A_l = pd.read_csv(
            str(hidedeconv_path) + f"/data/A_{hconf.higher_ct_cols[layer]}.csv",
            index_col=0,
        )

        X_ls.append(X_l)
        A_ls.append(A_l)

    hide_model = HIDE(X_ls, A_ls)

    # Load learned gene weights and store them back in the model
    for layer in range(hide_model.L):
        if layer == 0:
            g_l = pd.read_csv(
                str(hidedeconv_path) + "/processed/g_sub.csv", index_col=0
            ).to_numpy()
        else:
            g_l = pd.read_csv(
                str(hidedeconv_path)
                + f"/processed/g_{hconf.higher_ct_cols[layer - 1]}.csv",
                index_col=0,
            ).to_numpy()
        hide_model.g_l[layer] = tensor(
            g_l[:, 0], dtype=float32
        )  # manually set type, otherwise torch might be unhappy!

    if hconf.domainTransfer:
        adata = ad.read_h5ad(hconf.sc_file_name)
        common_sc_genes = [gene for gene in X_sub.index if gene in adata.var_names]
        adata = adata[:, common_sc_genes].copy()

        deconv_res, deconv_errs = predict_deconvolution_results(
            hide_model,
            adata,
            Y_bulk,
            hconf.sub_ct_col,
            hconf.n_cells_per_bulk,
            domain_transfer_bulk_count=hconf.domain_transfer_bulk_count,
            preds_per_bulk=hconf.preds_per_bulk,
            alpha_window=hconf.alpha_window,
            domain_transfer=True,
            seed=2304,
            return_errors=True,
        )

    else:
        deconv_res, deconv_errs = predict_deconvolution_results(
            hide_model,
            None,
            Y_bulk,
            hconf.sub_ct_col,
            hconf.n_cells_per_bulk,
            domain_transfer_bulk_count=hconf.domain_transfer_bulk_count,
            preds_per_bulk=hconf.preds_per_bulk,
            alpha_window=hconf.alpha_window,
            domain_transfer=False,
        )

    for layer in range(hide_model.L):
        C_l = deconv_res[layer]

        if layer == 0:
            C_l.to_csv(str(hidedeconv_path) + f"/results/{results_name}/C_sub.csv")
            if len(deconv_errs) > 0:
                deconv_errs[layer].to_csv(
                    str(hidedeconv_path) + f"/results/{results_name}/err_C_sub.csv"
                )
        else:
            C_l.to_csv(
                str(hidedeconv_path)
                + f"/results/{results_name}/C_{hconf.higher_ct_cols[layer - 1]}.csv",
            )
            if len(deconv_errs) > 0:
                deconv_errs[layer].to_csv(
                    str(hidedeconv_path)
                    + f"/results/{results_name}/err_C_{hconf.higher_ct_cols[layer - 1]}.csv",
                )

    return deconv_res
