"""
=====================================================
Pipeline for deconvolution with HIDE
=====================================================
"""

from pathlib import Path
import pandas as pd
from torch import tensor, float32
import os

from ..config import hidedeconv_config
from ..models import HIDE
from ..preprocessing import create_bulks, get_domain_transfer_factor
import anndata as ad
import numpy as np
import scanpy as sc


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
        Y_bulk = (Y_bulk * 1e6) / Y_bulk.sum(axis=0)
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
        # Load anndata for preparing bulks for domain transfer (follows the same logic as in preprocess)
        adata = ad.read_h5ad(hconf.sc_file_name)
        common_sc_genes = [gene for gene in X_sub.index if gene in adata.var_names]
        adata = adata[:, common_sc_genes].copy()
        sc.pp.normalize_total(adata, target_sum=1e4)

        Y_domTra, _ = create_bulks(
            adata, 1000, hconf.n_cells_per_bulk, hconf.sub_ct_col, seed=2304
        )

        bulk_names = Y_bulk.columns.tolist()
        n_bulks = len(bulk_names)

        # Calculate slicing window for cross validation
        preds_per_bulk = int(hconf.preds_per_bulk)
        alpha_window = int(hconf.alpha_window)
        window_step = max(1, int(round((n_bulks - alpha_window) / preds_per_bulk)))

        starts = list(range(0, n_bulks, window_step))

        common_genes = Y_domTra.index.intersection(Y_bulk.index).intersection(
            X_sub.index
        )
        Y_domTra = Y_domTra.loc[common_genes]

        # Bulk names must be stored as dictionary, in order for correct assignment of results later
        accum = [{bn: [] for bn in bulk_names} for _ in range(hide_model.L)]

        for s in starts:
            alpha_idx = [(s + offset) % n_bulks for offset in range(alpha_window)]
            alpha_cols = [bulk_names[i] for i in alpha_idx]

            # Bulk for alpha calculation must not be used for deconvolution
            Y_alpha = Y_bulk.loc[common_genes, alpha_cols]

            alpha = get_domain_transfer_factor(Y_alpha, Y_domTra)
            alpha = alpha.loc[common_genes]

            alpha_inv = 1.0 / alpha
            alpha_inv.replace([np.inf, -np.inf], 0, inplace=True)
            alpha_inv.fillna(1.0, inplace=True)

            # "test_bulks" for used for deconvolution
            test_cols = [bn for bn in bulk_names if bn not in alpha_cols]
            Y_test = Y_bulk.loc[common_genes, test_cols]
            Y_test_adj = Y_test.mul(alpha_inv, axis=0)

            preds = hide_model.predict(Y_test_adj, norm=True)["prediction"]

            for layer in range(hide_model.L):
                C_pred = preds[layer]
                for col in C_pred.columns:
                    accum[layer][col].append(C_pred[[col]])

        # mean and std for each bulk
        deconv_res = []
        for layer in range(hide_model.L):
            #
            cols = []
            errs = []
            for bn in bulk_names:
                lst = accum[layer][bn]

                stacked = pd.concat(lst, axis=1)

                mean_s = stacked.mean(axis=1)
                std_s = stacked.std(axis=1, ddof=0)

                cols.append(mean_s)
                errs.append(std_s)

            C_mean = pd.concat(cols, axis=1)
            C_mean.columns = bulk_names
            C_std = pd.concat(errs, axis=1)
            C_std.columns = bulk_names

            if layer == 0:
                C_mean.to_csv(
                    str(hidedeconv_path) + f"/results/{results_name}/C_sub.csv"
                )
                C_std.to_csv(
                    str(hidedeconv_path) + f"/results/{results_name}/err_C_sub.csv"
                )
            else:
                C_mean.to_csv(
                    str(hidedeconv_path)
                    + f"/results/{results_name}/C_{hconf.higher_ct_cols[layer - 1]}.csv",
                )
                C_std.to_csv(
                    str(hidedeconv_path)
                    + f"/results/{results_name}/err_C_{hconf.higher_ct_cols[layer - 1]}.csv",
                )

            deconv_res.append(C_mean)

        return deconv_res
    else:
        deconv_res = hide_model.predict(Y_bulk, norm=True)["prediction"]

        for layer in range(hide_model.L):
            C_l = deconv_res[layer]

            if layer == 0:
                C_l.to_csv(
                    str(hidedeconv_path) + f"/results/{results_name}/C_sub.csv",
                )
            else:
                C_l.to_csv(
                    str(hidedeconv_path)
                    + f"/results/{results_name}/C_{hconf.higher_ct_cols[layer - 1]}.csv",
                )

        return deconv_res
