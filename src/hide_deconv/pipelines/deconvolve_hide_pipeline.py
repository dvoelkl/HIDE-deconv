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

    # Deconvolve and save estimated compositions
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
