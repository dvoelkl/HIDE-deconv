"""
=====================================================
Pipeline for training HIDE and estimating the
optimal gene weights
=====================================================
"""

from pathlib import Path
import pandas as pd

from ..config import hidedeconv_config
from ..models import HIDE
from ..visualization import plot_loss, plot_eval


def train_pipeline(hidedeconv_path: Path, plt: bool = True):

    hconf = hidedeconv_config.load(str(hidedeconv_path) + "/config.json")

    # Load the data
    Y_train = pd.read_csv(str(hidedeconv_path) + "/processed/Y_train.csv", index_col=0)
    C_train = pd.read_csv(str(hidedeconv_path) + "/processed/C_train.csv", index_col=0)

    X_sub = pd.read_csv(str(hidedeconv_path) + "/data/X_sub.csv", index_col=0)
    A_sub = pd.read_csv(str(hidedeconv_path) + "/data/A_sub.csv", index_col=0)

    X_ls = [X_sub]
    A_ls = [A_sub]
    for l1 in range(len(hconf.higher_ct_cols)):
        X_l = pd.read_csv(
            str(hidedeconv_path) + f"/data/X_{hconf.higher_ct_cols[l1]}.csv",
            index_col=0,
        )
        A_l = pd.read_csv(
            str(hidedeconv_path) + f"/data/A_{hconf.higher_ct_cols[l1]}.csv",
            index_col=0,
        )

        X_ls.append(X_l)
        A_ls.append(A_l)

    hide_model = HIDE(X_ls, A_ls)

    loss = hide_model.train(Y_train, C_train, hconf.n_hide_iter)

    for l2, g_l in enumerate(hide_model.g_l):
        if l2 == 0:
            pd.Series(g_l.detach().numpy(), index=X_sub.index).to_csv(
                str(hidedeconv_path) + "/processed/g_sub.csv"
            )
        else:
            pd.Series(g_l.detach().numpy(), index=X_sub.index).to_csv(
                str(hidedeconv_path)
                + f"/processed/g_{hconf.higher_ct_cols[l2 - 1]}.csv"
            )

    hconf.trained = True
    hconf.save(str(hidedeconv_path) + "/config.json")

    if plt:
        plot_loss(loss, str(hidedeconv_path) + "/figures/training_loss.png", hconf)

        C_est = hide_model.predict(Y_train, norm=True)["prediction"][0]

        df = plot_eval(
            C_train, C_est, str(hidedeconv_path) + "/figures/training_results.png"
        )
        df.to_csv(str(hidedeconv_path) + "/results/training_results.csv")
