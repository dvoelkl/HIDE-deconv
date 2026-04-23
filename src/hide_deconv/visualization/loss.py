"""
=====================================================
Functions for visualization of loss
=====================================================
"""

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

from ..config import hidedeconv_config


def plot_loss(loss: list, save_path: str, hconf: hidedeconv_config):
    """
    Creates a lineplot for a list of loss elements.

    Parameters
    ----------
    loss : list
        List of numerical loss values

    save_path : str
        Path, where the loss plot will be stored.
    """
    losses = pd.DataFrame(loss, index=range(1, len(loss) + 1))

    sns.set_style("white")
    fig, ax = plt.subplots()
    sns.lineplot(data=losses, ax=ax, legend=False)

    ax.set_title("Training Loss")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss")

    ax.text(
        0.98,
        0.98,
        f"Genes: {hconf.n_genes}\n Bulks: {hconf.n_train_bulks}\n Layer: {1 + len(hconf.higher_ct_cols)}\n Cell/Bulk: {hconf.n_cells_per_bulk}\n DomTr: {hconf.domainTransfer}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
    )

    fig.savefig(save_path, bbox_inches="tight", dpi=300)

    plt.close(fig)
