"""
=====================================================
Functions for visualization of differential gene expression
=====================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


def plot_volcano(results: pd.DataFrame, out_path: Path) -> None:
    """
    Visualizes pydeseq2 results as volcano plot

    Parameters
    ----------
    results : pd.DataFrame
        Results of pydeseq2 that must at least contain columns 'padj' and 'log2FoldChange'
    out_path : Path
        Path, where the figures is saved
    """

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    padj = results["padj"].fillna(1.0).clip(lower=1e-300)
    log_padj = -np.log10(padj)
    significant = padj < 0.05

    plt.figure(figsize=(7, 6))
    plt.scatter(
        results["log2FoldChange"],
        log_padj,
        c=np.where(significant, "#b22222", "#4c78a8"),
        s=14,
        alpha=0.8,
        linewidths=0,
    )
    plt.axvline(0.0, color="#666666", linewidth=1)
    plt.axhline(-np.log10(0.05), color="#666666", linewidth=1, linestyle="--")
    plt.xlabel("log2 fold change")
    plt.ylabel("-log10 adjusted p-value")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
