"""
=====================================================
Tests for the "lazy" deconvolution
=====================================================
"""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd

from hide_deconv import deconvolution


def make_test_adata() -> ad.AnnData:
    genes = ["gene_1", "gene_2", "gene_3", "gene_4"]
    obs = pd.DataFrame(
        {
            "cell_type": ["a1", "a1", "a2", "a2", "b1", "b1", "b2", "b2"],
            "major": ["a", "a", "a", "a", "b", "b", "b", "b"],
        },
        index=[
            "cell_1",
            "cell_2",
            "cell_3",
            "cell_4",
            "cell_5",
            "cell_6",
            "cell_7",
            "cell_8",
        ],
    )
    x = np.array(
        [
            [8, 2, 1, 1],
            [7, 2, 1, 1],
            [1, 8, 1, 1],
            [1, 7, 1, 1],
            [1, 1, 8, 2],
            [1, 1, 7, 2],
            [1, 1, 1, 8],
            [1, 1, 1, 7],
        ],
        dtype=float,
    )
    return ad.AnnData(x, obs=obs, var=pd.DataFrame(index=genes))


def make_test_bulk() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sample_1": [30, 16, 11, 9],
            "sample_2": [9, 11, 16, 30],
        },
        index=["gene_1", "gene_2", "gene_3", "gene_4"],
    )


def test_deconvolution_returns_layer_list() -> None:
    adata = make_test_adata()
    bulk = make_test_bulk()

    results = deconvolution(
        adata,
        bulk,
        celltype_cols=["cell_type", "major"],
        n_genes=4,
        n_train_bulks=8,
        n_cells_per_bulk=4,
        n_iter=2,
        domain_transfer=False,
        seed=1,
    )

    assert len(results) == 2
    assert results[0].shape == (4, 2)
    assert results[1].shape == (2, 2)
    assert np.isfinite(results[0].to_numpy()).all()


def test_deconvolution_defaults_to_cell_type_column() -> None:
    adata = make_test_adata().copy()
    adata.obs = adata.obs[["cell_type"]]
    bulk = make_test_bulk()

    results = deconvolution(
        adata,
        bulk,
        n_genes=4,
        n_train_bulks=8,
        n_cells_per_bulk=4,
        n_iter=2,
        domain_transfer=False,
        seed=2,
    )

    assert len(results) == 1
    assert results[0].shape == (4, 2)
