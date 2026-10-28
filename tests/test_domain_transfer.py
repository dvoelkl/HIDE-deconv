"""
=====================================================
Tests for the domain transfer during deconvolution
=====================================================
"""

import numpy as np
import scanpy as sc

from hide_deconv import generate_example
from hide_deconv.pipelines.deconvolve_hide_pipeline import (
    get_domain_transfer_windows,
    normalize_bulk_to_cpm,
    predict_deconvolution_results,
)
from hide_deconv.pipelines.lazy_deconvolution_pipeline import setup_model
from hide_deconv.preprocessing import get_common_genes

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


class TestDomainTransferWindows:
    """
    Tests for the assignment of bulk samples to the domain-transfer windows.
    """

    def test_windows_mix_sorted_cohorts(self) -> None:
        """
        Test that windows of a bulk sorted by cohort contain samples of both cohorts.
        """
        bulk_names = [f"healthy_{i:02d}" for i in range(30)] + [
            f"disease_{i:02d}" for i in range(30)
        ]

        windows = get_domain_transfer_windows(bulk_names, 25, 5, 2304)

        assert any(
            any(name.startswith("healthy") for name in window)
            and any(name.startswith("disease") for name in window)
            for window in windows
        )

    def test_windows_independent_of_order(self) -> None:
        """
        Test that the windows do not depend on the order of the sample names.
        """
        bulk_names = [f"sample_{i:02d}" for i in range(40)]
        shuffled = list(np.random.default_rng(0).permutation(bulk_names))

        assert get_domain_transfer_windows(
            bulk_names, 25, 5, 2304
        ) == get_domain_transfer_windows(shuffled, 25, 5, 2304)

    def test_windows_contain_each_sample_once(self) -> None:
        """
        Test that each window consists of alpha_window distinct samples.
        """
        bulk_names = [f"sample_{i:02d}" for i in range(40)]

        windows = get_domain_transfer_windows(bulk_names, 25, 5, 2304)

        assert all(len(set(window)) == 25 for window in windows)
        assert set().union(*windows) == set(bulk_names)


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


class TestDomainTransferPrediction:
    """
    Tests for the prediction with domain transfer.
    """

    def test_prediction_independent_of_column_order(self) -> None:
        """
        Test that the predictions do not depend on the column order of the bulk.
        """
        adata, bulk, _ = generate_example(n_cells=600, n_genes=500, n_patients=40)

        bulk = normalize_bulk_to_cpm(bulk)
        adata = sc.pp.normalize_total(adata, target_sum=1e4, copy=True)
        adata = adata[:, get_common_genes(adata, bulk)].copy()

        model, adata = setup_model(
            adata,
            ["cell_type", "major"],
            n_genes=200,
            n_train_bulks=200,
            n_cells_per_bulk=50,
            n_iter=20,
            seed=42,
        )
        bulk = bulk.loc[model.gene_labels]
        permuted = bulk[list(np.random.default_rng(0).permutation(bulk.columns))]

        results, _ = predict_deconvolution_results(
            model, adata, bulk, "cell_type", 50, domain_transfer=True
        )
        results_permuted, _ = predict_deconvolution_results(
            model, adata, permuted, "cell_type", 50, domain_transfer=True
        )

        for C, C_permuted in zip(results, results_permuted):
            assert list(C_permuted.columns) == list(permuted.columns)
            assert np.allclose(C.to_numpy(), C_permuted[C.columns].to_numpy())
