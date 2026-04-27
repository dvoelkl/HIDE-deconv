"""
=====================================================
Unit tests for HIDE model
=====================================================
"""

import pytest
import torch
import pandas as pd
import numpy as np

from hide_deconv.models.HIDE import HIDE


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


class TestHIDEInitialization:
    """
    Tests for HIDE model initialization.
    """

    def test_hide_init_with_valid_data(self, sample_reference_data) -> None:
        """
        Test that HIDE can be initialized with valid reference data.
        """
        X_l = [sample_reference_data]
        A_l = [pd.DataFrame(np.eye(10), index=[f"celltype_{i}" for i in range(1, 11)])]

        model = HIDE(X_l, A_l)

        assert model.L == 1
        assert model.p == 100
        assert len(model.g_l) == 1
        assert model.g_l[0].shape == torch.Size([100])

    def test_hide_init_with_multiple_layers(self, sample_reference_data) -> None:
        """
        Test that HIDE can be initialized with multiple hierarchy layers.
        """
        X_l = [sample_reference_data, sample_reference_data]
        A_l = [
            pd.DataFrame(np.eye(10), index=[f"celltype_{i}" for i in range(1, 11)]),
            pd.DataFrame(np.eye(10), index=[f"celltype_{i}" for i in range(1, 11)]),
        ]

        model = HIDE(X_l, A_l)

        assert model.L == 2
        assert len(model.g_l) == 2

    def test_hide_init_stores_gene_labels(self, sample_reference_data) -> None:
        """
        Test that HIDE correctly stores gene labels from reference data.
        """
        X_l = [sample_reference_data]
        A_l = [pd.DataFrame(np.eye(10), index=[f"celltype_{i}" for i in range(1, 11)])]

        model = HIDE(X_l, A_l)

        expected_genes = [f"gene_{i}" for i in range(1, 101)]
        assert list(model.gene_labels) == expected_genes

    def test_hide_init_stores_celltype_labels(self, sample_reference_data) -> None:
        """
        Test that HIDE correctly stores cell type labels from reference data.
        """
        X_l = [sample_reference_data]
        A_l = [pd.DataFrame(np.eye(10), index=[f"celltype_{i}" for i in range(1, 11)])]

        model = HIDE(X_l, A_l)

        assert len(model.celltype_layer_labels) == 1
        expected_celltypes = [f"celltype_{i}" for i in range(1, 11)]
        assert list(model.celltype_layer_labels[0]) == expected_celltypes

    def test_hide_init_with_lambdaNMSE(self, sample_reference_data) -> None:
        """
        Test that HIDE stores lambdaNMSE parameter correctly.
        """
        X_l = [sample_reference_data]
        A_l = [pd.DataFrame(np.eye(10), index=[f"celltype_{i}" for i in range(1, 11)])]
        lambda_val = 0.5

        model = HIDE(X_l, A_l, lambdaNMSE=lambda_val)

        assert model.lambdaNMSE == lambda_val

    def test_hide_init_gene_weights_in_valid_range(self, sample_reference_data) -> None:
        """
        Test that HIDE initializes gene weights in expected range [0.001, 0.1].
        """

        X_l = [sample_reference_data]
        A_l = [pd.DataFrame(np.eye(10), index=[f"celltype_{i}" for i in range(1, 11)])]

        model = HIDE(X_l, A_l)
        gene_weights = model.g_l[0].detach().numpy()

        assert np.all(gene_weights >= 0.001)
        assert np.all(gene_weights <= 0.1)


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


class TestHIDELoss:
    """
    Tests for HIDE loss calculation.
    """

    def test_hide_loss_returns_three_tensors(self, sample_reference_data) -> None:
        """
        Test that get_loss returns three loss components: total, correlation, NMSE.
        """
        X_l = [sample_reference_data]
        A_l = [pd.DataFrame(np.eye(10), index=[f"celltype_{i}" for i in range(1, 11)])]
        model = HIDE(X_l, A_l)

        C = torch.randn(10, 50)
        C_est = torch.randn(10, 50)

        total_loss, corr_loss, nmse_loss = model.get_loss(C, C_est)

        assert isinstance(total_loss, torch.Tensor)
        assert isinstance(corr_loss, torch.Tensor)
        assert isinstance(nmse_loss, torch.Tensor)

    def test_hide_loss_scalar_output(self, sample_reference_data) -> None:
        """
        Test that get_loss returns scalar tensors (shape []).
        """
        X_l = [sample_reference_data]
        A_l = [pd.DataFrame(np.eye(10), index=[f"celltype_{i}" for i in range(1, 11)])]
        model = HIDE(X_l, A_l)

        C = torch.randn(10, 50)
        C_est = torch.randn(10, 50)

        total_loss, corr_loss, nmse_loss = model.get_loss(C, C_est)

        assert total_loss.shape == torch.Size([])
        assert corr_loss.shape == torch.Size([])
        assert nmse_loss.shape == torch.Size([])

    def test_hide_loss_with_zero_lambdaNMSE(self, sample_reference_data) -> None:
        """
        Test that total loss equals correlation loss when lambdaNMSE is zero.
        """
        X_l = [sample_reference_data]
        A_l = [pd.DataFrame(np.eye(10), index=[f"celltype_{i}" for i in range(1, 11)])]
        model = HIDE(X_l, A_l, lambdaNMSE=0.0)

        C = torch.randn(10, 50)
        C_est = torch.randn(10, 50)

        total_loss, corr_loss, nmse_loss = model.get_loss(C, C_est)

        assert torch.allclose(total_loss, corr_loss, atol=1e-5)

    def test_hide_loss_with_nonzero_lambdaNMSE(self, sample_reference_data) -> None:
        """
        Test that total loss includes NMSE contribution when lambdaNMSE > 0.
        """
        X_l = [sample_reference_data]
        A_l = [pd.DataFrame(np.eye(10), index=[f"celltype_{i}" for i in range(1, 11)])]
        lambda_val = 0.5
        model = HIDE(X_l, A_l, lambdaNMSE=lambda_val)

        C = torch.randn(10, 50)
        C_est = torch.randn(10, 50)

        total_loss, corr_loss, nmse_loss = model.get_loss(C, C_est)

        expected_loss = corr_loss + lambda_val * nmse_loss
        assert torch.allclose(total_loss, expected_loss, atol=1e-5)

    def test_hide_loss_shape_mismatch_C_samples(self, sample_reference_data) -> None:
        """
        Test that get_loss handles mismatched number of samples in C and C_est.
        """
        X_l = [sample_reference_data]
        A_l = [pd.DataFrame(np.eye(10), index=[f"celltype_{i}" for i in range(1, 11)])]
        model = HIDE(X_l, A_l)

        C = torch.randn(10, 50)
        C_est = torch.randn(10, 30)

        with pytest.raises(RuntimeError):
            model.get_loss(C, C_est)


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


class TestHIDETraining:
    """
    Tests for HIDE model training.
    """

    def test_hide_train_returns_loss_list(
        self, sample_reference_data, sample_bulk_data, sample_composition_data
    ) -> None:
        """
        Test that train method returns a list of losses.
        """
        X_l = [sample_reference_data]
        A_l = [pd.DataFrame(np.eye(10), index=[f"celltype_{i}" for i in range(1, 11)])]
        model = HIDE(X_l, A_l)

        losses = model.train(sample_bulk_data, sample_composition_data, iter=5)

        assert isinstance(losses, list)
        assert len(losses) == 5
        assert all(isinstance(loss, (int, float)) for loss in losses)

    def test_hide_train_loss_decreases_over_iterations(
        self, sample_reference_data, sample_bulk_data, sample_composition_data
    ) -> None:
        """
        Test that loss decreases over training iterations.
        """
        X_l = [sample_reference_data]
        A_l = [pd.DataFrame(np.eye(10), index=[f"celltype_{i}" for i in range(1, 11)])]
        model = HIDE(X_l, A_l)

        losses = model.train(sample_bulk_data, sample_composition_data, iter=20)

        assert losses[-1] < losses[0]

    def test_hide_train_gene_weights_non_negative(
        self, sample_reference_data, sample_bulk_data, sample_composition_data
    ) -> None:
        """
        Test that gene weights remain non-negative after training.
        """
        X_l = [sample_reference_data]
        A_l = [pd.DataFrame(np.eye(10), index=[f"celltype_{i}" for i in range(1, 11)])]
        model = HIDE(X_l, A_l)

        model.train(sample_bulk_data, sample_composition_data, iter=5)

        gene_weights = model.g_l[0].detach().numpy()

        assert np.all(gene_weights >= 0)

    def test_hide_train_updates_parameters(
        self, sample_reference_data, sample_bulk_data, sample_composition_data
    ) -> None:
        """
        Test that training updates gene weight parameters.
        """
        X_l = [sample_reference_data]
        A_l = [pd.DataFrame(np.eye(10), index=[f"celltype_{i}" for i in range(1, 11)])]
        model = HIDE(X_l, A_l)

        initial_weights = model.g_l[0].detach().clone()

        model.train(sample_bulk_data, sample_composition_data, iter=5)

        final_weights = model.g_l[0].detach()

        assert not torch.allclose(initial_weights, final_weights)


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


class TestHIDEPrediction:
    """
    Tests for HIDE model prediction.
    """

    def test_hide_predict_returns_dataframe(
        self, sample_reference_data, sample_bulk_data
    ) -> None:
        """
        Test that predict method returns a pandas DataFrame.
        """
        X_l = [sample_reference_data]
        A_l = [pd.DataFrame(np.eye(10), index=[f"celltype_{i}" for i in range(1, 11)])]
        model = HIDE(X_l, A_l)

        result = model.predict(sample_bulk_data)["prediction"][0]

        assert isinstance(result, pd.DataFrame)

    def test_hide_predict_output_shape(
        self, sample_reference_data, sample_bulk_data
    ) -> None:
        """
        Test that predict output has correct shape (cell types x samples).
        """
        X_l = [sample_reference_data]
        A_l = [pd.DataFrame(np.eye(10), index=[f"celltype_{i}" for i in range(1, 11)])]
        model = HIDE(X_l, A_l)

        result = model.predict(sample_bulk_data)["prediction"][0]

        assert result.shape == (10, 50)

    def test_hide_predict_output_non_negative(
        self, sample_reference_data, sample_bulk_data
    ) -> None:
        """
        Test that predicted compositions are non-negative.
        """
        X_l = [sample_reference_data]
        A_l = [pd.DataFrame(np.eye(10), index=[f"celltype_{i}" for i in range(1, 11)])]
        model = HIDE(X_l, A_l)

        result = model.predict(sample_bulk_data)["prediction"][0]

        assert (result >= 0).all().all()

    def test_hide_predict_sample_names_preserved(
        self, sample_reference_data, sample_bulk_data
    ) -> None:
        """
        Test that predict preserves sample names in output.
        """
        X_l = [sample_reference_data]
        A_l = [pd.DataFrame(np.eye(10), index=[f"celltype_{i}" for i in range(1, 11)])]
        model = HIDE(X_l, A_l)

        result = model.predict(sample_bulk_data)["prediction"][0]

        expected_samples = [f"sample_{i}" for i in range(1, 51)]
        assert list(result.columns) == expected_samples
