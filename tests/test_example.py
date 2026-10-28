"""
=====================================================
Tests for the synthetic example dataset
=====================================================
"""

import anndata as ad
import numpy as np
import pandas as pd

from hide_deconv import generate_example
from hide_deconv.cli_commands import create_example_data
from hide_deconv.constants import MSG_SUCCESS
from hide_deconv.utils import check_bulk_raw

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


class TestGenerateExample:
    """
    Tests for the generate_example function.
    """

    def test_returns_raw_counts(self) -> None:
        """
        Test that single cells and bulk consist of raw counts.
        """

        adata, bulk, _ = generate_example(n_cells=300, n_genes=500, n_patients=20)

        X = adata.X.toarray()
        assert np.allclose(X, np.round(X))
        assert (X >= 0).all()
        assert check_bulk_raw(bulk)
        assert adata.shape == (300, 500)
        assert bulk.shape == (500, 20)

    def test_hierarchy_is_consistent(self) -> None:
        """
        Test that each subtype belongs to exactly one major cell type.
        """

        adata, _, _ = generate_example(n_cells=300, n_genes=500, n_patients=20)

        n_major = adata.obs.groupby("cell_type", observed=True)["major"].nunique()

        assert (n_major == 1).all()
        assert adata.obs["cell_type"].nunique() == 6
        assert adata.obs["major"].nunique() == 3

    def test_sample_sheet_matches_bulk(self) -> None:
        """
        Test that the sample sheet contains all bulk samples and the expected columns.
        """

        _, bulk, sample_sheet = generate_example(
            n_cells=300, n_genes=500, n_patients=20
        )

        assert list(sample_sheet["SampleID"]) == list(bulk.columns)
        assert set(sample_sheet.columns) == {
            "SampleID",
            "disease",
            "medication",
            "age",
            "sex",
            "survival_time",
            "event",
        }
        assert set(sample_sheet["disease"]) == {"healthy", "disease"}
        assert (
            sample_sheet.loc[sample_sheet["disease"] == "healthy", "medication"]
            == "none"
        ).all()
        assert set(
            sample_sheet.loc[sample_sheet["disease"] == "disease", "medication"]
        ) <= {"Drug_A", "Drug_B", "Drug_C"}
        assert set(sample_sheet["event"]) <= {0, 1}

    def test_is_reproducible(self) -> None:
        """
        Test that the same seed generates the same data.
        """

        adata_1, bulk_1, sheet_1 = generate_example(
            seed=1, n_cells=300, n_genes=500, n_patients=20
        )
        adata_2, bulk_2, sheet_2 = generate_example(
            seed=1, n_cells=300, n_genes=500, n_patients=20
        )

        assert (adata_1.X != adata_2.X).nnz == 0
        pd.testing.assert_frame_equal(bulk_1, bulk_2)
        pd.testing.assert_frame_equal(sheet_1, sheet_2)


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


class TestExampleCommand:
    """
    Tests for the example CLI command.
    """

    def test_writes_example_files(self, tmp_path) -> None:
        """
        Test that the command stores AnnData file, bulk and sample sheet.
        """

        ret = create_example_data(tmp_path)

        assert ret == MSG_SUCCESS

        adata = ad.read_h5ad(tmp_path / "single_cells.h5ad")
        bulk = pd.read_csv(tmp_path / "bulk.csv", index_col=0)
        sample_sheet = pd.read_csv(tmp_path / "sample_sheet.csv")

        assert list(sample_sheet["SampleID"]) == list(bulk.columns)
        assert list(bulk.index) == list(adata.var_names)
