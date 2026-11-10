"""
=====================================================
Tests for AnnData command helpers and annotation flow
=====================================================
"""

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

import hide_deconv.cli_commands.anndata_command as anndata_command
from hide_deconv.constants import (
    HB_GENES_ENSEMBL,
    MALAT1_ENSEMBL,
    MITO_GENES_ENSEMBL,
    MSG_SUCCESS,
    MSG_USER_ABORT,
    RIBO_GENES_ENSEMBL,
)
from hide_deconv.pipelines import preprocess_anndata_file
from hide_deconv.utils import (
    add_annotation_columns_from_template,
    create_annotation_template,
    get_var_names_from_column,
    switch_adata_var_names,
)


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


def create_sample_anndata() -> ad.AnnData:
    """
    Create a small AnnData object for annotation tests.
    """

    x = np.array(
        [
            [1, 0, 3],
            [0, 2, 1],
            [4, 1, 0],
        ]
    )

    obs = pd.DataFrame(
        {
            "cell_type": ["a", "b", "a"],
        },
        index=["cell_1", "cell_2", "cell_3"],
    )

    var = pd.DataFrame(index=["gene_1", "gene_2", "gene_3"])

    return ad.AnnData(X=x, obs=obs, var=var)


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


class TestAnnotationHelpers:
    """
    Tests for pure annotation helper functions.
    """

    def test_create_annotation_template_duplicates_selected_column(self) -> None:
        """
        Test that the template contains the selected column and the example column.
        """

        adata = create_sample_anndata()

        template = create_annotation_template(adata, "cell_type")

        assert list(template.columns) == ["cell_type", "new_cell_layer_example"]
        assert template["cell_type"].tolist() == ["a", "b"]
        assert template["new_cell_layer_example"].tolist() == ["a", "b"]

    def test_add_annotation_columns_from_template_adds_new_obs_columns(self) -> None:
        """
        Test that edited template columns are copied into adata.obs.
        """

        adata = create_sample_anndata()
        template = create_annotation_template(adata, "cell_type")
        template["higher_layer"] = ["x", "y"]

        updated_adata, added_columns = add_annotation_columns_from_template(
            adata, template, "cell_type"
        )

        assert added_columns == ["new_cell_layer_example", "higher_layer"]
        assert "new_cell_layer_example" in updated_adata.obs.columns
        assert "higher_layer" in updated_adata.obs.columns
        assert updated_adata.obs["higher_layer"].tolist() == ["x", "y", "x"]


class TestAddAnnotationCommand:
    """
    Tests for the interactive add-annotation command.
    """

    def test_add_annotation_writes_template_and_updates_obs(
        self, monkeypatch, tmp_path
    ) -> None:
        """
        Test that the command writes a template and stores new obs columns.
        """

        adata = create_sample_anndata()
        ad_path = tmp_path / "single_cells.h5ad"
        adata.write_h5ad(ad_path)

        template_path = tmp_path / "single_cells_annotation_template.csv"

        class MockPrompt:
            def __init__(self, value):
                self.value = value

            def execute(self):
                return self.value

        class MockFilePath:
            def __init__(self, value):
                self.value = value

            def execute(self):
                return self.value

        class MockConfirm:
            def __init__(self, callback):
                self.callback = callback

            def execute(self):
                self.callback()
                return True

        def edit_template_file() -> None:
            template_df = pd.read_csv(template_path)
            template_df["higher_layer"] = ["x", "y"]
            template_df.to_csv(template_path, index=False)

        monkeypatch.setattr(
            anndata_command.inquirer,
            "filepath",
            lambda *args, **kwargs: MockFilePath(str(ad_path)),
        )
        monkeypatch.setattr(
            anndata_command, "prompt", lambda *args, **kwargs: ["cell_type"]
        )
        monkeypatch.setattr(
            anndata_command.inquirer,
            "confirm",
            lambda *args, **kwargs: MockConfirm(edit_template_file),
        )
        monkeypatch.setattr(
            anndata_command.console, "print_exception", lambda *args, **kwargs: None
        )

        result = anndata_command.add_annotation()

        assert result == 0
        assert template_path.exists()

        annotated_path = tmp_path / "single_cells_annotated.h5ad"
        assert annotated_path.exists()

        annotated = ad.read_h5ad(annotated_path)
        assert "higher_layer" in annotated.obs.columns
        assert annotated.obs["higher_layer"].tolist() == ["x", "y", "x"]


class TestAnndataUmapCommand:
    """
    Tests for the interactive AnnData UMAP command.
    """

    def test_create_anndata_umap_plot_uses_obs_column_labels(
        self, monkeypatch, tmp_path
    ) -> None:
        """
        Test that the command loads AnnData and forwards obs labels to the UMAP plot.
        """

        adata = create_sample_anndata()
        adata.obs["group"] = ["g1", "g2", "g1"]
        ad_path = tmp_path / "single_cells.h5ad"
        adata.write_h5ad(ad_path)

        captured = {}

        class MockFilePath:
            def __init__(self, value):
                self.value = value

            def execute(self):
                return self.value

        monkeypatch.setattr(
            anndata_command.inquirer,
            "filepath",
            lambda *args, **kwargs: MockFilePath(str(ad_path)),
        )

        class MockSelect:
            def execute(self):
                return "group"

        monkeypatch.setattr(
            anndata_command.inquirer,
            "select",
            lambda *args, **kwargs: MockSelect(),
        )
        monkeypatch.setattr(
            anndata_command.console, "print_exception", lambda *args, **kwargs: None
        )

        def capture_plot_anndata_umap(data, out_path, obs_col=None, title_suffix=""):
            captured["data"] = data
            captured["out_path"] = out_path
            captured["obs_col"] = obs_col
            captured["title_suffix"] = title_suffix

        monkeypatch.setattr(
            anndata_command, "plot_anndata_umap", capture_plot_anndata_umap
        )

        result = anndata_command.create_anndata_umap_plot()

        assert result == 0
        assert captured["data"].obs["group"].tolist() == ["g1", "g2", "g1"]
        assert captured["obs_col"] == "group"
        assert captured["title_suffix"] == " AnnData"
        assert captured["out_path"] == str(tmp_path / "single_cells_group_umap.png")


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


def create_ensembl_anndata(sparse: bool = False) -> ad.AnnData:
    """
    Create a small AnnData object with Ensembl IDs as var_names and a duplicate gene symbol.
    """

    x = np.array(
        [
            [1, 2, 3, 4],
            [5, 6, 7, 8],
        ],
        dtype=np.float32,
    )

    obs = pd.DataFrame({"cell_type": ["a", "b"]}, index=["cell_1", "cell_2"])
    var = pd.DataFrame(
        {
            "feature_name": ["g1", "g2", "g1", "g3"],
            "feature_length": [10, 20, 30, 40],
        },
        index=["ENSG1", "ENSG2", "ENSG3", "ENSG4"],
    )

    adata = ad.AnnData(X=sp.csr_matrix(x) if sparse else x, obs=obs, var=var)
    adata.layers["counts"] = x.copy()

    return adata


class TestSwitchVarHelpers:
    """
    Tests for switching the gene annotation of an AnnData object.
    """

    def test_duplicate_genes_are_summed(self) -> None:
        """
        Test that genes with the same new identifier are summed and the old identifiers are kept.
        """

        switched = switch_adata_var_names(
            create_ensembl_anndata(), "feature_name", "feature_id"
        )

        assert list(switched.var_names) == ["g1", "g2", "g3"]
        assert switched.var.index.name == "feature_name"
        assert list(switched.var["feature_id"]) == ["ENSG1;ENSG3", "ENSG2", "ENSG4"]
        assert list(switched.var["feature_length"]) == [10, 20, 40]
        assert "feature_name" not in switched.var.columns
        np.testing.assert_array_equal(switched.X, np.array([[4, 2, 4], [12, 6, 8]]))
        np.testing.assert_array_equal(switched.layers["counts"], switched.X)

    def test_sparse_matrix_stays_sparse(self) -> None:
        """
        Test that a sparse expression matrix is not densified.
        """

        switched = switch_adata_var_names(
            create_ensembl_anndata(sparse=True), "feature_name", "feature_id"
        )

        assert sp.issparse(switched.X)
        np.testing.assert_array_equal(
            switched.X.toarray(), np.array([[4, 2, 4], [12, 6, 8]])
        )

    def test_switch_without_duplicates_and_back(self) -> None:
        """
        Test that switching without duplicates keeps all genes and can be reverted.
        """

        adata = create_ensembl_anndata()
        adata.var.index.name = "feature_id"
        adata.var["feature_name"] = ["g1", "g2", "g3", "g4"]
        adata.varm["loadings"] = np.ones((4, 2))

        switched = switch_adata_var_names(adata, "feature_name", "feature_id")
        reverted = switch_adata_var_names(switched, "feature_id", "feature_name")

        assert list(switched.var_names) == ["g1", "g2", "g3", "g4"]
        assert "loadings" in switched.varm
        assert list(reverted.var_names) == ["ENSG1", "ENSG2", "ENSG3", "ENSG4"]
        assert list(reverted.var["feature_name"]) == ["g1", "g2", "g3", "g4"]
        np.testing.assert_array_equal(reverted.X, adata.X)

    def test_missing_values_keep_old_identifier(self) -> None:
        """
        Test that missing values in the new column are replaced by the current identifier.
        """

        adata = create_ensembl_anndata()
        adata.var["feature_name"] = ["g1", None, "", "g3"]

        new_names = get_var_names_from_column(adata, "feature_name")

        assert list(new_names) == ["g1", "ENSG2", "ENSG3", "g3"]

    def test_invalid_columns(self) -> None:
        """
        Test that unknown columns and already existing columns raise an error.
        """

        adata = create_ensembl_anndata()

        with pytest.raises(ValueError):
            switch_adata_var_names(adata, "unknown", "feature_id")

        with pytest.raises(ValueError):
            switch_adata_var_names(adata, "feature_name", "feature_length")


class TestSwitchVarCommand:
    """
    Tests for the interactive switch-var command.
    """

    def setup_prompts(self, monkeypatch, ad_path, confirm: bool) -> dict:
        """
        Replace all prompts of the command with predefined answers.
        """

        captured = {}

        class MockPrompt:
            def __init__(self, value, **kwargs):
                self.value = value
                captured.update(kwargs)

            def execute(self):
                return self.value

        monkeypatch.setattr(
            anndata_command.inquirer,
            "filepath",
            lambda *args, **kwargs: MockPrompt(str(ad_path)),
        )
        monkeypatch.setattr(
            anndata_command.inquirer,
            "select",
            lambda *args, **kwargs: MockPrompt(
                "feature_name", choices=kwargs["choices"]
            ),
        )
        monkeypatch.setattr(
            anndata_command.inquirer,
            "text",
            lambda *args, **kwargs: MockPrompt(kwargs["default"]),
        )
        monkeypatch.setattr(
            anndata_command.inquirer,
            "confirm",
            lambda *args, **kwargs: MockPrompt(confirm, confirm_called=True),
        )
        monkeypatch.setattr(
            anndata_command.console, "print_exception", lambda *args, **kwargs: None
        )

        return captured

    def test_switch_var_merges_genes_after_confirmation(
        self, monkeypatch, tmp_path
    ) -> None:
        """
        Test that the command warns before merging genes and saves the switched file.
        """

        ad_path = tmp_path / "single_cells.h5ad"
        create_ensembl_anndata().write_h5ad(ad_path)

        captured = self.setup_prompts(monkeypatch, ad_path, confirm=True)

        result = anndata_command.switch_anndata_var()

        assert result == MSG_SUCCESS
        assert captured["confirm_called"]
        assert [choice.value for choice in captured["choices"]] == [
            "feature_length",
            "feature_name",
        ]
        assert "3 unique" in captured["choices"][1].name

        switched = ad.read_h5ad(tmp_path / "single_cells_feature_name.h5ad")
        assert list(switched.var_names) == ["g1", "g2", "g3"]
        assert list(switched.var["previous_var_names"]) == [
            "ENSG1;ENSG3",
            "ENSG2",
            "ENSG4",
        ]

    def test_switch_var_aborts_without_confirmation(
        self, monkeypatch, tmp_path
    ) -> None:
        """
        Test that no file is written, if merging genes is not confirmed.
        """

        ad_path = tmp_path / "single_cells.h5ad"
        create_ensembl_anndata().write_h5ad(ad_path)

        self.setup_prompts(monkeypatch, ad_path, confirm=False)

        result = anndata_command.switch_anndata_var()

        assert result == MSG_USER_ABORT
        assert not (tmp_path / "single_cells_feature_name.h5ad").exists()

    def test_switch_var_without_duplicates_does_not_ask(
        self, monkeypatch, tmp_path
    ) -> None:
        """
        Test that no confirmation is necessary, if no genes have to be merged.
        """

        adata = create_ensembl_anndata()
        adata.var["feature_name"] = ["g1", "g2", "g3", "g4"]
        ad_path = tmp_path / "single_cells.h5ad"
        adata.write_h5ad(ad_path)

        captured = self.setup_prompts(monkeypatch, ad_path, confirm=False)

        result = anndata_command.switch_anndata_var()

        assert result == MSG_SUCCESS
        assert "confirm_called" not in captured
        assert (tmp_path / "single_cells_feature_name.h5ad").exists()


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


def create_qc_anndata(use_ensembl: bool) -> ad.AnnData:
    """
    Create an AnnData object with mitochondrial, ribosomal, haemoglobin and MALAT1 genes.
    """

    symbols = ["MT-CO1", "RPL3", "HBB", "MALAT1", "CD4", "CD8A"]
    ensembl = [
        "ENSG00000198804",
        "ENSG00000100316",
        "ENSG00000244734",
        MALAT1_ENSEMBL,
        "ENSG00000010610",
        "ENSG00000153563",
    ]

    # Scanpy QC metrics require at least 500 genes
    fillers = [f"filler_{i}" for i in range(500)]
    symbols += fillers
    ensembl += fillers

    rng = np.random.default_rng(0)
    x = rng.poisson(5, size=(40, len(symbols))).astype(np.float32)
    x[:5, 0] = 5000  # Cells with high mitochondrial expression
    x[5:8, 2] = 5000  # Cells with high haemoglobin expression

    obs = pd.DataFrame(
        {"cell_type": ["a", "b"] * 20}, index=[f"cell_{i}" for i in range(40)]
    )
    var = pd.DataFrame(index=ensembl if use_ensembl else symbols)

    return ad.AnnData(X=x, obs=obs, var=var)


class TestPreprocessGeneIdentifiers:
    """
    Tests for the single cell preprocessing with gene symbols and Ensembl IDs.
    """

    def test_ensembl_ids_are_contained_in_constants(self) -> None:
        """
        Test that the Ensembl ID constants contain the expected genes.
        """

        assert len(MITO_GENES_ENSEMBL) == 37
        assert "ENSG00000198804" in MITO_GENES_ENSEMBL  # MT-CO1
        assert "ENSG00000100316" in RIBO_GENES_ENSEMBL  # RPL3
        assert "ENSG00000244734" in HB_GENES_ENSEMBL  # HBB
        assert MALAT1_ENSEMBL == "ENSG00000251562"

    def test_symbols_and_ensembl_ids_give_same_result(self) -> None:
        """
        Test that the preprocessing filters the same cells and genes for both identifiers.
        """

        kwargs = {
            "min_cell": 1,
            "min_gene_count": 1,
            "max_gene_count": 10000,
            "filter_low_expressed_genes": False,
        }
        by_symbol = preprocess_anndata_file(
            create_qc_anndata(use_ensembl=False), "cell_type", **kwargs
        )
        by_ensembl = preprocess_anndata_file(
            create_qc_anndata(use_ensembl=True), "cell_type", **kwargs
        )

        assert list(by_symbol.obs_names) == list(by_ensembl.obs_names)
        assert not by_symbol.obs_names.isin([f"cell_{i}" for i in range(8)]).any()
        assert list(by_symbol.var_names[:3]) == ["MALAT1", "CD4", "CD8A"]
        assert list(by_ensembl.var_names[:3]) == [
            MALAT1_ENSEMBL,
            "ENSG00000010610",
            "ENSG00000153563",
        ]
        assert by_symbol.n_vars == by_ensembl.n_vars == 503
