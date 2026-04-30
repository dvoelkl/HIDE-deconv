"""
=====================================================
Tests for AnnData command helpers and annotation flow
=====================================================
"""

import anndata as ad
import numpy as np
import pandas as pd

import hide_deconv.cli_commands.anndata_command as anndata_command
from hide_deconv.utils import (
    add_annotation_columns_from_template,
    create_annotation_template,
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
