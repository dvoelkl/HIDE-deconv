"""
=====================================================
Tests for bulk module
=====================================================
"""

import pandas as pd

from hide_deconv.cli_commands import bulk_command
from hide_deconv.constants import MSG_FAILURE, MSG_SUCCESS


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


class DummyPrompt:
    """
    Simple helper to emulate InquirerPy prompt objects.
    """

    def __init__(self, value):
        self.value = value

    def execute(self):
        return self.value


def prompt(value):
    """
    Create prompt object for a fixed value.
    """

    return DummyPrompt(value)


def select_sequence(values):
    """
    Create a select mock that returns values in sequence.
    """

    values_iter = iter(values)

    def _mock_select(*args, **kwargs):
        return DummyPrompt(next(values_iter))

    return _mock_select


def create_bulk_file(tmp_path):
    """
    Create a small bulk expression file for testing.
    """

    bulk_path = tmp_path / "bulk.csv"

    bulk = pd.DataFrame(
        [[10, 20, 30], [5, 15, 25]],
        index=["gene_1", "gene_2"],
        columns=["sample_3", "sample_1", "sample_2"],
    )
    bulk.to_csv(bulk_path)

    return bulk_path, bulk


def create_sample_sheet(tmp_path, file_name="sample_sheet.csv"):
    """
    Create a small sample sheet for testing.
    """

    sample_sheet_path = tmp_path / file_name

    sample_sheet = pd.DataFrame(
        {
            "SampleID": ["sample_1", "sample_2", "sample_3"],
            "Cohort": ["A", "B", "A"],
            "Batch": ["X", "X", "Y"],
        }
    )
    sample_sheet.to_csv(sample_sheet_path, index=False)

    return sample_sheet_path


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


class TestCreateBulkPcaPlot:
    """
    Tests for create_bulk_pca_plot command logic.
    """

    def test_create_bulk_pca_plot_without_sample_sheet(
        self, monkeypatch, tmp_path
    ) -> None:
        """
        Test that create_bulk_pca_plot writes a PCA plot without annotation.
        """

        bulk_path, bulk = create_bulk_file(tmp_path)
        captured = {}

        def capture_plot_pca(
            data, out_path, labeling=None, group_name="Cohorts", title_suffix=""
        ):
            captured["data"] = data
            captured["out_path"] = out_path
            captured["labeling"] = labeling
            captured["group_name"] = group_name

        monkeypatch.setattr(
            bulk_command.inquirer,
            "filepath",
            lambda **kwargs: prompt(str(bulk_path)),
        )
        monkeypatch.setattr(bulk_command.Confirm, "ask", lambda *args, **kwargs: False)
        monkeypatch.setattr(bulk_command, "plot_pca", capture_plot_pca)

        result = bulk_command.create_bulk_pca_plot()

        assert result == MSG_SUCCESS
        assert captured["data"].equals(bulk)
        assert captured["labeling"] is None
        assert captured["group_name"] == "Cohorts"
        assert captured["out_path"] == f"{bulk_path.parent}/{bulk_path.stem}_pca.png"

    def test_create_bulk_pca_plot_with_sample_sheet(
        self, monkeypatch, tmp_path
    ) -> None:
        """
        Test that create_bulk_pca_plot annotates PCA plot with sample sheet labels.
        """

        bulk_path, bulk = create_bulk_file(tmp_path)
        sample_sheet_path = create_sample_sheet(tmp_path)
        captured = {}

        def capture_plot_pca(
            data, out_path, labeling=None, group_name="Cohorts", title_suffix=""
        ):
            captured["data"] = data
            captured["out_path"] = out_path
            captured["labeling"] = labeling
            captured["group_name"] = group_name

        monkeypatch.setattr(
            bulk_command.inquirer,
            "filepath",
            select_sequence([str(bulk_path), str(sample_sheet_path)]),
        )
        monkeypatch.setattr(bulk_command.Confirm, "ask", lambda *args, **kwargs: True)
        monkeypatch.setattr(
            bulk_command.inquirer,
            "select",
            select_sequence(["SampleID", "Cohort"]),
        )
        monkeypatch.setattr(bulk_command, "sample_ids_valid", lambda a, b: True)
        monkeypatch.setattr(bulk_command, "plot_pca", capture_plot_pca)

        result = bulk_command.create_bulk_pca_plot()

        assert result == MSG_SUCCESS
        assert captured["data"].equals(bulk)
        assert captured["labeling"] == ["A", "A", "B"]
        assert captured["group_name"] == "Cohort"
        assert (
            captured["out_path"]
            == f"{bulk_path.parent}/{bulk_path.stem}_Cohort_pca.png"
        )

    def test_create_bulk_pca_plot_returns_failure_on_invalid_bulk_file(
        self, monkeypatch, tmp_path
    ) -> None:
        """
        Test that create_bulk_pca_plot returns MSG_FAILURE if bulk file cannot be read.
        """

        bulk_path = tmp_path / "missing_bulk.csv"

        monkeypatch.setattr(
            bulk_command.inquirer,
            "filepath",
            lambda **kwargs: prompt(str(bulk_path)),
        )
        monkeypatch.setattr(bulk_command, "plot_pca", lambda *args, **kwargs: None)

        result = bulk_command.create_bulk_pca_plot()

        assert result == MSG_FAILURE


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


class TestCreateBulkUmapPlot:
    """
    Tests for create_bulk_umap_plot command logic.
    """

    def test_create_bulk_umap_plot_without_sample_sheet(
        self, monkeypatch, tmp_path
    ) -> None:
        """
        Test that create_bulk_umap_plot writes a UMAP plot without annotation.
        """

        bulk_path, bulk = create_bulk_file(tmp_path)
        captured = {}

        def capture_plot_umap(
            data, out_path, labeling=None, group_name="Cohorts", title_suffix=""
        ):
            captured["data"] = data
            captured["out_path"] = out_path
            captured["labeling"] = labeling
            captured["group_name"] = group_name

        monkeypatch.setattr(
            bulk_command.inquirer,
            "filepath",
            lambda **kwargs: prompt(str(bulk_path)),
        )
        monkeypatch.setattr(bulk_command.Confirm, "ask", lambda *args, **kwargs: False)
        monkeypatch.setattr(bulk_command, "plot_umap", capture_plot_umap)

        result = bulk_command.create_bulk_umap_plot()

        assert result == MSG_SUCCESS
        assert captured["data"].equals(bulk)
        assert captured["labeling"] is None
        assert captured["group_name"] == "Cohorts"
        assert captured["out_path"] == f"{bulk_path.parent}/{bulk_path.stem}_umap.png"

    def test_create_bulk_umap_plot_with_sample_sheet(
        self, monkeypatch, tmp_path
    ) -> None:
        """
        Test that create_bulk_umap_plot annotates UMAP plot with sample sheet labels.
        """

        bulk_path, bulk = create_bulk_file(tmp_path)
        sample_sheet_path = create_sample_sheet(
            tmp_path, file_name="sample_sheet_umap.csv"
        )
        captured = {}

        def capture_plot_umap(
            data, out_path, labeling=None, group_name="Cohorts", title_suffix=""
        ):
            captured["data"] = data
            captured["out_path"] = out_path
            captured["labeling"] = labeling
            captured["group_name"] = group_name

        monkeypatch.setattr(
            bulk_command.inquirer,
            "filepath",
            select_sequence([str(bulk_path), str(sample_sheet_path)]),
        )
        monkeypatch.setattr(bulk_command.Confirm, "ask", lambda *args, **kwargs: True)
        monkeypatch.setattr(
            bulk_command.inquirer,
            "select",
            select_sequence(["SampleID", "Cohort"]),
        )
        monkeypatch.setattr(bulk_command, "sample_ids_valid", lambda a, b: True)
        monkeypatch.setattr(bulk_command, "plot_umap", capture_plot_umap)

        result = bulk_command.create_bulk_umap_plot()

        assert result == MSG_SUCCESS
        assert captured["data"].equals(bulk)
        assert captured["labeling"] == ["A", "A", "B"]
        assert captured["group_name"] == "Cohort"
        assert (
            captured["out_path"]
            == f"{bulk_path.parent}/{bulk_path.stem}_Cohort_umap.png"
        )

    def test_create_bulk_umap_plot_returns_failure_on_invalid_sample_sheet(
        self, monkeypatch, tmp_path
    ) -> None:
        """
        Test that create_bulk_umap_plot returns MSG_FAILURE if sample sheet cannot be read.
        """

        bulk_path, _ = create_bulk_file(tmp_path)
        broken_sheet = tmp_path / "missing_sample_sheet.csv"

        monkeypatch.setattr(
            bulk_command.inquirer,
            "filepath",
            select_sequence([str(bulk_path), str(broken_sheet)]),
        )
        monkeypatch.setattr(bulk_command.Confirm, "ask", lambda *args, **kwargs: True)
        monkeypatch.setattr(bulk_command, "plot_umap", lambda *args, **kwargs: None)

        result = bulk_command.create_bulk_umap_plot()

        assert result == MSG_FAILURE


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


class TestMergeBulks:
    """
    Tests for merge_bulks command logic.
    """

    def test_merge_bulks_without_quality_report(self, monkeypatch, tmp_path) -> None:
        """
        Test that merge_bulks combines multiple bulk files and writes the merged result.
        """

        bulk_path_1 = tmp_path / "bulk_1.csv"
        bulk_path_2 = tmp_path / "bulk_2.csv"
        merged_path = tmp_path / "merged_bulks.csv"

        bulk_1 = pd.DataFrame(
            [[10, 20], [5, 15]],
            index=["gene_1", "gene_2"],
            columns=["sample_1", "sample_2"],
        )
        bulk_2 = pd.DataFrame(
            [[7, 9], [3, 8]],
            index=["gene_1", "gene_2"],
            columns=["sample_3", "sample_4"],
        )

        bulk_1.to_csv(bulk_path_1)
        bulk_2.to_csv(bulk_path_2)

        merged_bulk = pd.DataFrame(
            [[1, 2, 3, 4], [5, 6, 7, 8]],
            index=["gene_1", "gene_2"],
            columns=["sample_1", "sample_2", "sample_3", "sample_4"],
        )

        captured = {}

        def capture_combine_bulk_dataframes(data_frames, quality_control_path=""):
            captured["data_frames"] = data_frames
            captured["quality_control_path"] = quality_control_path
            return merged_bulk

        monkeypatch.setattr(
            bulk_command.inquirer,
            "filepath",
            select_sequence([str(bulk_path_1), str(bulk_path_2)]),
        )
        monkeypatch.setattr(
            bulk_command.inquirer,
            "text",
            select_sequence([str(merged_path)]),
        )
        confirm_values = iter([True, False, False])

        monkeypatch.setattr(
            bulk_command.Confirm,
            "ask",
            lambda *args, **kwargs: next(confirm_values),
        )
        monkeypatch.setattr(
            bulk_command,
            "combine_bulk_dataframes",
            capture_combine_bulk_dataframes,
        )

        result = bulk_command.merge_bulks()

        assert result == MSG_SUCCESS
        assert len(captured["data_frames"]) == 2
        assert captured["data_frames"][0].equals(bulk_1)
        assert captured["data_frames"][1].equals(bulk_2)
        assert captured["quality_control_path"] == ""
        assert pd.read_csv(merged_path, index_col=0).equals(merged_bulk)

    def test_merge_bulks_with_quality_report(self, monkeypatch, tmp_path) -> None:
        """
        Test that merge_bulks combines multiple bulk files, creates a quality report and writes the merged result.
        """

        bulk_path_1 = tmp_path / "bulk_1.csv"
        bulk_path_2 = tmp_path / "bulk_2.csv"
        merged_path = tmp_path / "merged_bulks.csv"
        quality_report_path = tmp_path / "qc_report.html"

        bulk_1 = pd.DataFrame(
            [[10, 20], [5, 15]],
            index=["gene_1", "gene_2"],
            columns=["sample_1", "sample_2"],
        )
        bulk_2 = pd.DataFrame(
            [[7, 9], [3, 8]],
            index=["gene_1", "gene_2"],
            columns=["sample_3", "sample_4"],
        )

        bulk_1.to_csv(bulk_path_1)
        bulk_2.to_csv(bulk_path_2)

        merged_bulk = pd.DataFrame(
            [[1, 2, 3, 4], [5, 6, 7, 8]],
            index=["gene_1", "gene_2"],
            columns=["sample_1", "sample_2", "sample_3", "sample_4"],
        )

        captured = {}

        def capture_combine_bulk_dataframes(data_frames, quality_control_path=""):
            captured["data_frames"] = data_frames
            captured["quality_control_path"] = quality_control_path
            return merged_bulk

        monkeypatch.setattr(
            bulk_command.inquirer,
            "filepath",
            select_sequence([str(bulk_path_1), str(bulk_path_2)]),
        )
        monkeypatch.setattr(
            bulk_command.inquirer,
            "text",
            select_sequence([str(quality_report_path), str(merged_path)]),
        )
        confirm_values = iter([True, False, True])

        monkeypatch.setattr(
            bulk_command.Confirm,
            "ask",
            lambda *args, **kwargs: next(confirm_values),
        )
        monkeypatch.setattr(
            bulk_command,
            "combine_bulk_dataframes",
            capture_combine_bulk_dataframes,
        )

        result = bulk_command.merge_bulks()

        assert result == MSG_SUCCESS
        assert len(captured["data_frames"]) == 2
        assert captured["data_frames"][0].equals(bulk_1)
        assert captured["data_frames"][1].equals(bulk_2)
        assert captured["quality_control_path"] == str(quality_report_path)
        assert pd.read_csv(merged_path, index_col=0).equals(merged_bulk)
