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
            data,
            out_path,
            labeling=None,
            group_name="Cohorts",
            title_suffix="",
            **kwargs,
        ):
            captured["data"] = data
            captured["out_path"] = out_path
            captured["labeling"] = labeling
            captured["group_name"] = group_name
            captured["kwargs"] = kwargs

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
        assert captured["kwargs"]["biplot"] is True

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
            data,
            out_path,
            labeling=None,
            group_name="Cohorts",
            title_suffix="",
            **kwargs,
        ):
            captured["data"] = data
            captured["out_path"] = out_path
            captured["labeling"] = labeling
            captured["group_name"] = group_name
            captured["kwargs"] = kwargs

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
        assert captured["kwargs"]["biplot"] is True

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


class TestSubsetBulk:
    """
    Tests for subset_bulk command logic.
    """

    def test_subset_bulk_writes_subsetted_csv(self, monkeypatch, tmp_path) -> None:
        """
        Test that subset_bulk keeps selected sample columns and writes the result.
        """

        bulk_path, bulk = create_bulk_file(tmp_path)
        sample_sheet_path = create_sample_sheet(tmp_path)
        captured = {}

        def capture_checkbox(*args, **kwargs):
            class MockCheckbox:
                def execute(self):
                    return ["B"]

            captured["checkbox_kwargs"] = kwargs
            return MockCheckbox()

        monkeypatch.setattr(
            bulk_command.inquirer,
            "filepath",
            select_sequence([str(bulk_path), str(sample_sheet_path)]),
        )
        monkeypatch.setattr(
            bulk_command.inquirer,
            "select",
            select_sequence(["SampleID", "Cohort"]),
        )
        monkeypatch.setattr(bulk_command.inquirer, "checkbox", capture_checkbox)

        result = bulk_command.subset_bulk()

        assert result == MSG_SUCCESS
        subset_path = tmp_path / "bulk_Cohort_subset.csv"
        assert subset_path.exists()
        subset = pd.read_csv(subset_path, index_col=0)
        assert subset.equals(bulk[["sample_2"]])
        assert captured["checkbox_kwargs"]["choices"] == ["A", "B"]

    def test_subset_bulk_handles_non_string_cohort_values(
        self, monkeypatch, tmp_path
    ) -> None:
        """
        Test that subset_bulk filters cohort values even when they are not strings.
        """

        bulk_path, bulk = create_bulk_file(tmp_path)
        sample_sheet_path = tmp_path / "sample_sheet_numeric.csv"
        sample_sheet = pd.DataFrame(
            {
                "SampleID": ["sample_1", "sample_2", "sample_3"],
                "Cohort": [1, 2, 1],
                "Batch": ["X", "X", "Y"],
            }
        )
        sample_sheet.to_csv(sample_sheet_path, index=False)

        def capture_checkbox(*args, **kwargs):
            class MockCheckbox:
                def execute(self):
                    return ["2"]

            return MockCheckbox()

        monkeypatch.setattr(
            bulk_command.inquirer,
            "filepath",
            select_sequence([str(bulk_path), str(sample_sheet_path)]),
        )
        monkeypatch.setattr(
            bulk_command.inquirer,
            "select",
            select_sequence(["SampleID", "Cohort"]),
        )
        monkeypatch.setattr(bulk_command.inquirer, "checkbox", capture_checkbox)

        result = bulk_command.subset_bulk()

        assert result == MSG_SUCCESS
        subset_path = tmp_path / "bulk_Cohort_subset.csv"
        subset = pd.read_csv(subset_path, index_col=0)
        assert subset.equals(bulk[["sample_2"]])

    def test_subset_bulk_returns_failure_on_invalid_bulk_file(
        self, monkeypatch, tmp_path
    ) -> None:
        """
        Test that subset_bulk returns MSG_FAILURE if bulk file cannot be read.
        """

        bulk_path = tmp_path / "missing_bulk.csv"

        monkeypatch.setattr(
            bulk_command.inquirer,
            "filepath",
            lambda **kwargs: prompt(str(bulk_path)),
        )
        monkeypatch.setattr(bulk_command, "console", bulk_command.console)

        result = bulk_command.subset_bulk()

        assert result == MSG_FAILURE


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


class TestMergeBulks:
    """
    Tests for merge_bulks command logic.
    """

    def test_merge_bulks(self, monkeypatch, tmp_path) -> None:
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

        batches_info = pd.DataFrame(
            ["batch_0", "batch_0", "batch_1", "batch_1"],
            index=["sample_1", "sample_2", "sample_3", "sample_4"],
            columns=["batch"],
        )

        captured = {}

        def capture_combine_bulk_dataframes(data_frames):
            captured["data_frames"] = data_frames
            return merged_bulk, batches_info

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
        assert pd.read_csv(merged_path, index_col=0).equals(merged_bulk)
        assert pd.read_csv(
            tmp_path / "merged_bulks_batch_info.csv", index_col=0
        ).equals(batches_info)


class TestCreateBulkDeg:
    """
    Tests for create_bulk_deg command logic.
    """

    def test_create_bulk_deg_filters_condition_columns_to_two_cohorts(
        self, monkeypatch, tmp_path
    ) -> None:
        """
        Test that only sample sheet columns with exactly two cohorts are shown.
        """

        bulk_path = tmp_path / "bulk.csv"
        bulk = pd.DataFrame(
            [[10, 11, 12], [5, 6, 7]],
            index=["gene_1", "gene_2"],
            columns=["sample_1", "sample_2", "sample_3"],
        )
        bulk.to_csv(bulk_path)

        sample_sheet_path = tmp_path / "sample_sheet.csv"
        sample_sheet = pd.DataFrame(
            {
                "SampleID": ["sample_1", "sample_2", "sample_3"],
                "Condition": ["A", "A", "B"],
                "ThreeCohorts": ["X", "Y", "Z"],
                "OneCohort": ["K", "K", "K"],
            }
        )
        sample_sheet.to_csv(sample_sheet_path, index=False)

        captured = {}

        def mock_select(*args, **kwargs):
            captured.setdefault("select_calls", []).append(kwargs)

            class MockSelect:
                def execute(self_inner):
                    if len(captured["select_calls"]) == 1:
                        return "SampleID"
                    if len(captured["select_calls"]) == 2:
                        return "Condition"
                    if len(captured["select_calls"]) == 3:
                        return "A"
                    return ""

            return MockSelect()

        monkeypatch.setattr(
            bulk_command.inquirer,
            "filepath",
            select_sequence([str(bulk_path), str(sample_sheet_path)]),
        )
        monkeypatch.setattr(bulk_command.inquirer, "select", mock_select)
        monkeypatch.setattr(bulk_command.Confirm, "ask", lambda *args, **kwargs: False)
        monkeypatch.setattr(
            bulk_command,
            "pydeseq2_preprocess",
            lambda *args, **kwargs: (bulk.T, sample_sheet.set_index("SampleID")),
        )
        monkeypatch.setattr(
            bulk_command,
            "run_pydeseq2",
            lambda *args, **kwargs: pd.DataFrame({"padj": [0.01]}, index=["gene_1"]),
        )

        result = bulk_command.create_bulk_deg()

        assert result == MSG_SUCCESS
        condition_choices = [
            choice.value for choice in captured["select_calls"][1]["choices"]
        ]
        assert condition_choices == ["Condition"]

    def test_create_bulk_deg_returns_failure_for_non_raw_counts(
        self, monkeypatch, tmp_path
    ) -> None:
        """
        Test that create_bulk_deg aborts when the bulk file is not raw counts.
        """

        bulk_path = tmp_path / "bulk.csv"
        bulk = pd.DataFrame(
            [[10.5, 11.0], [5.0, 6.0]],
            index=["gene_1", "gene_2"],
            columns=["sample_1", "sample_2"],
        )
        bulk.to_csv(bulk_path)

        monkeypatch.setattr(
            bulk_command.inquirer,
            "filepath",
            lambda **kwargs: prompt(str(bulk_path)),
        )

        result = bulk_command.create_bulk_deg()

        assert result == MSG_FAILURE
