"""
=====================================================
Tests for analyze module
=====================================================
"""

import pandas as pd

from hide_deconv.cli_commands import analyze_command
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


def create_mock_project_root(tmp_path):
    """
    Create minimal project folder layout used by analyze tests.
    """

    hidedeconv_path = tmp_path / "project"
    (hidedeconv_path / "results" / "proj").mkdir(parents=True)
    return hidedeconv_path


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


class TestAnalyzeDifferences:
    """
    Tests for analyze_differences command logic.
    """

    def test_analyze_differences_returns_failure_without_projects(
        self, monkeypatch, tmp_path
    ) -> None:
        """
        Test that analyze_differences returns MSG_FAILURE if no projects exist.
        """

        monkeypatch.setattr(
            analyze_command,
            "get_deconvolution_results",
            lambda hidedeconv_path: [],
        )

        result = analyze_command.analyze_differences(tmp_path)

        assert result == MSG_FAILURE

    def test_analyze_differences_runs_mwu_and_saves_result(
        self, monkeypatch, tmp_path
    ) -> None:
        """
        Test that analyze_differences runs MWU for two cohorts and writes output.
        """

        hidedeconv_path = create_mock_project_root(tmp_path)

        bulk = pd.DataFrame(
            [[0.2, 0.8], [0.8, 0.2]],
            index=["ct_a", "ct_b"],
            columns=["sample_1", "sample_2"],
        )

        sample_sheet = pd.DataFrame(
            {
                "SampleID": ["sample_1", "sample_2"],
                "Cohort": ["A", "B"],
            }
        )
        sample_sheet_path = hidedeconv_path / "sample_sheet.csv"
        sample_sheet.to_csv(sample_sheet_path, index=False)

        monkeypatch.setattr(
            analyze_command,
            "get_deconvolution_results",
            lambda path: ["proj"],
        )
        monkeypatch.setattr(
            analyze_command,
            "load_project_bulk",
            lambda path: ("proj", "sub", bulk),
        )
        monkeypatch.setattr(
            analyze_command.inquirer,
            "filepath",
            lambda **kwargs: prompt(str(sample_sheet_path)),
        )
        monkeypatch.setattr(
            analyze_command.inquirer,
            "select",
            select_sequence(["SampleID", "Cohort"]),
        )
        monkeypatch.setattr(analyze_command, "sample_ids_valid", lambda a, b: True)

        mwu_result = pd.DataFrame(
            {
                "mean[A]": [0.2, 0.8],
                "std[A]": [0.0, 0.0],
                "mean[B]": [0.8, 0.2],
                "std[B]": [0.0, 0.0],
                "p": [0.01, 0.01],
                "p_adj": [0.02, 0.02],
            },
            index=bulk.index,
        )

        monkeypatch.setattr(
            analyze_command, "run_mann_whitney_u", lambda *args: mwu_result
        )
        monkeypatch.setattr(analyze_command, "print_mwu_summary", lambda result: None)
        monkeypatch.setattr(
            analyze_command,
            "run_kruskal_wallis",
            lambda *args: (_ for _ in ()).throw(AssertionError("Unexpected call")),
        )

        result = analyze_command.analyze_differences(hidedeconv_path)

        expected_output = (
            hidedeconv_path / "results" / "proj" / "sub" / "mwu_Cohort.csv"
        )

        assert result == MSG_SUCCESS
        assert expected_output.exists()


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


class TestBenchmarkResult:
    """
    Tests for benchmark_result command logic.
    """

    def test_benchmark_result_saves_metrics_table(self, monkeypatch, tmp_path) -> None:
        """
        Test that benchmark_result stores returned benchmark metrics as csv.
        """

        hidedeconv_path = create_mock_project_root(tmp_path)

        bulk = pd.DataFrame(
            [[0.2, 0.8], [0.8, 0.2]],
            index=["ct_a", "ct_b"],
            columns=["sample_1", "sample_2"],
        )

        groundtruth = pd.DataFrame(
            [[0.3, 0.7], [0.7, 0.3]],
            index=["ct_a", "ct_b"],
            columns=["sample_1", "sample_2"],
        )
        groundtruth_path = hidedeconv_path / "groundtruth.csv"
        groundtruth.to_csv(groundtruth_path)

        benchmark_scores = pd.DataFrame({"pearson": [0.95]}, index=["overall"])

        monkeypatch.setattr(
            analyze_command, "get_deconvolution_results", lambda path: ["proj"]
        )
        monkeypatch.setattr(
            analyze_command,
            "load_project_bulk",
            lambda path: ("proj", "sub", bulk),
        )
        monkeypatch.setattr(
            analyze_command.inquirer,
            "filepath",
            lambda **kwargs: prompt(str(groundtruth_path)),
        )
        monkeypatch.setattr(
            analyze_command,
            "plot_eval",
            lambda C_true, bulk, out_path: benchmark_scores,
        )

        result = analyze_command.benchmark_result(hidedeconv_path)

        expected_output = hidedeconv_path / "results" / "proj" / "benchmark_sub.csv"

        assert result == MSG_SUCCESS
        assert expected_output.exists()

    def test_benchmark_result_returns_failure_on_mismatching_labels(
        self, monkeypatch, tmp_path
    ) -> None:
        """
        Test that benchmark_result returns MSG_FAILURE if plotting raises KeyError.
        """

        hidedeconv_path = create_mock_project_root(tmp_path)

        bulk = pd.DataFrame(
            [[0.2, 0.8]],
            index=["ct_a"],
            columns=["sample_1", "sample_2"],
        )

        groundtruth = pd.DataFrame(
            [[0.3, 0.7]],
            index=["ct_other"],
            columns=["sample_1", "sample_2"],
        )
        groundtruth_path = hidedeconv_path / "groundtruth.csv"
        groundtruth.to_csv(groundtruth_path)

        monkeypatch.setattr(
            analyze_command, "get_deconvolution_results", lambda path: ["proj"]
        )
        monkeypatch.setattr(
            analyze_command,
            "load_project_bulk",
            lambda path: ("proj", "sub", bulk),
        )
        monkeypatch.setattr(
            analyze_command.inquirer,
            "filepath",
            lambda **kwargs: prompt(str(groundtruth_path)),
        )

        def raise_key_error(*args, **kwargs):
            raise KeyError("mismatching labels")

        monkeypatch.setattr(analyze_command, "plot_eval", raise_key_error)

        result = analyze_command.benchmark_result(hidedeconv_path)

        assert result == MSG_FAILURE


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


class TestPcaAndUmap:
    """
    Tests for pca and umap helper commands.
    """

    def test_create_pca_plot_uses_reindexed_labels(self, monkeypatch, tmp_path) -> None:
        """
        Test that create_pca_plot aligns cohort labels to bulk column order.
        """

        hidedeconv_path = create_mock_project_root(tmp_path)

        bulk = pd.DataFrame(
            [[0.3, 0.7]],
            index=["ct_a"],
            columns=["sample_2", "sample_1"],
        )
        sample_sheet = pd.DataFrame(
            {
                "SampleID": ["sample_1", "sample_2"],
                "Cohort": ["A", "B"],
            }
        )
        sample_sheet_path = hidedeconv_path / "sample_sheet.csv"
        sample_sheet.to_csv(sample_sheet_path, index=False)

        captured = {}

        def capture_plot_pca(data, out_path, labeling, group_name):
            captured["labeling"] = labeling
            captured["group_name"] = group_name
            captured["out_path"] = out_path

        monkeypatch.setattr(
            analyze_command, "get_deconvolution_results", lambda path: ["proj"]
        )
        monkeypatch.setattr(
            analyze_command,
            "load_project_bulk",
            lambda path: ("proj", "sub", bulk),
        )
        monkeypatch.setattr(
            analyze_command.inquirer,
            "filepath",
            lambda **kwargs: prompt(str(sample_sheet_path)),
        )
        monkeypatch.setattr(
            analyze_command.inquirer,
            "select",
            select_sequence(["SampleID", "Cohort"]),
        )
        monkeypatch.setattr(analyze_command, "sample_ids_valid", lambda a, b: True)
        monkeypatch.setattr(analyze_command, "plot_pca", capture_plot_pca)

        result = analyze_command.create_pca_plot(hidedeconv_path)

        assert result == MSG_SUCCESS
        assert captured["labeling"] == ["B", "A"]
        assert captured["group_name"] == "Cohort"
        assert captured["out_path"].endswith("/results/proj/pca_sub_Cohort.png")

    def test_create_umap_plot_uses_reindexed_labels(
        self, monkeypatch, tmp_path
    ) -> None:
        """
        Test that create_umap_plot aligns cohort labels to bulk column order.
        """

        hidedeconv_path = create_mock_project_root(tmp_path)

        bulk = pd.DataFrame(
            [[0.3, 0.7]],
            index=["ct_a"],
            columns=["sample_2", "sample_1"],
        )
        sample_sheet = pd.DataFrame(
            {
                "SampleID": ["sample_1", "sample_2"],
                "Cohort": ["A", "B"],
            }
        )
        sample_sheet_path = hidedeconv_path / "sample_sheet.csv"
        sample_sheet.to_csv(sample_sheet_path, index=False)

        captured = {}

        def capture_plot_umap(data, out_path, labeling, group_name):
            captured["labeling"] = labeling
            captured["group_name"] = group_name
            captured["out_path"] = out_path

        monkeypatch.setattr(
            analyze_command, "get_deconvolution_results", lambda path: ["proj"]
        )
        monkeypatch.setattr(
            analyze_command,
            "load_project_bulk",
            lambda path: ("proj", "sub", bulk),
        )
        monkeypatch.setattr(
            analyze_command.inquirer,
            "filepath",
            lambda **kwargs: prompt(str(sample_sheet_path)),
        )
        monkeypatch.setattr(
            analyze_command.inquirer,
            "select",
            select_sequence(["SampleID", "Cohort"]),
        )
        monkeypatch.setattr(analyze_command, "sample_ids_valid", lambda a, b: True)
        monkeypatch.setattr(analyze_command, "plot_umap", capture_plot_umap)

        result = analyze_command.create_umap_plot(hidedeconv_path)

        assert result == MSG_SUCCESS
        assert captured["labeling"] == ["B", "A"]
        assert captured["group_name"] == "Cohort"
        assert captured["out_path"].endswith("/results/proj/umap_sub_Cohort.png")


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


class TestSurvivalAnalysis:
    """
    Tests for survival_analysis command logic.
    """

    def test_survival_analysis_returns_failure_without_projects(
        self, monkeypatch, tmp_path
    ) -> None:
        """
        Test that survival_analysis returns MSG_FAILURE if no projects exist.
        """

        monkeypatch.setattr(
            analyze_command,
            "get_deconvolution_results",
            lambda hidedeconv_path: [],
        )

        result = analyze_command.survival_analysis(tmp_path)

        assert result == MSG_FAILURE

    def test_survival_analysis_runs_and_writes_outputs(
        self, monkeypatch, tmp_path
    ) -> None:
        """
        Test that survival_analysis writes Cox results and creates plots.
        """

        hidedeconv_path = create_mock_project_root(tmp_path)

        bulk = pd.DataFrame(
            [[0.2, 0.8, 0.4], [0.8, 0.2, 0.6]],
            index=["ct_sig", "ct_bg"],
            columns=["sample_1", "sample_2", "sample_3"],
        )

        sample_sheet = pd.DataFrame(
            {
                "SampleID": ["sample_1", "sample_2", "sample_3"],
                "time": [5.0, 10.0, 8.0],
                "event": [1, 0, 1],
                "age": [60, 55, 62],
            }
        )
        sample_sheet_path = hidedeconv_path / "survival_sheet.csv"
        sample_sheet.to_csv(sample_sheet_path, index=False)

        cox_result = pd.DataFrame(
            {
                "celltype": ["ct_sig", "ct_bg"],
                "coef": [1.2, 0.1],
                "hr": [3.3, 1.1],
                "ci_lower": [1.2, 0.8],
                "ci_upper": [6.0, 1.5],
                "p_value": [0.001, 0.4],
                "concordance_index": [0.75, 0.75],
                "p_value_adj": [0.01, 0.4],
            }
        )

        km_calls = []
        forest_calls = []

        monkeypatch.setattr(
            analyze_command, "get_deconvolution_results", lambda path: ["proj"]
        )
        monkeypatch.setattr(
            analyze_command,
            "load_project_bulk",
            lambda path: ("proj", "sub", bulk),
        )
        monkeypatch.setattr(
            analyze_command.inquirer,
            "filepath",
            lambda **kwargs: prompt(str(sample_sheet_path)),
        )
        monkeypatch.setattr(
            analyze_command.inquirer,
            "select",
            select_sequence(["SampleID", "time", "event", "median"]),
        )
        monkeypatch.setattr(
            analyze_command.inquirer,
            "confirm",
            lambda **kwargs: prompt(False),
        )
        monkeypatch.setattr(analyze_command, "sample_ids_valid", lambda a, b: True)

        import hide_deconv.statistic as statistic_module
        import hide_deconv.visualization as visualization_module

        monkeypatch.setattr(
            statistic_module,
            "run_cox_regression",
            lambda *args, **kwargs: cox_result,
        )
        monkeypatch.setattr(statistic_module, "print_cox_summary", lambda result: None)

        def capture_km(*args, **kwargs):
            km_calls.append(kwargs["ct"] if "ct" in kwargs else args[5])

        def capture_forest(*args, **kwargs):
            forest_calls.append(kwargs["out_path"])

        monkeypatch.setattr(visualization_module, "plot_kaplan_meier", capture_km)
        monkeypatch.setattr(visualization_module, "plot_cox_forest", capture_forest)

        result = analyze_command.survival_analysis(hidedeconv_path)

        expected_output = (
            hidedeconv_path / "results" / "proj" / "sub" / "cox_regression_sub.csv"
        )

        assert result == MSG_SUCCESS
        assert expected_output.exists()
        assert km_calls == ["ct_sig"]
        assert len(forest_calls) == 1

    def test_survival_analysis_returns_failure_for_invalid_sample_ids(
        self, monkeypatch, tmp_path
    ) -> None:
        """
        Test that survival_analysis returns MSG_FAILURE for invalid sample ids.
        """

        hidedeconv_path = create_mock_project_root(tmp_path)

        bulk = pd.DataFrame(
            [[0.2, 0.8]],
            index=["ct_a"],
            columns=["sample_1", "sample_2"],
        )

        sample_sheet = pd.DataFrame(
            {
                "SampleID": ["other_1", "other_2"],
                "time": [5.0, 7.0],
                "event": [1, 1],
            }
        )
        sample_sheet_path = hidedeconv_path / "survival_sheet.csv"
        sample_sheet.to_csv(sample_sheet_path, index=False)

        monkeypatch.setattr(
            analyze_command, "get_deconvolution_results", lambda path: ["proj"]
        )
        monkeypatch.setattr(
            analyze_command,
            "load_project_bulk",
            lambda path: ("proj", "sub", bulk),
        )
        monkeypatch.setattr(
            analyze_command.inquirer,
            "filepath",
            lambda **kwargs: prompt(str(sample_sheet_path)),
        )
        monkeypatch.setattr(
            analyze_command.inquirer,
            "select",
            select_sequence(["SampleID"]),
        )
        monkeypatch.setattr(analyze_command, "sample_ids_valid", lambda a, b: False)

        result = analyze_command.survival_analysis(hidedeconv_path)

        assert result == MSG_FAILURE
