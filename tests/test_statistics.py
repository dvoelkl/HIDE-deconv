"""
=====================================================
Tests for statistic module
=====================================================
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from hide_deconv.statistic import (
    run_kruskal_wallis,
    run_mann_whitney_u,
    run_dunn as run_posthoc_dunn,
    run_cox_regression,
    pydeseq2_preprocess,
    run_pydeseq2,
)


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


class TestKruskalWallis:
    """
    Tests for Kruskal Wallis based statistics.
    """

    def test_run_kruskal_wallis_returns_expected_structure(
        self, sample_bulks_three_cohorts, sample_sheet_three_cohorts
    ) -> None:
        """
        Test that run_kruskal_wallis returns p and adjusted p values per cell type.
        """

        result = run_kruskal_wallis(
            sample_bulks_three_cohorts,
            sample_sheet_three_cohorts,
            sample_id_col="SampleID",
            cohort_col="Cohort",
        )

        assert isinstance(result, pd.DataFrame)
        assert list(result.index) == list(sample_bulks_three_cohorts.index)
        assert "p" in result.columns
        assert "p_adj" in result.columns
        assert np.all((result["p"] >= 0) & (result["p"] <= 1))
        assert np.all((result["p_adj"] >= 0) & (result["p_adj"] <= 1))

    def test_run_kruskal_wallis_detects_difference(
        self, sample_bulks_three_cohorts, sample_sheet_three_cohorts
    ) -> None:
        """Test that Kruskal-Wallis finds ct_shifted differing across cohorts."""
        res = run_kruskal_wallis(
            sample_bulks_three_cohorts,
            sample_sheet_three_cohorts,
            sample_id_col="SampleID",
            cohort_col="Cohort",
        )
        assert res.loc["ct_shifted", "p_adj"] < 0.05


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


class TestMannWhitneyU:
    """
    Tests for Mann-Whitney-U based statistics.
    """

    def test_run_mann_whitney_u_returns_expected_statistics(
        self, sample_bulks_two_cohorts, sample_sheet_two_cohorts
    ) -> None:
        """
        Test that run_mann_whitney_u returns summary statistics and p-values.
        """

        result = run_mann_whitney_u(
            sample_bulks_two_cohorts,
            sample_sheet_two_cohorts,
            sample_id_col="SampleID",
            cohort_col="Cohort",
        )

        assert isinstance(result, pd.DataFrame)
        assert list(result.index) == list(sample_bulks_two_cohorts.index)

        expected_columns = {
            "mean[A]",
            "std[A]",
            "mean[B]",
            "std[B]",
            "p",
            "p_adj",
        }
        assert expected_columns.issubset(set(result.columns))
        assert np.all((result["p"] >= 0) & (result["p"] <= 1))
        assert np.all((result["p_adj"] >= 0) & (result["p_adj"] <= 1))

    def test_run_mann_whitney_u_raises_for_more_than_two_cohorts(
        self, sample_bulks_three_cohorts, sample_sheet_three_cohorts
    ) -> None:
        """
        Test that run_mann_whitney_u rejects sample sheets with more than two cohorts.
        """

        with pytest.raises(Exception):
            run_mann_whitney_u(
                sample_bulks_three_cohorts,
                sample_sheet_three_cohorts,
                sample_id_col="SampleID",
                cohort_col="Cohort",
            )

    def test_run_mann_whitney_u_detects_difference(
        self, sample_bulks_two_cohorts, sample_sheet_two_cohorts
    ) -> None:
        """
        Test that MWU finds ct_a_high differing between cohorts A and B.
        """
        res = run_mann_whitney_u(
            sample_bulks_two_cohorts,
            sample_sheet_two_cohorts,
            sample_id_col="SampleID",
            cohort_col="Cohort",
        )
        assert res.loc["ct_a_high", "p_adj"] < 0.05

    def test_run_dunn_finds_pairwise_differences_when_kruskal_significant(
        self, sample_bulks_three_cohorts, sample_sheet_three_cohorts
    ) -> None:
        """
        Test that pairwise significance for ct_shifted is found.
        """
        krus = run_kruskal_wallis(
            sample_bulks_three_cohorts,
            sample_sheet_three_cohorts,
            sample_id_col="SampleID",
            cohort_col="Cohort",
        )
        # ensure the fixture produces a significant signal for the target celltype
        assert krus.loc["ct_shifted", "p_adj"] < 0.05

        dunn = run_posthoc_dunn(
            krus,
            sample_bulks_three_cohorts,
            sample_sheet_three_cohorts,
            sample_id_col="SampleID",
            cohort_col="Cohort",
            sign_level=0.05,
        )

        assert isinstance(dunn, pd.DataFrame)
        assert (dunn["celltype"] == "ct_shifted").all()
        assert (dunn["p_adj"] < 0.05).any()


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


class TestPosthocDunn:
    """
    Unit tests for posthoc Dunn statistics.
    """

    def test_run_posthoc_dunn_returns_pairwise_results(
        self, sample_bulks_three_cohorts, sample_sheet_three_cohorts
    ) -> None:
        """
        Test that run_posthoc_dunn returns pairwise cohort comparisons.
        """

        kruskal_results = run_kruskal_wallis(
            sample_bulks_three_cohorts,
            sample_sheet_three_cohorts,
            sample_id_col="SampleID",
            cohort_col="Cohort",
        )

        kruskal_results.loc[:, "p_adj"] = 1.0
        kruskal_results.loc["ct_shifted", "p_adj"] = 0.001

        dunn_results = run_posthoc_dunn(
            kruskal_results,
            sample_bulks_three_cohorts,
            sample_sheet_three_cohorts,
            sample_id_col="SampleID",
            cohort_col="Cohort",
            sign_level=0.05,
        )

        assert isinstance(dunn_results, pd.DataFrame)
        assert len(dunn_results) == 3
        assert set(dunn_results.columns) == {
            "celltype",
            "cohort_1",
            "cohort_2",
            "p_adj",
            "mean[cohort_1]",
            "mean[cohort_2]",
        }
        assert (dunn_results["celltype"] == "ct_shifted").all()
        assert np.all((dunn_results["p_adj"] >= 0) & (dunn_results["p_adj"] <= 1))

        expected_means = {
            "A": 0.20,
            "B": 0.442,
            "C": 0.73,
        }

        for _, row in dunn_results.iterrows():
            assert row["mean[cohort_1]"] == pytest.approx(
                expected_means[row["cohort_1"]]
            )
            assert row["mean[cohort_2]"] == pytest.approx(
                expected_means[row["cohort_2"]]
            )


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


class TestCoxRegression:
    """
    Tests for Cox regression based survival analysis.
    """

    def test_run_cox_regression_with_covariates_returns_expected_structure(
        self, sample_survival_data
    ) -> None:
        """
        Test that run_cox_regression returns expected columns and finite model outputs.
        """

        bulks, sample_sheet = sample_survival_data

        result = run_cox_regression(
            bulks,
            sample_sheet,
            sample_id_col="SampleID",
            time_col="time",
            event_col="event",
            covariates=["age", "sex"],
        )

        assert isinstance(result, pd.DataFrame)
        assert set(result["celltype"]) == set(bulks.index)

        expected_columns = {
            "celltype",
            "coef",
            "hr",
            "ci_lower",
            "ci_upper",
            "p_value",
            "concordance_index",
            "p_value_adj",
        }
        assert expected_columns.issubset(set(result.columns))

        assert np.all(
            (result["p_value_adj"].dropna() >= 0)
            & (result["p_value_adj"].dropna() <= 1)
        )
        assert (
            result[["coef", "hr", "p_value", "concordance_index"]]
            .notna()
            .any(axis=None)
        )

    def test_run_cox_regression_without_covariates_runs(
        self, sample_survival_data
    ) -> None:
        """
        Test that run_cox_regression also works when no covariates are provided.
        """

        bulks, sample_sheet = sample_survival_data

        result = run_cox_regression(
            bulks,
            sample_sheet,
            sample_id_col="SampleID",
            time_col="time",
            event_col="event",
            covariates=None,
        )

        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(bulks.index)
        assert "p_value_adj" in result.columns


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


class TestPyDESeq2:
    """
    Tests for PyDESeq2 based differential expression analysis.
    """

    def test_pydeseq2_preprocess_aligns_counts_and_metadata(self) -> None:
        """
        Test that preprocessing transposes the bulk matrix and aligns metadata.
        """

        bulk = pd.DataFrame(
            [[10, 20, 30], [5, 15, 25]],
            index=["gene_1", "gene_2"],
            columns=["sample_3", "sample_1", "sample_2"],
        )
        sample_sheet = pd.DataFrame(
            {
                "SampleID": ["sample_1", "sample_2", "sample_3"],
                "Condition": ["A", "B", "A"],
                "Batch": ["X", "X", "Y"],
            }
        )

        counts, metadata = pydeseq2_preprocess(
            bulk,
            sample_sheet,
            sample_id_col="SampleID",
            condition_col="Condition",
            covariates=["Batch"],
        )

        assert list(counts.index) == ["sample_3", "sample_1", "sample_2"]
        assert list(counts.columns) == ["gene_1", "gene_2"]
        assert list(metadata.index) == ["sample_3", "sample_1", "sample_2"]
        assert metadata.loc["sample_1", "Condition"] == "A"
        assert metadata.loc["sample_2", "Batch"] == "X"

    def test_run_pydeseq2_writes_results_and_plots(self, monkeypatch, tmp_path) -> None:
        """
        Test that run_pydeseq2 stores the DEG outputs using the PyDESeq2 wrapper.
        """

        counts = pd.DataFrame(
            [[10, 11], [5, 6]],
            index=["sample_1", "sample_2"],
            columns=["gene_1", "gene_2"],
        )
        metadata = pd.DataFrame(
            {"Condition": ["A", "B"]}, index=["sample_1", "sample_2"]
        )

        class DummyDDS:
            def __init__(self, *args, **kwargs):
                self.plot_calls = []

            def deseq2(self):
                return None

            def plot_MA(self, save_path=None, **kwargs):
                self.plot_calls.append(save_path)
                Path(save_path).write_text("ma")

        class DummyStats:
            def __init__(self, dds, contrast, quiet=True):
                self.results_df = pd.DataFrame(
                    {
                        "baseMean": [10.0],
                        "log2FoldChange": [1.5],
                        "lfcSE": [0.2],
                        "stat": [2.0],
                        "pvalue": [0.01],
                        "padj": [0.02],
                    },
                    index=["gene_1"],
                )

            def summary(self):
                return None

            def plot_MA(self, save_path=None, **kwargs):
                Path(save_path).write_text("ma")

        monkeypatch.setattr("hide_deconv.statistic.pydeseq2.DeseqDataSet", DummyDDS)
        monkeypatch.setattr("hide_deconv.statistic.pydeseq2.DeseqStats", DummyStats)

        result = run_pydeseq2(
            counts,
            metadata,
            condition_col="Condition",
            tested_condition="B",
            reference_condition="A",
            covariates=None,
            out_path=tmp_path / "bulk_deg",
        )

        assert result.loc["gene_1", "padj"] == 0.02
        assert (tmp_path / "bulk_deg_results.csv").exists()
        assert (tmp_path / "bulk_deg_ma.png").exists()
        assert (tmp_path / "bulk_deg_volcano.png").exists()
