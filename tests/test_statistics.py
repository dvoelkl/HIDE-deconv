"""
=====================================================
Tests for statistic module
=====================================================
"""

import pytest
import pandas as pd
import numpy as np

from hide_deconv.statistic import (
    run_kruskal_wallis,
    run_mann_whitney_u,
    run_dunn as run_posthoc_dunn,
    run_cox_regression,
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
        }
        assert (dunn_results["celltype"] == "ct_shifted").all()
        assert np.all((dunn_results["p_adj"] >= 0) & (dunn_results["p_adj"] <= 1))


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
