"""
Tests for PLS-DA analysis.
"""

import pandas as pd

from hide_deconv.statistic import run_plsda


def test_run_plsda_writes_expected_outputs(tmp_path) -> None:
    data = pd.DataFrame(
        [[10, 12, 14], [4, 5, 6], [8, 7, 9]],
        index=["gene_1", "gene_2", "gene_3"],
        columns=["sample_1", "sample_2", "sample_3"],
    )

    sample_sheet = pd.DataFrame(
        {
            "SampleID": ["sample_1", "sample_2", "sample_3"],
            "Cohort": ["A", "B", "A"],
        }
    )

    out_path = tmp_path / "plsda_result"

    scores = run_plsda(data, sample_sheet, "SampleID", "Cohort", out_path)

    assert list(scores.columns) == ["PLS1", "PLS2", "Cohort"]
    assert (tmp_path / "plsda_result.csv").exists()
    assert (tmp_path / "plsda_result.png").exists()
    assert (tmp_path / "plsda_result_vip.png").exists()
    assert (tmp_path / "plsda_result_loading.png").exists()
