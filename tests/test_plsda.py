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


def test_run_plsda_maps_samples_without_refitting(tmp_path) -> None:
    data = pd.DataFrame(
        [[10, 12, 14], [4, 5, 6], [8, 7, 9]],
        index=["gene_1", "gene_2", "gene_3"],
        columns=["sample_1", "sample_2", "sample_3"],
    )
    mapped = pd.DataFrame(
        [[11, 13], [4.5, 5.5], [8, 8.5]],
        index=["gene_1", "gene_2", "gene_3"],
        columns=["mapped_1", "mapped_2"],
    )
    sample_sheet = pd.DataFrame(
        {
            "SampleID": ["sample_1", "sample_2", "sample_3"],
            "Cohort": ["A", "B", "A"],
        }
    )

    scores = run_plsda(
        data,
        sample_sheet,
        "SampleID",
        "Cohort",
        tmp_path / "plsda_result",
        datasets_to_map=[mapped],
        labels_data_map=["mapped"],
    )

    assert list(scores.index) == [
        "sample_1",
        "sample_2",
        "sample_3",
        "mapped_1",
        "mapped_2",
    ]
    assert list(scores["Cohort"]) == ["A", "B", "A", "mapped", "mapped"]
