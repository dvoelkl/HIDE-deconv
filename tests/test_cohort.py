"""
Tests for cohort combine command.
"""

import pandas as pd

from hide_deconv.cli_commands import cohort_command
from hide_deconv.constants import MSG_SUCCESS


class DummyPrompt:
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


def create_sample_sheet(tmp_path, file_name="sample_sheet.csv"):
    """
    Create a small sample sheet for testing.
    """

    sample_sheet_path = tmp_path / file_name
    sample_sheet = pd.DataFrame(
        {
            "SampleID": ["sample_1", "sample_2", "sample_3", "sample_4"],
            "Cohort": ["CR", "NR", "PD", "SD"],
            "Score": [1.0, 2.0, 3.0, 4.0],
        }
    )
    sample_sheet.to_csv(sample_sheet_path, index=False)

    return sample_sheet_path


class TestCombineCohorts:
    """
    Tests for combine_cohorts command logic.
    """

    def test_combine_cohorts_categorical_writes_updated_sample_sheet(
        self, monkeypatch, tmp_path
    ) -> None:
        """
        Test that categorical cohort combining writes the expected sample sheet.
        """

        sample_sheet_path = create_sample_sheet(tmp_path)

        monkeypatch.setattr(
            cohort_command.inquirer,
            "filepath",
            select_sequence([str(sample_sheet_path)]),
        )
        monkeypatch.setattr(
            cohort_command.inquirer,
            "select",
            select_sequence(["Cohort"]),
        )
        monkeypatch.setattr(cohort_command.inquirer, "number", select_sequence([2]))
        monkeypatch.setattr(
            cohort_command.inquirer,
            "text",
            select_sequence(["Response", "Responder", "NonResponder"]),
        )
        monkeypatch.setattr(
            cohort_command.inquirer,
            "checkbox",
            select_sequence([["CR", "NR"], ["PD", "SD"]]),
        )

        result = cohort_command.combine_cohorts(numerical=False)

        assert result == MSG_SUCCESS
        out_path = tmp_path / "sample_sheet_Response.csv"
        combined = pd.read_csv(out_path)
        assert combined["Response"].tolist() == [
            "Responder",
            "Responder",
            "NonResponder",
            "NonResponder",
        ]

    def test_combine_cohorts_numerical_writes_high_low_sheet(
        self, monkeypatch, tmp_path
    ) -> None:
        """
        Test that numerical cohort combining writes a high/low sample sheet.
        """

        sample_sheet_path = create_sample_sheet(tmp_path)

        monkeypatch.setattr(
            cohort_command.inquirer,
            "filepath",
            select_sequence([str(sample_sheet_path)]),
        )
        monkeypatch.setattr(
            cohort_command.inquirer,
            "select",
            select_sequence(["Score", "mean"]),
        )
        monkeypatch.setattr(
            cohort_command.inquirer,
            "text",
            select_sequence(["ScoreGroup"]),
        )

        result = cohort_command.combine_cohorts(numerical=True)

        assert result == MSG_SUCCESS
        out_path = tmp_path / "sample_sheet_ScoreGroup.csv"
        combined = pd.read_csv(out_path)
        assert combined["ScoreGroup"].tolist() == ["low", "low", "high", "high"]
