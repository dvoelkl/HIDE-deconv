"""
=====================================================
Utility functions for cohort command group
=====================================================
"""

import pandas as pd
from InquirerPy import inquirer
from InquirerPy.base.control import Choice


def get_cohort_choices(
    sample_sheet: pd.DataFrame, available_columns: list[str], numerical: bool
) -> list[Choice]:
    """
    Select cohorts for selection list and display if how many unique cohorts exist if either interpreted as numerical or string value.
    """

    cohort_cols = []

    for col in available_columns:
        if numerical:
            values = pd.to_numeric(sample_sheet[col], errors="coerce")
            n_unique = values.dropna().nunique()
            label = f"{col} [Unique Cohorts: {n_unique}]"
        else:
            n_unique = sample_sheet[col].dropna().astype(str).nunique()
            label = f"{col} [Unique Cohorts: {n_unique}]"

        if n_unique > 1:
            cohort_cols.append(Choice(value=col, name=label))

    return cohort_cols


def combine_categorical_cohorts(
    sample_sheet: pd.DataFrame, cohort_col: str, new_col_name: str, n_groups: int
) -> pd.DataFrame:
    """
    Combines various categorical cohorts into new ones.
    """

    cohort_values = [str(value) for value in sample_sheet[cohort_col].dropna().unique()]
    remaining_values = cohort_values.copy()
    combined_values = pd.Series(pd.NA, index=sample_sheet.index, dtype="object")
    cohort_series = sample_sheet[cohort_col].astype(str)
    valid_rows = sample_sheet[cohort_col].notna()

    for group_idx in range(n_groups):
        group_name = inquirer.text(
            message=f"Enter name for cohort group {group_idx + 1}:",
            mandatory=True,
        ).execute()

        selected_values = inquirer.checkbox(
            message=f"Select cohort values for '{group_name}':",
            choices=remaining_values,
            max_height=5,
            mandatory=True,
            mandatory_message="Please select at least one entry.",
        ).execute()

        selected_values = [str(value) for value in selected_values]
        if len(selected_values) == 0:
            raise ValueError("Please select at least one entry.")

        combined_values.loc[valid_rows & cohort_series.isin(selected_values)] = (
            group_name
        )
        remaining_values = [
            value for value in remaining_values if value not in selected_values
        ]

        if group_idx < n_groups - 1 and len(remaining_values) == 0:
            raise ValueError(
                "Not enough cohort values left to create the requested groups."
            )

    sample_sheet[new_col_name] = combined_values
    return sample_sheet


def combine_numerical_cohorts(
    sample_sheet: pd.DataFrame,
    cohort_col: str,
    new_col_name: str,
    method: str,
    threshold: float | None = None,
) -> pd.DataFrame:
    """
    Combines numerical values in a high / low group split by either mean, median or greater equal
    """

    values = pd.to_numeric(sample_sheet[cohort_col], errors="coerce")
    valid_rows = values.notna()

    if valid_rows.sum() < 2:
        raise ValueError("The selected column needs at least two numerical values.")

    if method == "mean":
        calc_threshold = float(values.loc[valid_rows].mean())
    elif method == "median":
        calc_threshold = float(values.loc[valid_rows].median())
    elif method == "greater equal":
        calc_threshold = float(threshold)
    else:
        raise NotImplementedError("Method not implemented")

    combined_values = pd.Series(pd.NA, index=sample_sheet.index, dtype="object")
    combined_values.loc[valid_rows & (values >= calc_threshold)] = "high"
    combined_values.loc[valid_rows & (values < calc_threshold)] = "low"

    sample_sheet[new_col_name] = combined_values

    return sample_sheet
