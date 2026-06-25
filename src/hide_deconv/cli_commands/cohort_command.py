"""
=====================================================
ViewModel functions for cohort CLI commands
=====================================================
"""

from pathlib import Path

import pandas as pd
from InquirerPy import inquirer
from InquirerPy.validator import PathValidator
from rich.console import Console

from ..constants import MSG_FAILURE, MSG_SUCCESS
from ..utils.cohort_utils import (
    get_cohort_choices,
    combine_categorical_cohorts,
    combine_numerical_cohorts,
)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

console = Console()

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


def combine_cohorts(numerical: bool = False) -> int:
    """
    Combine cohort values in a sample sheet.
    """
    console.print("[bold blue]Cohort Merger[/bold blue]")

    try:
        samplesheet_path = inquirer.filepath(
            message="Select sample sheet:",
            default=str(Path.cwd()),
            mandatory=True,
            mandatory_message="A sample sheet (.csv) must be selected to continue.",
            validate=PathValidator(is_file=True, message="Input is not a file."),
        ).execute()

        sample_sheet = pd.read_csv(samplesheet_path)
        available_sample_cols = sample_sheet.columns.to_list()

        cohort_cols = get_cohort_choices(sample_sheet, available_sample_cols, numerical)

        if len(cohort_cols) == 0:
            console.print(
                "[red]No sample sheet column was found that could be used for combining.[/red]"
            )
            return MSG_FAILURE

        cohort_col = inquirer.select(
            message="Select column that should be combined:",
            choices=cohort_cols,
            height=5,
        ).execute()

        if numerical:
            method = inquirer.select(
                message="Split numerical values by mean, median or greater equal:",
                choices=["mean", "median", "greater equal"],
                default="median",
                height=5,
            ).execute()

            threshold = None
            if method == "greater equal":
                threshold = inquirer.number(
                    message="Enter numeric threshold for 'greater equal' split:",
                    float_allowed=True,
                    mandatory=True,
                    mandatory_message="A numeric threshold must be provided.",
                ).execute()

            new_col_name = inquirer.text(
                message="Enter name for the new sample sheet column:",
                mandatory=True,
            ).execute()

            if new_col_name in sample_sheet.columns:
                console.print(
                    f"[red]Column '{new_col_name}' already exists in the sample sheet.[/red]"
                )
                return MSG_FAILURE

            sample_sheet = combine_numerical_cohorts(
                sample_sheet, cohort_col, new_col_name, method, threshold
            )
        else:
            n_groups = inquirer.number(
                message="How many cohorts should be combined?",
                min_allowed=1,
                max_allowed=len(sample_sheet[cohort_col].dropna().unique()),
                default=2,
                mandatory=True,
                float_allowed=False,
                mandatory_message="A number greater than 1 must be entered.",
            ).execute()

            new_col_name = inquirer.text(
                message="Enter name for the new sample sheet column:",
                mandatory=True,
            ).execute()

            if new_col_name in sample_sheet.columns:
                console.print(
                    f"[red]Column '{new_col_name}' already exists in the sample sheet.[/red]"
                )
                return MSG_FAILURE

            sample_sheet = combine_categorical_cohorts(
                sample_sheet, cohort_col, new_col_name, int(n_groups)
            )

        out_path = Path(samplesheet_path).with_name(
            f"{Path(samplesheet_path).stem}_{new_col_name.replace(' ', '_')}.csv"
        )
        sample_sheet.to_csv(out_path, index=False)
        console.print(
            f"[bold green]Saved combined sample sheet to {out_path}[/bold green]"
        )
    except Exception:
        console.print_exception()
        console.print("[red]Cannot open sample sheet.[/red]")
        console.print("[dim]Please provide a valid sample sheet.[/dim]")
        return MSG_FAILURE

    return MSG_SUCCESS


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


def plot_km_cohort() -> int:
    """
    Perform survival analysis using Cox Proportional Hazards regression.
    Guides through selecting a sample sheet with clinical survival information.

    Parameters
    ----------
    hidedeconv_path : Path
        Path where project is located.

    Returns
    -------
    int
        MSG_SUCCESS if no exception occured. MSG_FAILURE if an exception occured.
    """
    console.print("[bold blue]Kaplan Meier Plotting[/bold blue]")
    ret = MSG_SUCCESS

    # Load samplesheet
    samplesheet_path = inquirer.filepath(
        message="Select sample sheet:",
        default=str(Path.cwd()),
        mandatory=True,
        mandatory_message="A sample sheet (.csv) must be selected.",
        validate=PathValidator(is_file=True, message="Input is not a file."),
    ).execute()

    try:
        sample_sheet = pd.read_csv(samplesheet_path)

        available_sample_cols = sample_sheet.columns.to_list()

        # Select survival time column
        time_col = inquirer.select(
            message="Select column with survival time:",
            choices=available_sample_cols,
            height=5,
        ).execute()

        # Select event column
        available_sample_cols_event = [
            col for col in available_sample_cols if col != time_col
        ]
        event_col = inquirer.select(
            message="Select column with event indicator (0=censored, 1=event):",
            choices=available_sample_cols_event,
            height=5,
        ).execute()

        # Select stratification
        available_sample_cols_cov = [
            col for col in available_sample_cols if col not in [time_col, event_col]
        ]

        stratification = inquirer.select(
            message="Select stratification column for Kaplan-Meier curves:",
            choices=available_sample_cols_cov,
            height=5,
        ).execute()

        if inquirer.confirm(
            message="Set maximum time interval for Kaplan-Meier curve?",
            default=False,
        ).execute():
            max_time = float(inquirer.number(
                "Enter maximum time point:",
                min_allowed=1,
                default=5,
                mandatory=True,
                mandatory_message="A number greater than 0 must be entered.",
            ).execute())
        else:
            max_time = -1.0

        out_path = Path(samplesheet_path).with_name(
            f"{Path(samplesheet_path).stem}_KM_{stratification.replace(' ', '_')}.png"
        )

        from ..visualization import plot_kaplan_meier_cohort

        with console.status(
            "[bold blue]Creating Kaplan-Meier plots...[/bold blue]",
            spinner="dots",
        ):
            plot_kaplan_meier_cohort(
                sample_sheet,
                stratification,
                time_col,
                event_col,
                out_path=str(out_path),
                max_time=max_time,
            )

            console.print(f"[green]Saved Kaplan Meier plot to {str(out_path)}[/green]")

    except Exception:
        console.print("[red]Cannot open sample sheet.[/red]")
        console.print("[dim]Please provide a valid sample sheet.[/dim]")
        console.print_exception()
        ret = MSG_FAILURE

    return ret
