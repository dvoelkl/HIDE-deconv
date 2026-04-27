"""
=====================================================
ViewModel functions for all CLI bulk commands
=====================================================
"""

from rich.console import Console
from pathlib import Path
import pandas as pd

from ..constants import (
    MSG_FAILURE,
    MSG_SUCCESS,
)

from InquirerPy import inquirer
from InquirerPy.base.control import Choice
from InquirerPy.validator import PathValidator

from rich.prompt import Confirm

from ..visualization import plot_pca, plot_umap
from ..utils import sample_ids_valid

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

console = Console()

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


def create_bulk_pca_plot() -> int:
    """
    Select a RNA-seq bulk and create a PCA plot.
    """
    ret = MSG_SUCCESS

    bulk_path = inquirer.filepath(
        message="Enter path to bulk RNA expression file:",
        default=str(Path.cwd()),
        mandatory=True,
        mandatory_message="A bulk RNA expression file (.csv) must be selected to continue.",
        validate=PathValidator(is_file=True, message="Input is not a file."),
    ).execute()

    try:
        with console.status(
            "[bold blue]Loading bulk RNA expression file...[/bold blue]", spinner="dots"
        ):
            bulk = pd.read_csv(bulk_path, index_col=0)
    except Exception:
        console.print_exception()
        return MSG_FAILURE

    fSelectSampleSheet = Confirm.ask(
        "Select additional sample sheet for annotating?", default=True
    )
    labels = None
    if fSelectSampleSheet:
        samplesheet_path = inquirer.filepath(
            message="Select sample sheet:",
            default=str(Path.cwd()),
            mandatory=True,
            mandatory_message="An sample sheet (.csv) must be selected to continue.",
            validate=PathValidator(is_file=True, message="Input is not a file."),
        ).execute()

        try:
            sample_sheet = pd.read_csv(samplesheet_path)

            available_sample_cols = sample_sheet.columns.to_list()

            # Select column to link sample sheet with deconvoluted results
            available_sample_cols = sample_sheet.columns.to_list()

            sample_id_col = inquirer.select(
                message="Select column that holds sample ids:",
                choices=available_sample_cols,
                default=available_sample_cols[0],
                height=5,
            ).execute()

            if sample_ids_valid(sample_sheet[sample_id_col], bulk.columns.to_list()):
                available_sample_cols.remove(sample_id_col)

                cohort_cols = [
                    Choice(
                        value=col,
                        name=f"{col} [Unique Cohorts: {len(sample_sheet[col].unique())}]",
                    )
                    for col in available_sample_cols
                    if len(sample_sheet[col].unique()) > 1
                ]

                cohort_col = inquirer.select(
                    message="Select column that will be used to annotate cohorts:",
                    choices=cohort_cols,
                    height=5,
                ).execute()

                ids = sample_sheet[sample_id_col]
                cohorts = sample_sheet[cohort_col]
                labels = labels = (
                    pd.Series(cohorts.values, index=ids).reindex(bulk.columns).to_list()
                )
        except Exception:
            console.print_exception()
            console.print("[red]Cannot open sample sheet.[/red]")
            console.print("[dim]Please provide a valid sample sheet.[/dim]")
            return MSG_FAILURE

    if labels is None:
        plot_pca(
            bulk,
            out_path=f"{Path(bulk_path).parent}/{Path(bulk_path).stem}" + "_pca.png",
            title_suffix=" RNA-seq",
        )
    else:
        plot_pca(
            bulk,
            out_path=f"{Path(bulk_path).parent}/{Path(bulk_path).stem}_{cohort_col}_pca.png",
            labeling=labels,
            group_name=cohort_col,
            title_suffix=" RNA-seq",
        )

    return ret


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


def create_bulk_umap_plot() -> int:
    """
    Select a RNA-seq bulk and create an UMAP plot.
    """

    ret = MSG_SUCCESS

    bulk_path = inquirer.filepath(
        message="Enter path to bulk RNA expression file:",
        default=str(Path.cwd()),
        mandatory=True,
        mandatory_message="A bulk RNA expression file (.csv) must be selected to continue.",
        validate=PathValidator(is_file=True, message="Input is not a file."),
    ).execute()

    try:
        with console.status(
            "[bold blue]Loading bulk RNA expression file...[/bold blue]", spinner="dots"
        ):
            bulk = pd.read_csv(bulk_path, index_col=0)
    except Exception:
        console.print_exception()
        return MSG_FAILURE

    fSelectSampleSheet = Confirm.ask(
        "Select additional sample sheet for annotating?", default=True
    )
    labels = None
    if fSelectSampleSheet:
        samplesheet_path = inquirer.filepath(
            message="Select sample sheet:",
            default=str(Path.cwd()),
            mandatory=True,
            mandatory_message="An sample sheet (.csv) must be selected to continue.",
            validate=PathValidator(is_file=True, message="Input is not a file."),
        ).execute()

        try:
            sample_sheet = pd.read_csv(samplesheet_path)

            available_sample_cols = sample_sheet.columns.to_list()

            # Select column to link sample sheet with deconvoluted results
            available_sample_cols = sample_sheet.columns.to_list()

            sample_id_col = inquirer.select(
                message="Select column that holds sample ids:",
                choices=available_sample_cols,
                default=available_sample_cols[0],
                height=5,
            ).execute()

            if sample_ids_valid(sample_sheet[sample_id_col], bulk.columns.to_list()):
                available_sample_cols.remove(sample_id_col)

                cohort_cols = [
                    Choice(
                        value=col,
                        name=f"{col} [Unique Cohorts: {len(sample_sheet[col].unique())}]",
                    )
                    for col in available_sample_cols
                    if len(sample_sheet[col].unique()) > 1
                ]

                cohort_col = inquirer.select(
                    message="Select column that will be used to annotate cohorts:",
                    choices=cohort_cols,
                    height=5,
                ).execute()

                ids = sample_sheet[sample_id_col]
                cohorts = sample_sheet[cohort_col]
                labels = labels = (
                    pd.Series(cohorts.values, index=ids).reindex(bulk.columns).to_list()
                )
        except Exception:
            console.print_exception()
            console.print("[red]Cannot open sample sheet.[/red]")
            console.print("[dim]Please provide a valid sample sheet.[/dim]")
            return MSG_FAILURE

    if labels is None:
        plot_umap(
            bulk,
            out_path=f"{Path(bulk_path).parent}/{Path(bulk_path).stem}" + "_umap.png",
            title_suffix=" RNA-seq",
        )
    else:
        plot_umap(
            bulk,
            out_path=f"{Path(bulk_path).parent}/{Path(bulk_path).stem}_{cohort_col}_umap.png",
            labeling=labels,
            group_name=cohort_col,
            title_suffix=" RNA-seq",
        )

    return ret


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
