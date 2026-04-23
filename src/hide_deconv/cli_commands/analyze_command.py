"""
=====================================================
ViewModel functions for all CLI analyze commands
=====================================================
"""

import os
from rich.console import Console
from pathlib import Path
import pandas as pd

from InquirerPy import inquirer
from InquirerPy.base.control import Choice
from InquirerPy.validator import PathValidator

from ..constants import MSG_SUCCESS, MSG_FAILURE
from ..utils import get_deconvolution_results, sample_ids_valid, load_project_bulk
from ..statistic import (
    run_mann_whitney_u,
    print_mwu_summary,
    run_kruskal_wallis,
    run_dunn,
    print_dunn_summary,
)
from ..visualization import plot_eval, plot_pca

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

console = Console()

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


def analyze_differences(hidedeconv_path: Path) -> int:
    """
    Analyze differences between cohorts. Guides through selecting
    a deconvolution project and sample sheet with clinical information.

    Depending on number of cohorts automatically decides between
    Mann Whitney U or Kruskal Wallis test.

    Parameters
    ----------
    hidedeconv_path : Path
        Path, where project is located.

    Returns
    -------
    int
        MSG_SUCCESS if no exception occured. MSG_FAILURE if an exception occured.
    """

    ret = MSG_SUCCESS

    # Select deconvoluted data
    available_projects = get_deconvolution_results(hidedeconv_path)

    if len(available_projects) > 0:
        selected_project, selected_ct_layer, bulk = load_project_bulk(hidedeconv_path)

        # Load samplesheet
        samplesheet_path = inquirer.filepath(
            message="Select sample sheet:",
            default=str(hidedeconv_path.expanduser()),
            mandatory=True,
            mandatory_message="An sample sheet (.csv) must be selected to continue analysis.",
            validate=PathValidator(is_file=True, message="Input is not a file."),
        ).execute()

        try:
            sample_sheet = pd.read_csv(samplesheet_path, index_col=0)

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
                    message="Select column that will be used to split in cohorts:",
                    choices=cohort_cols,
                    height=5,
                ).execute()

                # Create subfolder in results for storing each cell type layer independently
                if not os.path.exists(
                    str(hidedeconv_path)
                    + "/results/"
                    + selected_project
                    + "/"
                    + selected_ct_layer
                ):
                    os.mkdir(
                        str(hidedeconv_path)
                        + "/results/"
                        + selected_project
                        + "/"
                        + selected_ct_layer
                    )

                # Decide wether Mann Whitney U or Kruskal-Wallis + Dunn Test
                if len(sample_sheet[cohort_col].unique()) == 2:
                    with console.status(
                        "[bold blue]Running Mann-Whitney-U Test...[/bold blue]",
                        spinner="dots",
                    ):
                        mwu_res = run_mann_whitney_u(
                            bulk, sample_sheet, sample_id_col, cohort_col
                        )
                        mwu_res.to_csv(
                            str(hidedeconv_path)
                            + "/results/"
                            + selected_project
                            + "/"
                            + selected_ct_layer
                            + f"/mwu_{str(cohort_col).replace(' ', '_')}.csv"
                        )

                    console.print(
                        f"[green]Saved Mann-Whitney-U test results in {
                            str(hidedeconv_path)
                            + '/results/'
                            + selected_project
                            + '/'
                            + selected_ct_layer
                            + f'/mwu_{str(cohort_col).replace(" ", "_")}.csv'
                        }[/green]"
                    )

                    print_mwu_summary(mwu_res)
                else:
                    console.print("Running Kruskal-Wallis Test...")
                    with console.status(
                        "[bold blue]Running Kruskal Wallis Test...[/bold blue]",
                        spinner="dots",
                    ):
                        krus_res = run_kruskal_wallis(
                            bulk, sample_sheet, sample_id_col, cohort_col
                        )

                    with console.status(
                        "[bold blue]Running Posthoc Dunn Test...[/bold blue]",
                        spinner="dots",
                    ):
                        dunn_res = run_dunn(
                            krus_res, bulk, sample_sheet, sample_id_col, cohort_col
                        )

                    krus_res.to_csv(
                        str(hidedeconv_path)
                        + "/results/"
                        + selected_project
                        + "/"
                        + selected_ct_layer
                        + f"/kruskal_{str(cohort_col).replace(' ', '_')}.csv"
                    )

                    dunn_res.to_csv(
                        str(hidedeconv_path)
                        + "/results/"
                        + selected_project
                        + "/"
                        + selected_ct_layer
                        + f"/posthoc_dunn_{str(cohort_col).replace(' ', '_')}.csv"
                    )

                    console.print(
                        f"[green]Saved Kruskal Wallis test results in {
                            str(hidedeconv_path)
                            + '/results/'
                            + selected_project
                            + '/'
                            + selected_ct_layer
                            + f'/kruskal_{str(cohort_col).replace(" ", "_")}.csv'
                        }[/green]"
                    )

                    console.print(
                        f"[green]Saved Posthoc Dunn test results in {
                            str(hidedeconv_path)
                            + '/results/'
                            + selected_project
                            + '/'
                            + selected_ct_layer
                            + f'/posthoc_dunn_{str(cohort_col).replace(" ", "_")}.csv'
                        }[/green]"
                    )

                    print_dunn_summary(dunn_res)
            else:
                console.print(
                    f"[red]Bulk sample ids are no subset of {sample_id_col} column of sample sheet.[/red]"
                )

        except Exception:
            console.print("[red]Cannot open sample sheet.[/red]")
            console.print("[dim]Please provide a valid sample sheet.[/dim]")
    else:
        console.print(
            f"[red]No deconvolved project available at {hidedeconv_path.expanduser()}[/red]"
        )
        # Hinweis auf HIDEOUT entfernt
        ret = MSG_FAILURE

    return ret


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


def benchmark_result(hidedeconv_path: Path) -> int:
    """
    Calculate various metrics between an estimated composition and its ground truth equivalent.
    Results are saved as figure and as table.

    Parameters
    ----------
    hidedeconv_path : Path
        Path, where project is located.

    Returns
    -------
    int
        MSG_SUCCESS if no exception occured. MSG_FAILURE if an exception occured.

    """

    ret = MSG_SUCCESS

    # Select deconvoluted data
    available_projects = get_deconvolution_results(hidedeconv_path)

    if len(available_projects) > 0:
        # Load project, ct_layer and bulk
        selected_project, selected_ct_layer, bulk = load_project_bulk(hidedeconv_path)

        # Load file to compare
        groundtruth_path = inquirer.filepath(
            message="Select ground truth composition:",
            default=str(hidedeconv_path.expanduser()),
            mandatory=True,
            mandatory_message="An table (.csv) must be selected to continue analysis.",
            validate=PathValidator(is_file=True, message="Input is not a file."),
        ).execute()

        try:
            C_true = pd.read_csv(groundtruth_path, index_col=0)

            results = plot_eval(
                C_true,
                bulk,
                out_path=str(hidedeconv_path)
                + f"/results/{selected_project}/benchmark_{selected_ct_layer}.png",
            )

            results.to_csv(
                str(hidedeconv_path)
                + f"/results/{selected_project}/benchmark_{selected_ct_layer}.csv"
            )

            console.print("[green]Saved benchmark results to[/green]")
            console.print(
                "[dim]"
                + f"/results/{selected_project}/benchmark_{selected_ct_layer}"
                + "[/dim]"
            )

        except KeyError:
            console.print(
                "[red]The cell type labels between the selected deconvolution results and the ground truth composition differ.[/red]"
            )
            console.print(
                "[dim]Please ensure that these are exactly at the same cell type layer.[/dim]"
            )
            ret = MSG_FAILURE
        except Exception:
            console.print_exception()
            ret = MSG_FAILURE

    return ret


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


def create_pca_plot(hidedeconv_path: Path) -> int:
    ret = MSG_SUCCESS

    # Select deconvoluted data
    available_projects = get_deconvolution_results(hidedeconv_path)

    if len(available_projects) > 0:
        # Load project, ct_layer and bulk
        selected_project, selected_ct_layer, bulk = load_project_bulk(hidedeconv_path)

        samplesheet_path = inquirer.filepath(
            message="Select sample sheet:",
            default=str(hidedeconv_path.expanduser()),
            mandatory=True,
            mandatory_message="An sample sheet (.csv) must be selected to continue analysis.",
            validate=PathValidator(is_file=True, message="Input is not a file."),
        ).execute()

        try:
            sample_sheet = pd.read_csv(samplesheet_path, index_col=0)

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
                    message="Select column that will be used to split in cohorts:",
                    choices=cohort_cols,
                    height=5,
                ).execute()

                ids = sample_sheet[sample_id_col]
                cohorts = sample_sheet[cohort_col]
                labels = (
                    pd.Series(cohorts.values, index=ids).reindex(bulk.columns).to_list()
                )

                plot_pca(
                    bulk,
                    out_path=str(hidedeconv_path)
                    + f"/results/{selected_project}/pca_{selected_ct_layer}_{cohort_col}.png",
                    labeling=labels,
                    group_name=cohort_col,
                )

        except Exception:
            pass

    return ret


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


def survival_analysis(hidedeconv_path: Path) -> int:
    """
    Perform survival analysis using Cox Proportional Hazards regression.
    Guides through selecting a deconvolution project and sample sheet with
    clinical survival information.

    Parameters
    ----------
    hidedeconv_path : Path
        Path where project is located.

    Returns
    -------
    int
        MSG_SUCCESS if no exception occured. MSG_FAILURE if an exception occured.
    """

    ret = MSG_SUCCESS

    available_projects = get_deconvolution_results(hidedeconv_path)

    if len(available_projects) > 0:
        selected_project, selected_ct_layer, bulk = load_project_bulk(hidedeconv_path)

        # Load samplesheet
        samplesheet_path = inquirer.filepath(
            message="Select sample sheet:",
            default=str(hidedeconv_path.expanduser()),
            mandatory=True,
            mandatory_message="A sample sheet (.csv) must be selected.",
            validate=PathValidator(is_file=True, message="Input is not a file."),
        ).execute()

        try:
            sample_sheet = pd.read_csv(samplesheet_path, index_col=0)

            available_sample_cols = sample_sheet.columns.to_list()

            sample_id_col = inquirer.select(
                message="Select column that holds sample ids:",
                choices=available_sample_cols,
                default=available_sample_cols[0],
                height=5,
            ).execute()

            if sample_ids_valid(sample_sheet[sample_id_col], bulk.columns.to_list()):
                available_sample_cols.remove(sample_id_col)

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

                # Select covariates
                available_sample_cols_cov = [
                    col
                    for col in available_sample_cols
                    if col not in [time_col, event_col]
                ]

                include_covariates = inquirer.confirm(
                    message="Include covariates in Cox model?",
                    default=False,
                ).execute()

                covariates = None
                if include_covariates and len(available_sample_cols_cov) > 0:
                    covariates = inquirer.checkbox(
                        message="Select covariates:",
                        choices=available_sample_cols_cov,
                        height=5,
                    ).execute()

                # Select stratification method
                stratification = inquirer.select(
                    message="Select stratification method for Kaplan-Meier curves:",
                    choices=["median", "tertiles", "quartiles"],
                    default="median",
                ).execute()

                if not os.path.exists(
                    str(hidedeconv_path)
                    + "/results/"
                    + selected_project
                    + "/"
                    + selected_ct_layer
                ):
                    os.mkdir(
                        str(hidedeconv_path)
                        + "/results/"
                        + selected_project
                        + "/"
                        + selected_ct_layer
                    )

                # Run Cox Regression
                from ..statistic import run_cox_regression, print_cox_summary
                from ..visualization import plot_kaplan_meier, plot_cox_forest

                with console.status(
                    "[bold blue]Running Cox Proportional Hazards Analysis...[/bold blue]",
                    spinner="dots",
                ):
                    cox_res = run_cox_regression(
                        bulk,
                        sample_sheet,
                        sample_id_col,
                        time_col,
                        event_col,
                        covariates,
                    )
                    cox_res.to_csv(
                        str(hidedeconv_path)
                        + "/results/"
                        + selected_project
                        + "/"
                        + selected_ct_layer
                        + f"/cox_regression_{selected_ct_layer}.csv"
                    )

                console.print(
                    f"[green]Saved Cox regression results in {
                        str(hidedeconv_path)
                        + '/results/'
                        + selected_project
                        + '/'
                        + selected_ct_layer
                        + f'/cox_regression_{selected_ct_layer}.csv'
                    }[/green]"
                )

                print_cox_summary(cox_res)

                # Kaplan Meier plots only for significant cell types
                significant_cts = cox_res[cox_res["p_value_adj"] < 0.05][
                    "celltype"
                ].tolist()

                if len(significant_cts) > 0:
                    with console.status(
                        "[bold blue]Creating Kaplan-Meier plots...[/bold blue]",
                        spinner="dots",
                    ):
                        for ct in significant_cts:
                            plot_kaplan_meier(
                                bulk,
                                sample_sheet,
                                sample_id_col,
                                time_col,
                                event_col,
                                ct,
                                stratification=stratification,
                                out_path=str(hidedeconv_path)
                                + "/results/"
                                + selected_project
                                + "/"
                                + selected_ct_layer
                                + f"/km_{ct.replace(' ', '_')}_{selected_ct_layer}.png",
                            )

                    console.print(
                        f"[green]Saved Kaplan Meier plots in folder {
                            str(hidedeconv_path)
                            + '/results/'
                            + selected_project
                            + '/'
                            + selected_ct_layer
                            + '/'
                        }[/green]"
                    )

                # Generate forest plot
                with console.status(
                    "[bold blue]Generating forest plot...[/bold blue]",
                    spinner="dots",
                ):
                    plot_cox_forest(
                        cox_res,
                        out_path=str(hidedeconv_path)
                        + "/results/"
                        + selected_project
                        + "/"
                        + selected_ct_layer
                        + f"/cox_forest_{selected_ct_layer}.png",
                    )

                console.print(
                    f"[green]Saved forest plot in {
                        str(hidedeconv_path)
                        + '/results/'
                        + selected_project
                        + '/'
                        + selected_ct_layer
                        + f'/cox_forest_{selected_ct_layer}.png'
                    }[/green]"
                )

            else:
                console.print(
                    f"[red]Bulk sample ids are no subset of {sample_id_col} column of sample sheet.[/red]"
                )
                ret = MSG_FAILURE

        except Exception:
            console.print("[red]Cannot open sample sheet.[/red]")
            console.print("[dim]Please provide a valid sample sheet.[/dim]")
            console.print_exception()
            ret = MSG_FAILURE
    else:
        console.print(
            f"[red]No deconvolved project available at {hidedeconv_path.expanduser()}[/red]"
        )
        # Hinweis auf HIDEOUT entfernt
        ret = MSG_FAILURE

    return ret
