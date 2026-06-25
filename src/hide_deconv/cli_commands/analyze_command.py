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

from ..config import hidedeconv_config
from ..constants import MSG_SUCCESS, MSG_FAILURE
from ..utils import (
    get_deconvolution_results,
    sample_ids_valid,
    load_project_bulk,
    filter_sample_sheet,
)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

console = Console()

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


def create_hdiff_plot(hidedeconv_path: Path) -> int:
    """
    Create a hierarchical difference heatmap for two cohorts.

    Parameters
    ----------
    hidedeconv_path : Path
        Path, where project is located.

    Returns
    -------
    int
        MSG_SUCCESS if no exception occured. MSG_FAILURE if an exception occured.
    """
    from ..visualization import plot_hier_heat
    from ..statistic import run_mann_whitney_u

    console.print("[bold blue]Hierarchical Difference Heatmap[/bold blue]")

    available_projects = get_deconvolution_results(hidedeconv_path)

    if len(available_projects) == 0:
        console.print(
            f"[red]No deconvolved project available at {hidedeconv_path.expanduser()}[/red]"
        )
        return MSG_FAILURE

    selected_project = inquirer.select(
        message="Select result folder:",
        choices=available_projects,
        default=available_projects[0],
        mandatory=True,
    ).execute()

    try:
        hconf = hidedeconv_config.load(str(hidedeconv_path) + "/config.json")

        samplesheet_path = inquirer.filepath(
            message="Select sample sheet:",
            default=str(hidedeconv_path.expanduser()),
            mandatory=True,
            mandatory_message="A sample sheet (.csv) must be selected to continue analysis.",
            validate=PathValidator(is_file=True, message="Input is not a file."),
        ).execute()

        sample_sheet = pd.read_csv(samplesheet_path)
        available_sample_cols = sample_sheet.columns.to_list()

        sample_id_col = inquirer.select(
            message="Select column that holds sample ids:",
            choices=available_sample_cols,
            default=available_sample_cols[0],
            height=5,
        ).execute()

        bulk = pd.read_csv(
            Path(hidedeconv_path) / "results" / selected_project / "C_sub.csv",
            index_col=0,
        )

        if not sample_ids_valid(sample_sheet[sample_id_col], bulk.columns.to_list()):
            console.print(
                f"[red]Bulk sample ids are no subset of {sample_id_col} column of sample sheet.[/red]"
            )
            return MSG_FAILURE

        available_sample_cols.remove(sample_id_col)

        cohort_cols = []

        for col in available_sample_cols:
            n_unique = sample_sheet[col].dropna().nunique()

            if n_unique == 2:
                cohort_cols.append(
                    Choice(value=col, name=f"{col} [Unique Cohorts: {n_unique}]")
                )

        if len(cohort_cols) == 0:
            console.print(
                "[red]No sample sheet column was found with exactly two cohorts.[/red]"
            )
            return MSG_FAILURE

        cohort_col = inquirer.select(
            message="Select column that will be used to split in cohorts:",
            choices=cohort_cols,
            height=5,
        ).execute()

        _ids, sample_sheet = filter_sample_sheet(sample_sheet, sample_id_col)

        results_dir = Path(hidedeconv_path) / "results" / selected_project / "hdiff"
        results_dir.mkdir(parents=True, exist_ok=True)

        layer_names = ["sub"] + list(hconf.higher_ct_cols)

        projection_matrices = [
            pd.read_csv(Path(hidedeconv_path) / "data" / "A_sub.csv", index_col=0)
        ]

        for layer_name in layer_names[1:]:
            projection_matrices.append(
                pd.read_csv(
                    Path(hidedeconv_path) / "data" / f"A_{layer_name}.csv",
                    index_col=0,
                )
            )

        mwu_results = []
        cohort_tag = str(cohort_col).replace(" ", "_")

        with console.status(
            "[bold blue]Running Mann-Whitney-U tests...[/bold blue]",
            spinner="dots",
        ):
            for layer_name in layer_names:
                bulk = pd.read_csv(
                    Path(hidedeconv_path)
                    / "results"
                    / selected_project
                    / f"C_{layer_name}.csv",
                    index_col=0,
                )

                mwu_res = run_mann_whitney_u(
                    bulk, sample_sheet, sample_id_col, cohort_col
                )
                mwu_res.to_csv(results_dir / f"mwu_{layer_name}_{cohort_tag}.csv")
                mwu_results.append(mwu_res)

        mean_columns = [
            col for col in mwu_results[0].columns if col.startswith("mean[")
        ]

        cohort_1_name = mean_columns[0][5:-1]
        cohort_2_name = mean_columns[1][5:-1]

        plot_hier_heat(
            mwu_results[0],
            mwu_results[1:],
            layer_names,
            projection_matrices,
            cohort_1_name,
            cohort_2_name,
            results_dir / f"hdiff_{cohort_tag}.png",
        )

        console.print(
            f"[green]Saved hierarchical difference heatmap in {results_dir}[/green]"
        )
    except Exception:
        console.print_exception()
        console.print("[red]Cannot open sample sheet or project files.[/red]")
        console.print(
            "[dim]Please provide a valid sample sheet and deconvolution results.[/dim]"
        )
        return MSG_FAILURE

    return MSG_SUCCESS


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
    from ..statistic import (
        run_mann_whitney_u,
        run_dunn,
        run_kruskal_wallis,
        print_mwu_summary,
        print_dunn_summary,
    )

    console.print("[bold blue]Composition Difference Analysis[/bold blue]")

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
            sample_sheet = pd.read_csv(samplesheet_path)

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
                        name=f"{col} [Unique Cohorts: {len(sample_sheet[col].dropna().unique())}]",
                    )
                    for col in available_sample_cols
                    if len(sample_sheet[col].dropna().unique()) > 1
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
                if len(sample_sheet[cohort_col].dropna().unique()) == 2:
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
            console.print_exception()
            console.print("[red]Cannot open sample sheet.[/red]")
            console.print("[dim]Please provide a valid sample sheet.[/dim]")
    else:
        console.print(
            f"[red]No deconvolved project available at {hidedeconv_path.expanduser()}[/red]"
        )

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
    from ..visualization import plot_eval

    console.print("[bold blue]Benchmark Evaluation[/bold blue]")
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
    """
    Create a pca plot of deconvolved compositions. Guides through selecting a deconvolution project and sample sheet
    with clinical meta information.

    Parameters
    ----------
    hidedeconv_path : Path
        Path where project is located.

    Returns
    -------
    int
        MSG_SUCCESS if no exception occured. MSG_FAILURE if an exception occured.
    """
    from ..visualization import plot_pca

    console.print("[bold blue]Composition PCA Plotting[/bold blue]")

    ret = MSG_SUCCESS

    # Select deconvoluted data
    available_projects = get_deconvolution_results(hidedeconv_path)

    if len(available_projects) > 0:
        # Load project, ct_layer and bulk
        selected_project, selected_ct_layer, bulk = load_project_bulk(hidedeconv_path)

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

        samplesheet_path = inquirer.filepath(
            message="Select sample sheet:",
            default=str(hidedeconv_path.expanduser()),
            mandatory=True,
            mandatory_message="An sample sheet (.csv) must be selected to continue analysis.",
            validate=PathValidator(is_file=True, message="Input is not a file."),
        ).execute()

        try:
            sample_sheet = pd.read_csv(samplesheet_path)

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
                ids, sample_sheet = filter_sample_sheet(sample_sheet, sample_id_col)

                # Subset bulk to patients with corresponding metadata
                bulk = bulk[ids]

                cohorts = sample_sheet[cohort_col]
                labels = (
                    pd.Series(cohorts.values, index=ids).reindex(bulk.columns).to_list()
                )

                plot_pca(
                    bulk,
                    out_path=str(hidedeconv_path)
                    + f"/results/{selected_project}/{selected_ct_layer}/pca_{selected_ct_layer}_{cohort_col}.png",
                    labeling=labels,
                    group_name=cohort_col,
                    title_suffix=" Composition",
                    biplot=True,
                )
            else:
                console.print(
                    f"[red]Bulk sample ids are no subset of {sample_id_col} column of sample sheet.[/red]"
                )

        except Exception:
            console.print_exception()
            console.print("[red]Cannot open sample sheet.[/red]")
            console.print("[dim]Please provide a valid sample sheet.[/dim]")

    return ret


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


def create_umap_plot(hidedeconv_path: Path) -> int:
    """
    Create a umap plot of deconvolved compositions. Guides through selecting a deconvolution project and sample sheet
    with clinical meta information.

    Parameters
    ----------
    hidedeconv_path : Path
        Path where project is located.

    Returns
    -------
    int
        MSG_SUCCESS if no exception occured. MSG_FAILURE if an exception occured.
    """
    from ..visualization import plot_umap

    console.print("[bold blue]Composition UMAP Plotting[/bold blue]")
    ret = MSG_SUCCESS

    # Select deconvoluted data
    available_projects = get_deconvolution_results(hidedeconv_path)

    if len(available_projects) > 0:
        # Load project, ct_layer and bulk
        selected_project, selected_ct_layer, bulk = load_project_bulk(hidedeconv_path)

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

        samplesheet_path = inquirer.filepath(
            message="Select sample sheet:",
            default=str(hidedeconv_path.expanduser()),
            mandatory=True,
            mandatory_message="An sample sheet (.csv) must be selected to continue analysis.",
            validate=PathValidator(is_file=True, message="Input is not a file."),
        ).execute()

        try:
            sample_sheet = pd.read_csv(samplesheet_path)

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

                ids, sample_sheet = filter_sample_sheet(sample_sheet, sample_id_col)

                cohorts = sample_sheet[cohort_col]

                # Subset bulk to patients with corresponding metadata
                bulk = bulk[ids]

                labels = (
                    pd.Series(cohorts.values, index=ids).reindex(bulk.columns).to_list()
                )

                plot_umap(
                    bulk,
                    out_path=str(hidedeconv_path)
                    + f"/results/{selected_project}/{selected_ct_layer}/umap_{selected_ct_layer}_{cohort_col}.png",
                    labeling=labels,
                    group_name=cohort_col,
                    title_suffix=" Composition",
                )
            else:
                console.print(
                    f"[red]Bulk sample ids are no subset of {sample_id_col} column of sample sheet.[/red]"
                )

        except Exception:
            console.print_exception()
            console.print("[red]Cannot open sample sheet.[/red]")
            console.print("[dim]Please provide a valid sample sheet.[/dim]")

    return ret


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


def create_plsda_plot(hidedeconv_path: Path) -> int:
    """
    Create a PLS-DA plot of estimated compositions.
    """
    from ..statistic import run_plsda

    console.print("[bold blue]Composition PLS-DA Plotting[/bold blue]")

    ret = MSG_SUCCESS

    available_projects = get_deconvolution_results(hidedeconv_path)

    if len(available_projects) > 0:
        selected_project, selected_ct_layer, bulk = load_project_bulk(hidedeconv_path)

        samplesheet_path = inquirer.filepath(
            message="Select sample sheet:",
            default=str(hidedeconv_path.expanduser()),
            mandatory=True,
            mandatory_message="An sample sheet (.csv) must be selected to continue analysis.",
            validate=PathValidator(is_file=True, message="Input is not a file."),
        ).execute()

        try:
            sample_sheet = pd.read_csv(samplesheet_path)

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

                ids, sample_sheet = filter_sample_sheet(sample_sheet, sample_id_col)

                # Filter bulks
                bulk = bulk[ids]

                out_dir = (
                    Path(hidedeconv_path)
                    / "results"
                    / selected_project
                    / selected_ct_layer
                )
                out_path = out_dir / f"plsda_{str(cohort_col).replace(' ', '_')}"

                with console.status(
                    "[bold blue]Running PLS-DA...[/bold blue]",
                    spinner="dots",
                ):
                    run_plsda(bulk, sample_sheet, sample_id_col, cohort_col, out_path)
            else:
                console.print(
                    f"[red]Bulk sample ids are no subset of {sample_id_col} column of sample sheet.[/red]"
                )

        except Exception:
            console.print_exception()
            console.print("[red]Cannot open sample sheet.[/red]")
            console.print("[dim]Please provide a valid sample sheet.[/dim]")

    return ret


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


def create_kmean_plot(hidedeconv_path: Path) -> int:
    """
    Create a k-means PCA plot of deconvolved compositions.
    """

    from ..visualization import plot_kmeans_pca

    console.print("[bold blue]K-means PCA Plotting[/bold blue]")

    ret = MSG_SUCCESS

    available_projects = get_deconvolution_results(hidedeconv_path)

    if len(available_projects) > 0:
        selected_project, selected_ct_layer, bulk = load_project_bulk(hidedeconv_path)

        results_dir = (
            Path(hidedeconv_path) / "results" / selected_project / selected_ct_layer
        )
        results_dir.mkdir(parents=True, exist_ok=True)

        n_clusters = inquirer.number(
            message="Enter number of clusters:",
            min_allowed=2,
            max_allowed=len(bulk.columns),
            default=min(3, len(bulk.columns)),
            mandatory=True,
            float_allowed=False,
            mandatory_message="A number greater than 1 must be entered.",
        ).execute()

        use_samplesheet = inquirer.confirm(
            message="Select additional sample sheet for cohort annotation?",
            default=False,
        ).execute()

        labels = []
        group_name = "Cluster"

        try:
            if use_samplesheet:
                samplesheet_path = inquirer.filepath(
                    message="Select sample sheet:",
                    default=str(hidedeconv_path.expanduser()),
                    mandatory=True,
                    mandatory_message="A sample sheet (.csv) must be selected.",
                    validate=PathValidator(
                        is_file=True, message="Input is not a file."
                    ),
                ).execute()

                sample_sheet = pd.read_csv(samplesheet_path)
                available_sample_cols = sample_sheet.columns.to_list()

                sample_id_col = inquirer.select(
                    message="Select column that holds sample ids:",
                    choices=available_sample_cols,
                    default=available_sample_cols[0],
                    height=5,
                ).execute()

                if not sample_ids_valid(
                    sample_sheet[sample_id_col], bulk.columns.to_list()
                ):
                    console.print(
                        f"[red]Bulk sample ids are no subset of {sample_id_col} column of sample sheet.[/red]"
                    )
                    return MSG_FAILURE

                available_sample_cols.remove(sample_id_col)
                cohort_cols = [
                    Choice(
                        value=col,
                        name=f"{col} [Unique Cohorts: {len(sample_sheet[col].dropna().unique())}]",
                    )
                    for col in available_sample_cols
                    if sample_sheet[col].dropna().nunique() > 1
                ]

                if len(cohort_cols) > 0:
                    cohort_col = inquirer.select(
                        message="Select column that will be used to split in cohorts:",
                        choices=cohort_cols,
                        height=5,
                    ).execute()

                    ids, sample_sheet = filter_sample_sheet(sample_sheet, sample_id_col)
                    cohorts = sample_sheet[cohort_col]

                    # Filter bulks to only hold samples with metainfo
                    bulk = bulk[ids]

                    labels = (
                        pd.Series(cohorts.values, index=ids)
                        .reindex(bulk.columns)
                        .to_list()
                    )
                    group_name = cohort_col

                else:
                    console.print(
                        "[red]No sample sheet column was found with more than one cohort.[/red]"
                    )

            cluster_ass = plot_kmeans_pca(
                bulk,
                out_path=str(results_dir / f"kmean_PCA_{selected_ct_layer}.png"),
                n_clusters=int(n_clusters),
                labeling=labels,
                group_name=group_name,
                title_suffix="",
                biplot=True,
            )

            cluster_ass.to_csv(results_dir / "kmean_cluster.csv", index=False)

        except Exception:
            console.print_exception()
            console.print("[red]Cannot open sample sheet.[/red]")
            console.print("[dim]Please provide a valid sample sheet.[/dim]")
            return MSG_FAILURE

    else:
        console.print(
            f"[red]No deconvolved project available at {hidedeconv_path.expanduser()}[/red]"
        )
        ret = MSG_FAILURE

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
    console.print("[bold blue]Surival Analysis[/bold blue]")
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
            sample_sheet = pd.read_csv(samplesheet_path)

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
                from ..visualization import plot_kaplan_meier_comp, plot_cox_forest

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
                            plot_kaplan_meier_comp(
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
        ret = MSG_FAILURE

    return ret


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


def cell_type_clustering(hidedeconv_path: Path) -> int:
    """
    Clusters the composition and saves both the assigned clusters in a sample sheet file and a annotated umap plotin the results folder.

    Parameters
    ----------
    hidedeconv_path : Path
        Path where project is located.

    Returns
    -------
    int
        MSG_SUCCESS if no exception occured. MSG_FAILURE if an exception occured.
    """
    from ..visualization import plot_pca, plot_umap
    from ..statistic import run_clustering

    console.print("[bold blue]Clustering[/bold blue]")
    ret = MSG_SUCCESS

    # Select deconvoluted data
    available_projects = get_deconvolution_results(hidedeconv_path)

    if len(available_projects) > 0:
        # Load project, ct_layer and bulk
        selected_project, selected_ct_layer, bulk = load_project_bulk(hidedeconv_path)

        cluster_ass = run_clustering(bulk)

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

        cluster_ass.to_csv(
            str(hidedeconv_path)
            + "/results/"
            + selected_project
            + "/"
            + selected_ct_layer
            + f"/clusters_{selected_ct_layer}.csv"
        )

        plot_pca(
            bulk,
            str(hidedeconv_path)
            + "/results/"
            + selected_project
            + "/"
            + selected_ct_layer
            + f"/clusters_PCA_{selected_ct_layer}.png",
            labeling=cluster_ass["assigned_cluster"].to_list(),
            group_name="cluster",
        )

        plot_umap(
            bulk,
            str(hidedeconv_path)
            + "/results/"
            + selected_project
            + "/"
            + selected_ct_layer
            + f"/clusters_UMAP_{selected_ct_layer}.png",
            labeling=cluster_ass["assigned_cluster"].to_list(),
            group_name="cluster",
        )
    else:
        console.print(
            f"[red]No deconvolved project available at {hidedeconv_path.expanduser()}[/red]"
        )
        ret = MSG_FAILURE

    return ret


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


def gene_markerplot(hidedeconv_path: Path) -> int:
    """
    Plot important genes as markermap.

    Parameters
    ----------
    hidedeconv_path : Path
        Path where project is located.

    Returns
    -------
    int
        MSG_SUCCESS if no exception occured. MSG_FAILURE if an exception occured.
    """

    from ..visualization import plot_genemap

    console.print("[bold blue]Marker Gene Plotting[/bold blue]")
    ret = MSG_SUCCESS

    # Theoretically only training has to be done, but more convenient to reuse already existing methods
    available_projects = get_deconvolution_results(hidedeconv_path)

    if len(available_projects) > 0:
        try:
            selected_project, selected_ct_layer, bulk = load_project_bulk(
                hidedeconv_path
            )

            hconf = hidedeconv_config.load(str(hidedeconv_path) + "/config.json")
            n_genes = hconf.n_genes

            n_genes_to_display = inquirer.number(
                message="Choose number of genes to be displayed.",
                min_allowed=1,
                max_allowed=n_genes,
                default=50,
                mandatory=True,
                float_allowed=False,
                mandatory_message="A number greater than 1 must be entered.",
            ).execute()

            # Load selected gene weights and reference profile
            gene_weights = pd.read_csv(
                str(hidedeconv_path)
                + "/processed/"
                + "/g_"
                + selected_ct_layer
                + ".csv",
                index_col=0,
            )

            X = pd.read_csv(
                str(hidedeconv_path) + "/data/" + "/X_" + selected_ct_layer + ".csv",
                index_col=0,
            )

            relevant_genes_ordered = gene_weights.mul(
                X.var(axis=1), axis=0
            ).sort_values(by="0", ascending=False)

            outpath = Path(
                str(hidedeconv_path)
                + "/results/"
                + selected_project
                + "/"
                + selected_ct_layer
                + "/markermap.png"
            )

            plot_genemap(
                X,
                relevant_genes_ordered.index[0 : int(n_genes_to_display)],
                f"Most relevant genes of {selected_ct_layer} layer",
                outpath,
            )

        except Exception:
            ret = MSG_FAILURE

    return ret
