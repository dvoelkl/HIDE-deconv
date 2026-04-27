"""
=====================================================
Commandline interface for executing pipelines
without coding
=====================================================
"""

from pathlib import Path
import click

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm

from .constants import MSG_SUCCESS, MSG_FAILURE, MSG_USER_ABORT
from .utils import assert_init, assert_preprocessed, assert_trained
from .cli_commands import (
    setup_config,
    show_config,
    preprocess,
    setup_project,
    train_model,
    show_help,
    deconvolve_hide,
    deconvolve_command,
    create_simulation,
    analyze_differences,
    benchmark_result,
    create_pca_plot,
    download_single_cells,
    preprocess_anndata,
    survival_analysis,
    create_umap_plot,
    create_bulk_pca_plot,
    create_bulk_umap_plot,
)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

console = Console()

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


@click.group(invoke_without_command=True)
@click.pass_context
def cli(ctx: click.Context) -> None:
    """
    HIDE-Deconv command line interface entry point.
    """

    if ctx.invoked_subcommand is None:
        console.print(
            Panel.fit(
                "Interactive command line tool and python package for hierarchical deconvolution and analysis of bulk RNA-seq data.\n\n"
                "---\n\n"
                "[bold]Features[/bold]\n\n"
                "- Designed for AnnData single cell datasets\n"
                "- Open Source package, that can be run on safe servers\n"
                "- Hierarchical cell type deconvolution for any number of cell type annotation layers\n"
                "- Includes methods for post-deconvolution analysis\n"
                "- Usable via command line interface and Python API\n"
                "- Provides a guided workflow that allows users without programming experience to perform deconvolution on their own\n\n"
                "[dim]Run [i]hide-deconv help[/i] to read the quickstart guide.[/dim]",
                title="[bold]HIDE-Deconv[/bold]",
            )
        )


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


@cli.group("config")
def cli_config() -> None:
    """
    Commands related to the HIDE-Deconv config.
    """
    pass


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


@cli.group("deconv", invoke_without_command=True)
@click.pass_context
@click.option(
    "--path",
    "-p",
    "hidedeconv_path",
    default=".",
    show_default=True,
    type=click.Path(
        exists=True, file_okay=False, dir_okay=True, writable=True, path_type=Path
    ),
    help="Path, where the HIDE-Deconv project structure is located.",
)
@click.option(
    "--bulk",
    "-b",
    "alternative_bulk",
    default=None,
    show_default=False,
    type=click.Path(exists=True, file_okay=True, dir_okay=False, writable=True),
    help="Filepath of bulk to deconvolve. If not given, the bulk set in configuration is used",
)
@assert_trained
def cli_deconvolve(ctx, hidedeconv_path: Path, alternative_bulk=None) -> None:
    """
    Commands related to deconvolution.
    """

    # Only open deconvolution menu if no other submodel is invoked
    if ctx.invoked_subcommand is None:
        deconvolve_command(hidedeconv_path, alternative_bulk)


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


@cli.group("analyze")
def cli_analyze() -> None:
    """
    Commands related to the analysis of the deconvoluted results.
    """
    pass


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


@cli.group("anndata")
def cli_anndata() -> None:
    """
    Commands related to the processing of anndata files.
    """
    pass


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


@cli.group("bulk")
def cli_bulk() -> None:
    """
    Commands related to the processing of bulk files.
    """
    pass


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


@cli.command("help")
def cli_help_command():
    """
    Display a quick start guide for HIDE-Deconv.
    """
    show_help()


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


@cli.command("run")
@click.option(
    "--path",
    "-p",
    "hidedeconv_path",
    default=".",
    show_default=True,
    type=click.Path(
        exists=True, file_okay=False, dir_okay=True, writable=True, path_type=Path
    ),
    help="Path, where the HIDE-Deconv project structure will be initialized.",
)
@click.option(
    "--domain_transfer",
    "-dt",
    "fDomTransfer",
    is_flag=False,
    default=True,
    show_default=True,
    help="Account for domain transfer between single cell and bulks",
)
def cli_run_command(hidedeconv_path: Path, fAdv=False, fDomTransfer=True) -> int:
    """
    Complete walkthrough of the standard deconvolution process.
    Executes all commands necessary from starting a project until the final deconvolution.

    Note: This is not an automation command, as constant user input is necessary.
    """

    hidedeconv_path = hidedeconv_path.expanduser().resolve()

    console.print(
        Panel.fit(
            f"[bold]HIDE-Deconv[/bold]\nProject Path: [cyan]{hidedeconv_path}[/cyan]",
            border_style="blue",
        )
    )

    # Setup folder structure
    ret = setup_project(hidedeconv_path, fAdv)
    if ret == MSG_FAILURE:
        console.print("[red]Run failed.[/red]")
        console.print_exception()
        return MSG_FAILURE
    elif ret == MSG_USER_ABORT:
        console.print("[red]Run aborted.[/red]")
        return MSG_FAILURE

    if Confirm.ask("Run preprocessing now?", default=True):
        preprocess(hidedeconv_path, fDomTransfer)
        console.print("[green]Preprocessing completed successfully.[/green]")

        if Confirm.ask("Train model now?", default=True):
            train_model(hidedeconv_path)
            console.print("[green]Model trained successfully.[/green]")
            if Confirm.ask(
                "Choose deconvolution model and deconvolve now?", default=True
            ):
                deconvolve_command(hidedeconv_path, None)
            else:
                console.print("[dim]Next step [i]hide-deconv deconv[/i].[/dim]")
        else:
            console.print("[dim]Next step [i]hide-deconv train[/i].[/dim]")
    else:
        console.print("[dim]Next step [i]hide-deconv preprocess[/i].[/dim]")

    return MSG_SUCCESS


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


@cli.command("init")
@click.option(
    "--path",
    "-p",
    "hidedeconv_path",
    default=".",
    show_default=True,
    type=click.Path(
        exists=True, file_okay=False, dir_okay=True, writable=True, path_type=Path
    ),
    help="Path, where the HIDE-deconv project structure will be initialized.",
)
def cli_init_command(hidedeconv_path: Path, fAdv=False, fDomTransfer=True) -> int:
    """
    Initialize the HIDE-Deconv project structure at a given path.
    """

    hidedeconv_path = hidedeconv_path.expanduser().resolve()

    # Setup folder structure
    if setup_project(hidedeconv_path, fAdv) != MSG_SUCCESS:
        console.print("[red]Setup failed[/red]")
        console.print_exception()
        return MSG_FAILURE

    return MSG_SUCCESS


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


@cli_config.command("edit")
@click.option(
    "--path",
    "-p",
    "hidedeconv_path",
    default=".",
    show_default=True,
    type=click.Path(
        exists=True, file_okay=False, dir_okay=True, writable=True, path_type=Path
    ),
    help="Path, where the HIDE-Deconv project structure is located.",
)
@assert_init
def cli_config_edit(hidedeconv_path: Path, fAdv: bool = False) -> None:
    """
    Edits the parameters necessary for preprocessing.
    """

    setup_config(hidedeconv_path, fAdv)


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


@cli.command("preprocess")
@click.option(
    "--path",
    "-p",
    "hidedeconv_path",
    default=".",
    show_default=True,
    type=click.Path(
        exists=True, file_okay=False, dir_okay=True, writable=True, path_type=Path
    ),
    help="Path, where the HIDE-Deconv project structure is located.",
)
@click.option(
    "--domain_transfer",
    "-dt",
    "fDomTransfer",
    is_flag=True,
    default=True,
    show_default=True,
    help="Account for domain transfer between single cell and bulks",
)
@assert_init
def cli_preprocess(hidedeconv_path: Path, fDomTransfer) -> None:
    """
    Run preprocessing for HIDE-Deconv by aligning genes between single-cell and bulk data,
    creating reference/hierarchy matrices, generating training bulks and optionally
    accounting for domain transfer.
    """

    preprocess(hidedeconv_path, fDomTransfer)
    console.print("[green]Preprocessing completed successfully[/green]")

    if Confirm.ask("Train model now?", default=True):
        train_model(hidedeconv_path)
        console.print("[green]Model trained successfully.[/green]")
    else:
        console.print("[dim]Next step [i]hide-deconv train[/i][/dim]")


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


@cli.command("train")
@click.option(
    "--path",
    "-p",
    "hidedeconv_path",
    default=".",
    show_default=True,
    type=click.Path(
        exists=True, file_okay=False, dir_okay=True, writable=True, path_type=Path
    ),
    help="Path, where the HIDE-Deconv project structure is located.",
)
@assert_preprocessed
def cli_train(hidedeconv_path: Path) -> None:
    """
    Train the model.
    """
    train_model(hidedeconv_path)

    console.print("[green]Model trained successfully.[/green]")


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


@cli_config.command("show")
@click.option(
    "--path",
    "-p",
    "hidedeconv_path",
    default=".",
    show_default=True,
    type=click.Path(
        exists=True, file_okay=False, dir_okay=True, writable=True, path_type=Path
    ),
    help="Path, where the HIDE-Deconv project structure is located.",
)
@assert_init
def cli_config_show(hidedeconv_path: Path) -> None:
    """
    Displays the current HIDE-Deconv configuration.
    """
    show_config(hidedeconv_path)


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


@cli_deconvolve.command("hide")
@click.option(
    "--path",
    "-p",
    "hidedeconv_path",
    default=".",
    show_default=True,
    type=click.Path(
        exists=True, file_okay=False, dir_okay=True, writable=True, path_type=Path
    ),
    help="Path, where the HIDE-Deconv project structure is located.",
)
@click.option(
    "--bulk",
    "-b",
    "alternative_bulk",
    default=None,
    show_default=False,
    type=click.Path(exists=True, file_okay=True, dir_okay=False, writable=True),
    help="Filepath of bulk to deconvolve. If not given, the bulk set in configuration is used",
)
@assert_trained
def cli_deconv_hide(hidedeconv_path: Path, alternative_bulk=None) -> None:
    """
    Runs the deconvolution with HIDE.
    """

    deconvolve_hide(hidedeconv_path, alternative_bulk)


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


@cli.command("simulate")
@click.option(
    "--ad_path",
    "-ap",
    "ad_path",
    default=None,
    show_default=False,
    type=click.Path(
        exists=True, file_okay=True, dir_okay=False, writable=True, path_type=Path
    ),
    help="Path to the AnnData file.",
)
@click.option(
    "--out",
    "-o",
    "out_path",
    default=None,
    show_default=False,
    type=click.Path(
        exists=True, file_okay=False, dir_okay=True, writable=True, path_type=Path
    ),
    help="Path where created anndata and bulk files will be stored.",
)
@click.option(
    "--train_frac",
    "-tf",
    "train_frac",
    default=None,
    show_default=False,
    type=click.FLOAT,
    help="Percentage of all cells used for training data.",
)
@click.option(
    "--n_bulks",
    "-nb",
    "n_bulks",
    default=None,
    show_default=False,
    type=click.IntRange(min=1),
    help="Number of bulks simulated for testing.",
)
@click.option(
    "--n_cells_bulk",
    "-cb",
    "n_cell_per_bulks",
    default=None,
    show_default=False,
    type=click.IntRange(min=1),
    help="Number of cells accumulated to a single in-silico bulk.",
)
def cli_simulate(
    ad_path="", out_path="", train_frac=-1.0, n_bulks=-1, n_cell_per_bulks=-1
) -> None:
    """
    Split an AnnData file into a train and test anndata file and create bulks with known ground truth for testing purposes.
    """

    create_simulation(ad_path, out_path, train_frac, n_bulks, n_cell_per_bulks)


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


@cli_analyze.command("diff")
@click.option(
    "--path",
    "-p",
    "hidedeconv_path",
    default=".",
    show_default=True,
    type=click.Path(
        exists=True, file_okay=False, dir_okay=True, writable=True, path_type=Path
    ),
    help="Path, where the HIDE-Deconv project structure is located.",
)
@assert_trained
def cli_analyze_diff(hidedeconv_path: Path) -> None:
    """
    Analyze cohort differences in deconvolution results.

    Chooses either Mann-Withney U for two cohorts or Kruskal-Wallis and posthoc Dunn test
    for multiple cohorts.

    This command requires a sample sheet, where one column links the column names of the bulks
    with the metadata used for cohort splitting.
    """

    analyze_differences(hidedeconv_path)


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


@cli_analyze.command("benchmark")
@click.option(
    "--path",
    "-p",
    "hidedeconv_path",
    default=".",
    show_default=True,
    type=click.Path(
        exists=True, file_okay=False, dir_okay=True, writable=True, path_type=Path
    ),
    help="Path, where the HIDE-Deconv project structure is located.",
)
@assert_trained
def cli_analyze_benchmark(hidedeconv_path: Path) -> None:
    """
    Benchmark the deconvoluted results against a ground truth compositions and save the results.

    Used metrics include: Spearman Correlation, Pearson Correlation, RMSE, NMAE, Cosine Similarity, Kendall Tau.

    The ground truth composition file must be on the same cell type layer as the compositions that should be benchmarked.
    """

    benchmark_result(hidedeconv_path)


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


@cli_analyze.command("pca")
@click.option(
    "--path",
    "-p",
    "hidedeconv_path",
    default=".",
    show_default=True,
    type=click.Path(
        exists=True, file_okay=False, dir_okay=True, writable=True, path_type=Path
    ),
    help="Path, where the HIDE-Deconv project structure is located.",
)
@assert_trained
def cli_analyze_pca(hidedeconv_path: Path) -> None:
    """
    Perform a principal component analysis on the deconvoluted bulk and save the resulting scatter plot.
    """

    create_pca_plot(hidedeconv_path)


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


@cli_analyze.command("umap")
@click.option(
    "--path",
    "-p",
    "hidedeconv_path",
    default=".",
    show_default=True,
    type=click.Path(
        exists=True, file_okay=False, dir_okay=True, writable=True, path_type=Path
    ),
    help="Path, where the HIDE-Deconv project structure is located.",
)
@assert_trained
def cli_analyze_umap(hidedeconv_path: Path) -> None:
    """
    Perform a principal component analysis and uniform manifold projection on the deconvoluted bulk and save the resulting scatter plot.
    """

    create_umap_plot(hidedeconv_path)


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


@cli_analyze.command("survival")
@click.option(
    "--path",
    "-p",
    "hidedeconv_path",
    default=".",
    show_default=True,
    type=click.Path(
        exists=True, file_okay=False, dir_okay=True, writable=True, path_type=Path
    ),
    help="Path, where the HIDE-Deconv project structure is located.",
)
@assert_trained
def cli_analyze_survival(hidedeconv_path: Path) -> None:
    """
    Performs a survival analysis on selected results.
    """

    survival_analysis(hidedeconv_path)


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


@cli.command("download")
def cli_download() -> None:
    """
    Select and download an AnnData Single Cell file for certain pre-curated repositories.
    """

    download_single_cells()


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


@cli_anndata.command("preprocess")
def cli_anndata_preprocess() -> None:
    """
    Applies a standard AnnData preprocessing pipeline to a given AnnData file.
    Removes cells with low quality or high mitochondrial rna expression.
    Additionally excludes celltypes, that are below the min_cell threshold and removes genes that are either ribosomal, mitochondrial or have a very low expression level.\n
    **Note:** AnnData *var_names* must be Gene Names for this function.
    """

    preprocess_anndata()


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


@cli_bulk.command("pca")
def cli_bulk_pca() -> None:
    """
    Visualizes the RNA-seq bulk as PCA plot. Additionally allows to annotate each dot, if a sample sheet is provided.
    """

    create_bulk_pca_plot()


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


@cli_bulk.command("umap")
def cli_bulk_umap() -> None:
    """
    Visualizes the RNA-seq bulk as UMAP plot. Additionally allows to annotate each dot, if a sample sheet is provided.
    """

    create_bulk_umap_plot()


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if __name__ == "__main__":
    cli()
