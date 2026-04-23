"""
=====================================================
ViewModel functions for CLI deconvolve commands
=====================================================
"""

from ..constants import MSG_SUCCESS, MSG_FAILURE, MSG_USER_ABORT, MODEL_HIDE
from ..pipelines import deconvolve_hide_pipeline

from pathlib import Path

from rich.console import Console

from InquirerPy import inquirer
from InquirerPy.base.control import Choice

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

console = Console()

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


def deconvolve_command(
    hidedeconv_path: Path, alternative_bulk_path=None, verbose: bool = False
) -> int:
    """
    Provides menu wrapper to deconvolution methods and their parameters.

    Parameters
    ----------
    hidedeconv_path : Path
        Path to the folder where the project was initialized.

    alternative_bulk_path : Path, default = None
        Path to an alternative bulk, that should be used for deconvolution (instead of bulk from configuration)

    Returns
    -------
    int
        Return message indicating if deconvolution was successfull or a failure occured.
    """

    deconv_model = inquirer.select(
        message="Select deconvolution model:",
        choices=[
            Choice(value=MODEL_HIDE, name="HIDE"),
            Choice(value=MSG_USER_ABORT, name="Abort"),
        ],
        default=None,
    ).execute()

    # Evaluate user choice
    if deconv_model == MSG_USER_ABORT:
        return MSG_USER_ABORT
    elif deconv_model == MODEL_HIDE:
        deconvolve_hide(hidedeconv_path, alternative_bulk_path)
    return MSG_SUCCESS


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


def lambda_validator(val) -> bool:
    """
    Validates if input can be parsed as regularization parameter (float, non-negative).

    Returns
    -------
    bool
        True if can be parsed as regularization parameter
    """

    parsable = True

    try:
        val = float(val)

        if val < 0.0:
            parsable = False

    except ValueError:
        parsable = False

    return parsable


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


def deconvolve_hide(hidedeconv_path: Path, alternative_bulk_path=None) -> int:
    """
    Run HIDE deconvolution on bulk.

    Parameters
    ----------
    hideout_path : Path
        Path to the folder where the project was initialized.

    alternative_bulk_path : Path, default = None
        Path to an alternative bulk, that should be used for deconvolution (instead of bulk from configuration)

    Returns
    -------
    int
        Return message indicating if deconvolution was successfull or a failure occured.

    """

    try:
        with console.status(
            "[bold blue]Running HIDE Deconvolution on bulk RNA seq data...[/bold blue]",
            spinner="dots",
        ):
            deconvolve_hide_pipeline(hidedeconv_path, alternative_bulk_path)
    except KeyError:
        console.print(
            "[red]The gene labels between the selected bulk file and the used training set differ.[/red]"
        )
        console.print(
            "[dim]Either setup a new project using this bulk, such that the gene intersection can automatically be deduced or "
            "ensure, that the genes used for training are a subset of the genes in this bulk.[/dim]"
        )
        return MSG_FAILURE
    except Exception:
        console.print_exception()
        return MSG_FAILURE

    if alternative_bulk_path is None:
        results_name = "HIDE"
    else:
        results_name = f"HIDE_{Path(alternative_bulk_path).stem}"

    console.print("[green]HIDE Deconvolution successful.[/green]")
    console.print(
        f"[dim]Results stored in {str(hidedeconv_path) + f'/results/{results_name}/'}[/dim]"
    )

    return MSG_SUCCESS
