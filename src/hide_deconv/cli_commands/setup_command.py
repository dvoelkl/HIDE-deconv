"""
=====================================================
ViewModel functions for CLI setup command
=====================================================
"""

from ..constants import (
    MSG_FAILURE,
    MSG_SUCCESS,
    MSG_ALREADY_INITIALIZED,
    MSG_USER_ABORT,
)
from ..pipelines import init_hidedeconv, is_initialized
from ..preprocessing import get_adata_info
from ..config import hidedeconv_config
from . import setup_config

from pathlib import Path

import pandas as pd

from InquirerPy import inquirer, prompt
from InquirerPy.validator import PathValidator

from rich.console import Console
from rich.prompt import Confirm


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

console = Console()

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


def setup_project(hidedeconv_path: Path, fAdv: bool = False) -> int:
    """
    Setups the whole project structure at a given path.

    Parameters
    ----------
    hidedeconv_path : Path
        Path to the folder where the project will be initialized.
    fAdv : bool, default=False
        If set to true, displays advanced options

    Returns
    -------
    int
        Return message indicating if project was initialized or a failure occured.


    """
    ret = init(hidedeconv_path)
    if ret == MSG_ALREADY_INITIALIZED:
        if not Confirm.ask("Overwrite the current configuration?", default=False):
            return MSG_USER_ABORT
    elif ret != MSG_SUCCESS:
        return ret

    # Load anndata file
    # Future update: Add menu with different single cell formats
    ret = load_anndata(hidedeconv_path)
    if ret != MSG_SUCCESS:
        return ret

    # Load bulk file
    ret = load_bulk(hidedeconv_path)
    if ret != MSG_SUCCESS:
        return ret

    # Setup preprocessing parameters
    ret = setup_config(hidedeconv_path, fAdv)
    if ret != MSG_SUCCESS:
        return ret

    console.print("[green]Setup completed successfully.[/green]")
    return MSG_SUCCESS


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


def init(hidedeconv_path: Path) -> int:
    """

    Parameters
    ----------
    hidedeconv_path : Path
        Path to the folder where the project will be initialized.

    Returns
    -------
    int
        Return message indicating if project was initialized or a failure occured.

    """
    if is_initialized(str(hidedeconv_path)):
        console.print("[yellow]Project already initialized.[/yellow]")
        return MSG_ALREADY_INITIALIZED

    fCreate = Confirm.ask("Create HIDE-deconv project structure?", default=True)
    if not fCreate:
        console.print("[dim]Initialization aborted.[/dim]")
        return MSG_USER_ABORT

    try:
        with console.status(
            "[bold blue]Creating folders and config.json...[/bold blue]", spinner="dots"
        ):
            init_hidedeconv(str(hidedeconv_path))
    except Exception:
        return MSG_FAILURE

    console.print("[green]Initialization successful.[/green]")
    return MSG_SUCCESS


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


def load_anndata(hidedeconv_path: Path) -> int:
    """
    Loads an AnnData Single Cell file and guides the user through setting up of different annotation levels

    Parameters
    ----------
    hidedeconv_path : Path
        Path to the project folder structure

    Returns
    -------
    int
        Return message code
    """
    hconf = hidedeconv_config.load(str(hidedeconv_path) + "/config.json")

    ad_path = inquirer.filepath(
        message="Enter path to anndata single cell file:",
        default=str(hidedeconv_path),
        mandatory=True,
        mandatory_message="An AnnData file (.h5ad) must be selected to continue setup.",
        validate=PathValidator(is_file=True, message="Input is not a file."),
    ).execute()

    hconf.sc_file_name = ad_path

    try:
        with console.status(
            "[bold blue]Loading AnnData File...[/bold blue]", spinner="dots"
        ):
            dict = get_adata_info(ad_path)
    except Exception:
        console.print_exception()
        return MSG_FAILURE

    obs_levels = sorted(dict["obs"])

    message_subtype = {
        "type": "list",
        "message": "Select the column of the finest cell type annotation level:",
        "choices": obs_levels,
        "mandatory": True,
    }

    # Save subtype column to config and remove from list of available annotations
    column_subtype = str(prompt(message_subtype)[0])
    obs_levels.remove(column_subtype)
    hconf.sub_ct_col = column_subtype

    # If desired, add higher layers
    layer = 1
    higher_celltype_layer = []
    while Confirm.ask("Add higher cell type layer?", default=False):
        message_higher = {
            "type": "list",
            "message": f"Select the column of the next highest ({layer}) cell type annotation level:",
            "choices": obs_levels,
            "mandatory": True,
        }

        column_higher = str(prompt(message_higher)[0])
        obs_levels.remove(column_higher)
        higher_celltype_layer.append(column_higher)
        layer += 1

    hconf.higher_ct_cols = higher_celltype_layer

    console.print(
        f"[green]Loaded AnnData file with {dict['n_genes']} genes and {dict['n_cells']} cells"
    )

    hconf.save(str(hidedeconv_path) + "/config.json")
    return MSG_SUCCESS


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


def load_bulk(hidedeconv_path: Path) -> int:
    """
    Loads a bulk RNA expression file

    Parameters
    ----------
        Path to the project folder structure

    Returns
    -------
    int
        Return message code
    """

    hconf = hidedeconv_config.load(str(hidedeconv_path) + "/config.json")

    bulk_path = inquirer.filepath(
        message="Enter path to bulk RNA expression file:",
        default=str(hidedeconv_path),
        mandatory=True,
        mandatory_message="A bulk RNA expression file (.csv) must be selected to continue setup.",
        validate=PathValidator(is_file=True, message="Input is not a file."),
    ).execute()

    try:
        with console.status(
            "[bold blue]Loading bulk RNA expression file...[/bold blue]", spinner="dots"
        ):
            df_bulk = pd.read_csv(bulk_path, index_col=0)
    except Exception:
        console.print_exception()
        return MSG_FAILURE

    n_genes, n_samples = df_bulk.shape

    console.print(
        f"[green]Loaded bulk RNA expression file with {n_genes} genes and {n_samples} samples"
    )
    hconf.bulk_file_name = bulk_path

    hconf.save(str(hidedeconv_path) + "/config.json")
    return MSG_SUCCESS
