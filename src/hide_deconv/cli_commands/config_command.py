"""
=====================================================
ViewModel functions for all CLI config commands
=====================================================
"""

import pandas as pd
import anndata as ad

from InquirerPy import inquirer

from rich import box
from rich.console import Console
from rich.table import Table

from ..constants import MSG_SUCCESS, MSG_FAILURE
from ..config import hidedeconv_config
from ..preprocessing import get_common_genes
from pathlib import Path

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

console = Console()

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


def setup_config(hidedeconv_path: Path, fAdv: bool = False) -> int:
    """
    Sets the parameters necessary for preprocessing.
    """
    ret = config_genes(hidedeconv_path)
    if ret != MSG_SUCCESS:
        return ret

    ret = config_train_bulks(hidedeconv_path)
    if ret != MSG_SUCCESS:
        return ret

    ret = config_cells_per_bulk(hidedeconv_path)
    if ret != MSG_SUCCESS:
        return ret

    ret = config_hide_iter(hidedeconv_path)
    if ret != MSG_SUCCESS:
        return ret

    return MSG_SUCCESS


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


def config_genes(hidedeconv_path: Path) -> int:
    """
    Sets the number of used genes for training and deconvolution.
    """
    hconf = hidedeconv_config.load(str(hidedeconv_path) + "/config.json")
    ad_path = hconf.sc_file_name
    bulk_path = hconf.bulk_file_name

    # n_genes
    try:
        with console.status(
            "[bold blue]Getting shared genes...[/bold blue]", spinner="dots"
        ):
            adata = ad.read_h5ad(ad_path)
            bulk_df = pd.read_csv(bulk_path, index_col=0)

            n_shared_genes = len(get_common_genes(adata, bulk_df))
    except Exception:
        console.print_exception()
        return MSG_FAILURE

    if n_shared_genes == 0:
        console.print("[red bold]Number of shared genes is 0.[/red bold]")
        console.print(
            "[dim]Do Bulk and AnnData both have the same format for genes? (e.g. EnsemblID, GeneSymbol, Entrez)[/dim]"
        )
        return MSG_FAILURE
    else:
        console.print(f"Shared genes between single cells and bulk: {n_shared_genes}")

    n_genes = inquirer.number(
        "Enter number of genes to use for deconvolution:",
        min_allowed=1,
        max_allowed=n_shared_genes,
        default=min(1000, n_shared_genes),
        mandatory=True,
        mandatory_message=f"A number between 1 and {n_shared_genes} (number of shared genes) must be entered.",
    ).execute()
    hconf.n_genes = n_genes
    hconf.save(str(hidedeconv_path) + "/config.json")

    return MSG_SUCCESS


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


def config_cells_per_bulk(hidedeconv_path: Path) -> int:
    """
    Sets the number of cells used per training bulk
    """
    hconf = hidedeconv_config.load(str(hidedeconv_path) + "/config.json")
    n_cells_per_bulk = inquirer.number(
        "Enter number of cells accumulated to a single in-silico bulk:",
        min_allowed=1,
        default=100,
        mandatory=True,
        mandatory_message="A number greater than 0 must be entered.",
    ).execute()
    hconf.n_cells_per_bulk = n_cells_per_bulk
    hconf.save(str(hidedeconv_path) + "/config.json")

    return MSG_SUCCESS


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


def config_train_bulks(hidedeconv_path: Path) -> int:
    """
    Sets number of generated in-silico training bulks
    """
    hconf = hidedeconv_config.load(str(hidedeconv_path) + "/config.json")
    n_train_bulks = inquirer.number(
        "Enter number of in-silico bulks to be used in training:",
        min_allowed=1,
        default=5000,
        mandatory=True,
        mandatory_message="A number greater than 0 must be entered.",
    ).execute()
    hconf.n_train_bulks = n_train_bulks
    hconf.save(str(hidedeconv_path) + "/config.json")

    return MSG_SUCCESS


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


def config_hide_iter(hidedeconv_path: Path) -> int:
    """
    Sets the number of iterations used for HIDE
    """
    hconf = hidedeconv_config.load(str(hidedeconv_path) + "/config.json")
    n_hide_iter = inquirer.number(
        "Enter number of iterations for training:",
        min_allowed=1,
        default=1000,
        mandatory=True,
        mandatory_message="A number greater than 0 must be entered.",
    ).execute()
    hconf.n_hide_iter = n_hide_iter
    hconf.save(str(hidedeconv_path) + "/config.json")

    return MSG_SUCCESS


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


def show_config(hidedeconv_path: Path) -> int:
    """
    Shows a summary of all parameters
    """

    hconf = hidedeconv_config.load(str(hidedeconv_path) + "/config.json")

    summary = Table(
        box=box.ASCII2,
        show_header=True,
        header_style="bold cyan",
    )

    summary.add_column("Key", style="bold", no_wrap=True)
    summary.add_column("Value", overflow="fold")

    summary.add_row("Project path", str(hidedeconv_path))
    summary.add_row("Single Cell File", hconf.sc_file_name)
    summary.add_row("Bulk RNA Expression File", hconf.bulk_file_name)
    summary.add_row("Subtype column", hconf.sub_ct_col)
    summary.add_row("Higher layer columns", ", ".join(hconf.higher_ct_cols) or "-")

    summary.add_section()
    summary.add_row("n_genes", str(hconf.n_genes))
    summary.add_row("n_train_bulks", str(hconf.n_train_bulks))
    summary.add_row("n_hide_iter", str(hconf.n_hide_iter))

    summary.add_section()

    console.print(summary)

    return MSG_SUCCESS
