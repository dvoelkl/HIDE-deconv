"""
=====================================================
ViewModel functions for CLI simulate
=====================================================
"""

from ..constants import MSG_FAILURE, MSG_SUCCESS

from ..preprocessing import train_test_split_adata, create_bulks
from ..preprocessing import get_adata_info
import anndata as ad
from pathlib import Path

from InquirerPy import inquirer, prompt
from InquirerPy.validator import PathValidator

from rich.console import Console

"""
Planned implementations:
- AnnData Train Test split
- Interface to various scanpy function:
    - scanpy.pp.filter_cells ?
    - scanpy.pp.filter_genes
    -
- Analysis functions, e.g.
    - https://pydeseq2.readthedocs.io/en/stable/

"""
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

console = Console()

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


def create_simulation(
    ad_path="", output_path="", train_frac=-1.0, n_bulks=-1, n_cells_per_bulk=-1
) -> int:
    """
    Split AnnData File into train and test set and create test bulks with true proportions.

    Parameters
    ----------
    ad_path : str = ""
        Path to AnnData File. If set, question for filepath is skipped.
    output_path : str = ""
        Path, where the created data will be stored. If set, question for filepath is skipped.
    train_frac : float = -1.0
        Percentage of all cells used in training. If set, question for fraction is skipped.
    n_bulk : int = -1
        Number of generated test bulks. If set, question for number of test bulks is skipped.
    n_cells_per_bulk : int = -1
        Number of cells used to simulate a single test bulk. If set, question for number of cells is skipped.

    Returns
    -------
    int
        Return code. Either MSG_SUCCESS or MSG_FAILURE.
    """
    console.print("[bold blue]Data Simulation[/bold blue]")

    try:
        # Load anndata and select cell type column to split
        if ad_path is None:
            ad_path = Path(
                inquirer.filepath(
                    message="Enter path to anndata single cell file:",
                    default=str("."),
                    mandatory=True,
                    mandatory_message="An AnnData file (.h5ad) must be selected to continue simulation.",
                    validate=PathValidator(
                        is_file=True, message="Input is not a file."
                    ),
                ).execute()
            )
        else:
            ad_path = Path(ad_path)

        # Select an output path
        if output_path is None:
            output_path = inquirer.filepath(
                message="Enter path, where simulated data will be stored",
                default=str("."),
                mandatory=True,
                mandatory_message="A folder must be selected to continue simulation.",
                validate=PathValidator(
                    is_file=False, is_dir=True, message="Input is not a folder."
                ),
            ).execute()

        with console.status(
            "[bold blue]Loading AnnData File...[/bold blue]", spinner="dots"
        ):
            dict = get_adata_info(str(ad_path.expanduser()))
            adata = ad.read_h5ad(ad_path)

        obs_levels = sorted(dict["obs"])

        message_subtype = {
            "type": "list",
            "message": "Select the column of the finest cell type annotation level:",
            "choices": obs_levels,
            "mandatory": True,
        }

        celltype_col = str(prompt(message_subtype)[0])

        if train_frac is None:
            train_frac = float(
                inquirer.number(
                    "Enter percentage of all cells used for training:",
                    min_allowed=0.0,
                    max_allowed=1.0,
                    default=0.7,
                    mandatory=True,
                    float_allowed=True,
                    mandatory_message="A number between 0.0 and 1.0 must be entered.",
                ).execute()
            )

        # Split anndata depending on selected cell type column
        with console.status(
            "[bold blue]Splitting AnnData File...[/bold blue]", spinner="dots"
        ):
            train, test = train_test_split_adata(adata, celltype_col, train_frac)

        train.write_h5ad(str(output_path) + f"/{ad_path.stem}_train.h5ad")
        test.write_h5ad(str(output_path) + f"/{ad_path.stem}_test.h5ad")

        console.print("[green]Splitted AnnData file into train and test file.[/green]")

        # Get settings for creation of test bulks
        if n_bulks is None:
            n_bulks = int(
                inquirer.number(
                    "Enter number of in-silico bulks generated for testing:",
                    min_allowed=1,
                    default=1000,
                    mandatory=True,
                    mandatory_message="A number greater than 0 must be entered.",
                ).execute()
            )

        if n_cells_per_bulk is None:
            n_cells_per_bulk = int(
                inquirer.number(
                    "Enter number of cells accumulated to a single in-silico bulk:",
                    min_allowed=1,
                    default=100,
                    mandatory=True,
                    mandatory_message="A number greater than 0 must be entered.",
                ).execute()
            )

        with console.status(
            "[bold blue]Creating test bulks...[/bold blue]", spinner="dots"
        ):
            Y_test, C_test = create_bulks(
                test, n_bulks, n_cells_per_bulk, celltype_col=celltype_col
            )

        Y_test.to_csv(str(output_path) + f"/{ad_path.stem}_test_bulk.csv")
        C_test.to_csv(str(output_path) + f"/{ad_path.stem}_test_proportions.csv")

        console.print(
            "[green]Created in-silico test bulks with corresponding proportions.[/green]"
        )

    except Exception:
        console.print_exception()
        return MSG_FAILURE

    return MSG_SUCCESS
