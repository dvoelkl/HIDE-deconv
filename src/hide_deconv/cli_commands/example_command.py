"""
=====================================================
ViewModel functions for CLI example command
=====================================================
"""

from pathlib import Path

from rich.console import Console

from InquirerPy import inquirer
from InquirerPy.validator import PathValidator

from ..constants import MSG_SUCCESS, MSG_FAILURE

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

console = Console()

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


def create_example_data(out_path: Path | None = None) -> int:
    """
    Generate a synthetic example dataset and store the AnnData file, bulk and sample sheet.

    Parameters
    ----------
    out_path : Path | None = None
        Folder, where the example files will be stored. If None, the user is prompted.

    Returns
    -------
    int
        MSG_SUCCESS if the files were created, MSG_FAILURE otherwise.
    """

    from ..example import generate_example

    try:
        if out_path is None:
            out_path = inquirer.filepath(
                message="Enter path, where the example data will be stored:",
                default="./",
                mandatory=True,
                mandatory_message="A folder must be selected to store the example data.",
                validate=PathValidator(
                    is_file=False, is_dir=True, message="Input is not a folder."
                ),
            ).execute()

        out_path = Path(out_path).expanduser()

        with console.status(
            "[bold blue]Generating example data...[/bold blue]", spinner="dots"
        ):
            adata, bulk, sample_sheet = generate_example()

            adata.write_h5ad(out_path / "single_cells.h5ad")
            bulk.to_csv(out_path / "bulk.csv")
            sample_sheet.to_csv(out_path / "sample_sheet.csv", index=False)

    except Exception:
        console.print_exception()
        return MSG_FAILURE

    console.print(f"[green]Example data stored in {out_path}[/green]")
    console.print(
        "[dim]single_cells.h5ad: cell type columns 'cell_type' (finest layer) and 'major'[/dim]"
    )
    console.print(
        "[dim]sample_sheet.csv: sample ids in column 'SampleID', cohorts in 'disease' and 'medication', "
        "survival in 'survival_time' and 'event'[/dim]"
    )

    return MSG_SUCCESS
