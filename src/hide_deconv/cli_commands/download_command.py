"""
=====================================================
ViewModel functions for CLI download command
=====================================================
"""

from pathlib import Path

import requests
from InquirerPy import inquirer
from InquirerPy.base.control import Choice
from InquirerPy.validator import PathValidator
from rich.console import Console
from rich.table import Table

from ..constants import MSG_FAILURE, MSG_SUCCESS, MSG_USER_ABORT
from ..download import (
    count_by,
    download_sc_cellxgene,
    fetch_cellxgene_catalog,
    filter_datasets,
)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

console = Console()

ALL_VALUES = "__all__"
CHANGE_FILTERS = "__change_filters__"
CANCEL = "__cancel__"

MAX_TABLE_ROWS = 100

MSG_NO_INTERNET = "Internet access is necessary for the download command."

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


def format_filesize(n_bytes: int) -> str:
    """
    Format a file size in bytes as human readable string.
    """

    size = float(n_bytes)
    for unit in ["B", "KB", "MB", "GB"]:
        if size < 1000:
            return f"{size:.1f} {unit}" if unit != "B" else f"{int(size)} B"
        size /= 1000

    return f"{size:.1f} TB"


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


def select_organism(catalog: list[dict]) -> str:
    """
    Let the user select an organism via fuzzy search.
    """

    counts = count_by(catalog, "organism")
    choices = [Choice(value=org, name=f"{org} ({n})") for org, n in counts.items()]

    organism = None
    while organism is None:
        organism = inquirer.fuzzy(
            message="Select organism:",
            choices=choices,
            instruction="(type to search)",
        ).execute()

    return organism


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


def select_values(
    datasets: list[dict], field: str, all_label: str, message: str
) -> list[str] | None:
    """
    Let the user select one or multiple values of a catalog field via fuzzy search.

    Returns
    -------
    list[str] | None
        Selected values or None, if all values were selected.
    """

    counts = count_by(datasets, field)
    choices = [Choice(value=ALL_VALUES, name=all_label)] + [
        Choice(value=value, name=f"{value} ({n})") for value, n in counts.items()
    ]

    selection = []
    while not selection:
        selection = inquirer.fuzzy(
            message=message,
            choices=choices,
            multiselect=True,
            instruction="(type to search, TAB to select multiple)",
        ).execute()

    if ALL_VALUES in selection:
        return None

    return list(selection)


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


def print_dataset_table(datasets: list[dict]) -> None:
    """
    Print matching datasets with meta information as table.
    """

    table = Table(title=f"Matching datasets ({len(datasets)})", show_lines=False)
    table.add_column("#", justify="right")
    table.add_column("Title", max_width=50)
    table.add_column("Collection", max_width=40)
    table.add_column("Cells", justify="right")
    table.add_column("Assay", max_width=30)
    table.add_column("Suspension")
    table.add_column("Size", justify="right")
    table.add_column("Published")

    for i, dataset in enumerate(datasets[:MAX_TABLE_ROWS], start=1):
        table.add_row(
            str(i),
            dataset["title"],
            dataset["collection_name"],
            f"{dataset['cell_count']:,}",
            ", ".join(dataset["assay"]),
            ", ".join(dataset["suspension_type"]),
            format_filesize(dataset["filesize"]),
            dataset["published_at"],
        )

    console.print(table)

    if len(datasets) > MAX_TABLE_ROWS:
        console.print(
            f"[dim]Showing the {MAX_TABLE_ROWS} largest of {len(datasets)} datasets. "
            "All datasets can be searched in the next selection.[/dim]"
        )


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


def select_dataset(datasets: list[dict]) -> dict | str:
    """
    Let the user select a single dataset via fuzzy search.

    Returns
    -------
    dict | str
        Selected dataset, CHANGE_FILTERS or CANCEL.
    """

    choices = [
        Choice(
            value=i,
            name=f"{i + 1}. {d['title']} ({d['cell_count']:,} cells, {format_filesize(d['filesize'])})",
        )
        for i, d in enumerate(datasets)
    ]
    choices += [
        Choice(value=CHANGE_FILTERS, name="Change filters"),
        Choice(value=CANCEL, name="Cancel"),
    ]

    selection = None
    while selection is None:
        selection = inquirer.fuzzy(
            message="Select dataset to download:",
            choices=choices,
            instruction="(type to search)",
        ).execute()

    if selection in (CHANGE_FILTERS, CANCEL):
        return selection

    return datasets[selection]


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


def select_gene_id() -> str:
    """
    Let the user select the gene identifier, that is used as var_names.
    """

    return inquirer.select(
        message="Select gene identifier used as var_names:",
        choices=[
            Choice(value="symbol", name="Gene Symbols (e.g. CD4)"),
            Choice(value="ensembl", name="Ensembl IDs (e.g. ENSG00000010610)"),
        ],
        default="symbol",
    ).execute()


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


def download_cellxgene_command(protein_coding_only: bool = True) -> int:
    """
    Interactive selection and download of a single cell dataset from CELLxGENE Discover.

    Parameters
    ----------
    protein_coding_only : bool = True
        Keep only protein coding genes in the downloaded dataset.
    """

    try:
        try:
            with console.status("Fetching dataset list from CELLxGENE Discover..."):
                catalog = fetch_cellxgene_catalog()
        except (requests.ConnectionError, requests.Timeout):
            console.print(f"[red]{MSG_NO_INTERNET}[/red]")
            return MSG_FAILURE
        except requests.HTTPError as e:
            console.print(f"[red]CELLxGENE Discover is not reachable: {e}[/red]")
            return MSG_FAILURE

        console.print("[bold blue]CELLxGENE Single Cell Dataset Selection[/bold blue]")

        organism = select_organism(catalog)
        datasets_organism = filter_datasets(catalog, organism=organism)

        while True:
            tissues = select_values(
                datasets_organism, "tissue", "<All Tissues>", "Select tissue(s):"
            )
            datasets_tissue = filter_datasets(datasets_organism, tissues=tissues)

            diseases = select_values(
                datasets_tissue, "disease", "<All Diseases>", "Select disease(s):"
            )
            datasets = filter_datasets(datasets_tissue, diseases=diseases)

            if len(datasets) == 0:
                console.print("[yellow]No datasets match the selection.[/yellow]")
                continue

            datasets = sorted(datasets, key=lambda d: d["cell_count"], reverse=True)
            print_dataset_table(datasets)

            dataset = select_dataset(datasets)
            if dataset == CANCEL:
                return MSG_USER_ABORT
            if dataset != CHANGE_FILTERS:
                break

        save_dir = inquirer.filepath(
            message="Enter folder, where the AnnData file will be stored:",
            default="./",
            only_directories=True,
            mandatory=True,
            mandatory_message="A folder, where the downloaded AnnData file is stored must be selected.",
            validate=PathValidator(
                is_file=False, is_dir=True, message="Input is not a folder."
            ),
        ).execute()

        if not inquirer.confirm(
            message=f"Download '{dataset['title']}' ({format_filesize(dataset['filesize'])})?",
            default=True,
        ).execute():
            return MSG_USER_ABORT

        subset = False
        if tissues is not None or diseases is not None:
            subset = inquirer.confirm(
                message="Subset dataset to the selected tissues and diseases?",
                default=True,
            ).execute()

        gene_id = select_gene_id()

        ad_path = download_sc_cellxgene(
            dataset["dataset_id"],
            save_dir,
            tissues=tissues if subset else None,
            diseases=diseases if subset else None,
            catalog=catalog,
            gene_id=gene_id,
            protein_coding_only=protein_coding_only,
        )
    except KeyboardInterrupt:
        console.print("[red]Download aborted.[/red]")
        return MSG_USER_ABORT
    except (requests.ConnectionError, requests.Timeout):
        console.print(f"[red]{MSG_NO_INTERNET}[/red]")
        return MSG_FAILURE
    except Exception:
        console.print_exception()
        return MSG_FAILURE

    ad_path = Path(ad_path)
    citation_path = ad_path.with_name(f"{ad_path.stem}_citation.txt")

    console.print(f"[green]Data downloaded to {ad_path}[/green]")
    console.print(f"[green]Citation information stored in {citation_path}[/green]")
    console.print(
        "[dim]Please cite the original publication of the dataset and CZ CELLxGENE Discover.[/dim]"
    )

    return MSG_SUCCESS
