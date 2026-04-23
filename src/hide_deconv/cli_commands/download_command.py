"""
=====================================================
ViewModel functions for CLI download command
=====================================================
"""

from rich.console import Console

from ..utils import get_downloadable_projects
from ..download import download_file


from InquirerPy import inquirer
from InquirerPy.base.control import Choice
from InquirerPy.validator import PathValidator

from ..constants import MSG_SUCCESS, MSG_USER_ABORT

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

console = Console()

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


def download_single_cells() -> int:
    """
    Download pre-curated anndata repositories defined in the download/sc_repos.txt file
    """

    # Get downloadable files
    prj_list = get_downloadable_projects()

    console.print("[bold blue]Single Cell Reference Selection[/bold blue]")

    choices = []
    for key in prj_list.keys():
        choices.append(Choice(value=key, name=f"{prj_list[key]['source']} - {key}"))

    choices.append(Choice(value=None, name="Cancel"))

    selection = inquirer.select(
        message="Select Project to download:", choices=choices
    ).execute()

    if selection is None:
        return MSG_USER_ABORT

    save_path = inquirer.filepath(
        message="Enter path, where anndata file will be stored:",
        default="./",
        mandatory=True,
        mandatory_message="A path, where the downloaded anndata file is stored must be selected.",
        validate=PathValidator(
            is_file=False, is_dir=True, message="Input is not a folder."
        ),
    ).execute()

    ad_path = str(save_path) + "/" + str(selection) + ".h5ad"

    ret = download_file(prj_list[key]["link"], save_path=ad_path)

    if ret == MSG_SUCCESS:
        with open(str(save_path) + "/" + str(selection) + "_info.txt", "x") as f:
            f.write(prj_list[key]["meta_info"])

        console.print(f"[green]Data downloaded to {ad_path}[/green]")
        console.print(
            f"[dim]Information on citation can be found under {str(save_path) + '/' + str(selection) + '_info.txt'}"
        )

    return MSG_SUCCESS
