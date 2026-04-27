"""
=====================================================
Functions for downloading files
=====================================================
"""

from rich.console import Console
from rich.progress import (
    Progress,
    DownloadColumn,
    BarColumn,
    TextColumn,
    TransferSpeedColumn,
    TimeRemainingColumn,
)
from pathlib import Path


from ..constants import MSG_SUCCESS, MSG_FAILURE

import requests

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

console = Console()

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


def download_file(url, save_path) -> int:
    """
    Downloads a file from a given url and saves it at a specified path.

    Parameters
    ----------
    url : str
        URL of the file.
    save_path : str
        Path, where the downloaded file will be stored.
    
    Returns
    -------
    int
        Either MSG_SUCCESS or MSG_FAILURE
    
    """

    try:
        r = requests.get(url, stream=True, allow_redirects=True)

        chunk_size = 1024
        file_size = int(r.headers.get("Content-Length", None))

        with Progress(
            TextColumn("[bold blue]{task.description}", justify="right"),
            BarColumn(bar_width=40),
            "[progress.percentage]{task.percentage:>3.1f}%",
            DownloadColumn(),
            TransferSpeedColumn(),
            TimeRemainingColumn(),
            refresh_per_second=2,
        ) as progress:
            task = progress.add_task(
                f"[cyan]Downloading {Path(save_path).name}...", total=file_size
            )

            with open(save_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        progress.update(task, advance=len(chunk))
    except Exception:
        console.print_exception()
        return MSG_FAILURE

    return MSG_SUCCESS
