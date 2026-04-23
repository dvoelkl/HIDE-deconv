"""
=====================================================
ViewModel functions for CLI preprocess
=====================================================
"""

from ..constants import MSG_SUCCESS, MSG_FAILURE
from ..pipelines import preprocessing_pipeline
from ..config import hidedeconv_config

from pathlib import Path

from rich.console import Console

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

console = Console()

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


def preprocess(hidedeconv_path: Path, fDomTransfer) -> int:
    """
    Run preprocessing for HIDEOUT by aligning genes between single-cell and bulk data,
    creating reference/hierarchy matrices, generating training bulks and optionally
    accounting for domain transfer.
    """

    try:
        with console.status(
            "[bold blue]Preprocessing single cells and bulk RNA data...[/bold blue]",
            spinner="dots",
        ):
            preprocessing_pipeline(hidedeconv_path, fDomTransfer)
    except Exception:
        console.print_exception()
        return MSG_FAILURE

    hconf = hidedeconv_config.load(str(hidedeconv_path) + "/config.json")
    hconf.preprocessed = True
    hconf.domainTransfer = fDomTransfer
    hconf.save(str(hidedeconv_path) + "/config.json")

    console.print("[green]Finished preprocessing[/green]")

    return MSG_SUCCESS
