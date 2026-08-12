"""
=====================================================
ViewModel functions for CLI train command
=====================================================
"""

from ..constants import MSG_SUCCESS, MSG_FAILURE
from ..pipelines import train_pipeline

from pathlib import Path

from rich.console import Console


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

console = Console()

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


def train_model(hidedeconv_path: Path) -> int:
    """
    Trains the model to learn the optimal gene weights.
    """
    console.print("[bold blue]Training model...[/bold blue]")

    try:
        train_pipeline(hidedeconv_path)
    except Exception:
        console.print_exception()
        return MSG_FAILURE

    return MSG_SUCCESS
