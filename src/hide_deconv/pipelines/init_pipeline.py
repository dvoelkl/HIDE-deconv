"""
=====================================================
Pipeline for execution of hide-deconv setup command
=====================================================
"""

from pathlib import Path
import os
from ..config import hidedeconv_config

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


def is_initialized(folder_path: str) -> bool:
    """
    Checks if a hide-deconv folder structure is already initialized.

    Parameters
    ----------
    folder_path : str
        Path of the folder where the hide-deconv project is expected.

    Returns
    -------
    bool
        True: Project was already initialized.
        False: Project is not initialized.
    """
    project_dir = Path(folder_path).expanduser().resolve()

    if not project_dir.exists() or not project_dir.is_dir():
        return False

    has_config = (project_dir / "config.json").is_file()

    has_data_dir = (project_dir / "data").is_dir()
    has_processed_dir = (project_dir / "processed").is_dir()
    has_results_dir = (project_dir / "results").is_dir()
    has_figures_dir = (project_dir / "figures").is_dir()

    return (
        has_config
        and has_data_dir
        and has_processed_dir
        and has_results_dir
        and has_figures_dir
    )


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


def init_hidedeconv(folder_path: str):
    """
    Creates a hide-deconv project structure at the specified path.\n
    /.-data /\n
    /.-processed /\n
    /.-results /\n
    /.config.json

    Parameters
    ----------
    folder_path : str
        Path of the folder where the hide-deconv project will be created.
    """

    if not os.path.exists(folder_path + "/data/"):
        os.makedirs(folder_path + "/data/")

    if not os.path.exists(folder_path + "/processed/"):
        os.makedirs(folder_path + "/processed/")

    if not os.path.exists(folder_path + "/results/"):
        os.makedirs(folder_path + "/results/")

    if not os.path.exists(folder_path + "/figures/"):
        os.makedirs(folder_path + "/figures/")

    hconf = hidedeconv_config()
    hconf.save(folder_path + "/config.json")
