"""
=====================================================
Utility functions for command line interface
=====================================================
"""

import inspect
from pathlib import Path
import os
from functools import wraps
from rich.console import Console
import pandas as pd

from InquirerPy import inquirer

from ..config import hidedeconv_config

from ..constants import (
    MSG_ALREADY_INITIALIZED,
    MSG_NOT_INITIALIZED,
    MSG_ALREADY_PREPROCESSED,
    MSG_NOT_PREPROCESSED,
    MSG_ALREADY_TRAINED,
    MSG_NOT_TRAINED,
)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

console = Console()

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


def get_project_init_status(hidedeconv_path: Path) -> int:
    """
    Checks if project is correctly initialized at given path.

    Returns
    -------
    int
        Either MSG_ALREADY_INITIALIZED or MSG_NOT_INITIALIZED

    """

    ret = MSG_ALREADY_INITIALIZED

    try:
        if not os.path.exists(str(hidedeconv_path) + "/data/"):
            ret = MSG_NOT_INITIALIZED

        if not os.path.exists(str(hidedeconv_path) + "/processed/"):
            ret = MSG_NOT_INITIALIZED

        if not os.path.exists(str(hidedeconv_path) + "/results/"):
            ret = MSG_NOT_INITIALIZED

        hidedeconv_config.load(str(hidedeconv_path) + "/config.json")
    except Exception:
        ret = MSG_NOT_INITIALIZED

    return ret


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


def assert_init(func):
    """
    Decorator for function, that assures, that a project is located at the designated path.

    It is necessary, that one first argument of the decorated function is named hidedeconv_path

    Usage
    -----
    @assert_init

    deconvolve(hidedeconv_path : Path, ....)
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        bound = inspect.signature(func).bind_partial(*args, **kwargs)
        hidedeconv_path = bound.arguments.get("hidedeconv_path", None)

        if get_project_init_status(hidedeconv_path) == MSG_ALREADY_INITIALIZED:
            func(*args, **kwargs)
        else:
            console.print(
                f"[red]No project located at {hidedeconv_path.absolute()}.[/red]"
            )
            console.print(
                "[dim]Please create a project here using [i]hide-deconv init[/i] or attach a path to a valid project via the [i]-p <Path>[/i] keyword[/dim]"
            )

    return wrapper


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def get_project_preprocessed_status(hidedeconv_path: Path) -> int:
    """
    Checks if project at given path is preprocessed.

    Returns
    -------
    int
        Returns either MSG_NOT_PREPROCESSED or MSG_ALREADY_PREPROCESSED.

    """

    ret = MSG_ALREADY_PREPROCESSED

    try:
        hconf = hidedeconv_config.load(str(hidedeconv_path) + "/config.json")

        if not hconf.preprocessed:
            ret = MSG_NOT_PREPROCESSED

    except Exception:
        ret = MSG_NOT_PREPROCESSED

    return ret


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def assert_preprocessed(func):
    """
    Decorator for function, that assures, that a project is located and preprocessed at the designated path.

    It is necessary, that one argument of the decorated function is named hidedeconv_path

    Usage
    -----
    @assert_preprocessed

    deconvolve(hidedeconv_path : Path, ....)
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        bound = inspect.signature(func).bind_partial(*args, **kwargs)
        hidedeconv_path = bound.arguments.get("hidedeconv_path", None)

        if get_project_init_status(hidedeconv_path) == MSG_ALREADY_INITIALIZED:
            if (
                get_project_preprocessed_status(hidedeconv_path)
                == MSG_ALREADY_PREPROCESSED
            ):
                func(*args, **kwargs)
            else:
                console.print(
                    f"[red]Project located at {hidedeconv_path.absolute()} is not preprocessed.[/red]"
                )
                console.print(
                    "[dim]Please preprocess project here using [i]hide-deconv preprocess[/i] or attach a path to a preprocessed project via the [i]-p <Path>[/i] keyword.[/dim]"
                )
        else:
            console.print(
                f"[red]No project located at {hidedeconv_path.absolute()}.[/red]"
            )
            console.print(
                "[dim]Please create a project here using [i]hide-deconv init[/i] or attach a path to a valid project via the [i]-p <Path>[/i] keyword.[/dim]"
            )

    return wrapper


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


def get_project_trained_status(hidedeconv_path: Path) -> int:
    """
    Checks if project at given path is trained.

    Returns
    -------
    int
        Returns either MSG_NOT_TRAINED or MSG_ALREADY_TRAINED.

    """

    ret = MSG_ALREADY_TRAINED

    try:
        hconf = hidedeconv_config.load(str(hidedeconv_path) + "/config.json")

        if not hconf.trained:
            ret = MSG_NOT_TRAINED

    except Exception:
        ret = MSG_NOT_TRAINED

    return ret


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def assert_trained(func):
    """
    Decorator for function, that assures, that a project is located and preprocessed at the designated path.

    It is necessary, that one argument of the decorated function is named hidedeconv_path

    Usage
    -----
    @assert_preprocessed

    deconvolve(hidedeconv_path : Path, ....)
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        bound = inspect.signature(func).bind_partial(*args, **kwargs)
        hidedeconv_path = bound.arguments.get("hidedeconv_path", None)

        if get_project_init_status(hidedeconv_path) == MSG_ALREADY_INITIALIZED:
            if (
                get_project_preprocessed_status(hidedeconv_path)
                == MSG_ALREADY_PREPROCESSED
            ):
                if get_project_trained_status(hidedeconv_path) == MSG_ALREADY_TRAINED:
                    func(*args, **kwargs)
                else:
                    console.print(
                        f"[red]Project located at {hidedeconv_path.absolute()} is not trained.[/red]"
                    )
                    console.print(
                        "[dim]Please train project here using [i]hide-deconv train[/i] or attach a path to a trained project via the [i]-p <Path>[/i] keyword.[/dim]"
                    )
            else:
                console.print(
                    f"[red]Project located at {hidedeconv_path.absolute()} is not preprocessed.[/red]"
                )
                console.print(
                    "[dim]Please preprocess project here using [i]hide-deconv preprocess[/i] or attach a path to a preprocessed project via the [i]-p <Path>[/i] keyword.[/dim]"
                )
        else:
            console.print(
                f"[red]No project located at {hidedeconv_path.absolute()}.[/red]"
            )
            console.print(
                "[dim]Please create a project here using [i]hide-deconv init[/i] or attach a path to a valid project via the [i]-p <Path>[/i] keyword.[/dim]"
            )

    return wrapper


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


def get_deconvolution_results(hidedeconv_path: Path) -> list[str]:
    """
    Gets all deconvoluted results available in a project.

    Returns
    -------
    list[str]
        List of folder names of all deconvoluted results

    """

    ret = []

    try:
        hconf = hidedeconv_config.load(str(hidedeconv_path) + "/config.json")
        necessary_filenames = hconf.higher_ct_cols
        necessary_filenames.append("sub")
        necessary_filenames = set(necessary_filenames)

        # Get all folders in results directory
        result_folders = [
            f.path for f in os.scandir(str(hidedeconv_path) + "/results/") if f.is_dir()
        ]

        if len(result_folders) > 0:
            # Check for each folder that all composition files are present (and that this is not a relict)
            for folder in result_folders:
                compositions_filenames = [
                    Path(f.path).stem for f in os.scandir(folder) if f.is_file()
                ]
                compositions_filenames = list(
                    filter(lambda k: "C_" in k, compositions_filenames)
                )
                compositions_filenames = set([f[2:] for f in compositions_filenames])

                if compositions_filenames == necessary_filenames:
                    ret.append(Path(folder).stem)

    except Exception:
        ret = []

    return ret


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


def load_project_bulk(hidedeconv_path: Path) -> tuple[str, str, pd.DataFrame]:

    hconf = hidedeconv_config.load(str(hidedeconv_path) + "/config.json")

    # Select deconvoluted data
    available_projects = get_deconvolution_results(hidedeconv_path)

    selected_project = inquirer.select(
        message="Select result folder:",
        choices=available_projects,
        default=available_projects[0],
        mandatory=True,
    ).execute()

    available_layers = ["sub"]
    available_layers.extend(hconf.higher_ct_cols)

    selected_ct_layer = inquirer.select(
        message="Select cell type layer:",
        choices=available_layers,
        default="sub",
        mandatory=True,
    ).execute()

    bulk = pd.read_csv(
        str(hidedeconv_path)
        + "/results/"
        + selected_project
        + "/C_"
        + selected_ct_layer
        + ".csv",
        index_col=0,
    )

    return selected_project, selected_ct_layer, bulk
