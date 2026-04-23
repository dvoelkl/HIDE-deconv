"""
=====================================================
Utility functions for configuration
=====================================================
"""

from importlib import resources as imprsc
from .. import download
from ..constants import SC_PROJECT_DOWNLOAD_PATH


def get_downloadable_projects() -> dict[str, dict[str, str]]:
    """
    Get a list of all single cell repositories listed under ../download/sc_repos.txt

    Returns
    -------
    dict[str, dict[str, str]]
        Dictionary with data set name as keys and subdictionary containing "source", "link_date" and "link"
    """

    sc_project_list = imprsc.files(download) / SC_PROJECT_DOWNLOAD_PATH

    sc_projects = {}

    try:
        with sc_project_list.open(mode="r") as f:
            source = ""
            for line in f:
                if line.startswith("["):
                    source = line.removeprefix("[").removesuffix("]\n").strip()
                elif line.startswith(">"):
                    meta_info = line.removeprefix(">").removesuffix("\n").strip()
                elif line.startswith(":"):
                    data = line.split(":", maxsplit=3)
                    project = {
                        "source": str(source),
                        "link_date": str(data[2]).strip(),
                        "link": str(data[3].removesuffix("\n").strip()),
                        "meta_info": str(meta_info),
                    }

                    sc_projects[data[1].strip()] = project
    except FileNotFoundError:
        sc_projects = None

    return sc_projects
