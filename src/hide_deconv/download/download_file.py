"""
=====================================================
Functions for querying and downloading single cell
datasets from CELLxGENE Discover
=====================================================
"""

import warnings
from collections import Counter
from datetime import date
from pathlib import Path

import anndata as ad
import numpy as np
import requests
from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    TextColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)

from ..constants import (
    CELLXGENE_API_TIMEOUT,
    CELLXGENE_COLLECTIONS_URL,
    CELLXGENE_DATASETS_URL,
    CELLXGENE_PLATFORM_CITATION,
    DOWNLOAD_CHUNK_SIZE,
    DOWNLOAD_TIMEOUT,
)
from ..utils import switch_adata_var_names

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


def get_h5ad_asset(dataset: dict) -> tuple[str, int] | None:
    """
    Get the H5AD asset of a dataset entry returned by the CELLxGENE Discover API.

    Parameters
    ----------
    dataset : dict
        Dataset entry as returned by the CELLxGENE Discover API.

    Returns
    -------
    tuple[str, int] | None
        URL and file size in bytes of the H5AD file, or None if the dataset
        does not provide an H5AD file.
    """

    for asset in dataset.get("assets") or []:
        if asset.get("filetype") == "H5AD" and asset.get("url"):
            return asset["url"], int(asset.get("filesize") or 0)

    return None


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


def fetch_cellxgene_catalog() -> list[dict]:
    """
    Fetch the list of all public datasets from CELLxGENE Discover.

    The catalog is requested from the CELLxGENE Discover API on every call.
    Only datasets providing an H5AD file are returned.

    Returns
    -------
    list[dict]
        List of datasets. Each dataset is a dictionary with the keys
        "dataset_id", "title", "collection_name", "organism", "tissue",
        "disease", "assay", "suspension_type" (lists of labels), "cell_count",
        "published_at", "url" and "filesize". Additionally the citation information
        "collection_id", "collection_doi", "collection_doi_label", "citation" and
        "explorer_url" is stored.

    Raises
    ------
    requests.ConnectionError
        If the CELLxGENE Discover API cannot be reached.
    requests.Timeout
        If the CELLxGENE Discover API does not respond in time.
    requests.HTTPError
        If the CELLxGENE Discover API returns an error status.
    """

    response = requests.get(CELLXGENE_DATASETS_URL, timeout=CELLXGENE_API_TIMEOUT)
    response.raise_for_status()

    def labels(entries: list[dict] | None) -> list[str]:
        return list(dict.fromkeys(e["label"] for e in entries or [] if "label" in e))

    catalog = []
    for dataset in response.json():
        if dataset.get("tombstone"):
            continue

        asset = get_h5ad_asset(dataset)
        if asset is None:
            continue

        url, filesize = asset
        catalog.append(
            {
                "dataset_id": dataset["dataset_id"],
                "title": dataset.get("title", ""),
                "collection_name": dataset.get("collection_name", ""),
                "organism": labels(dataset.get("organism")),
                "tissue": labels(dataset.get("tissue")),
                "disease": labels(dataset.get("disease")),
                "assay": labels(dataset.get("assay")),
                "suspension_type": list(dataset.get("suspension_type") or []),
                "cell_count": int(dataset.get("cell_count") or 0),
                "published_at": (dataset.get("published_at") or "")[:10],
                "url": url,
                "filesize": filesize,
                "collection_id": dataset.get("collection_id") or "",
                "collection_doi": dataset.get("collection_doi") or "",
                "collection_doi_label": dataset.get("collection_doi_label") or "",
                "citation": dataset.get("citation") or "",
                "explorer_url": dataset.get("explorer_url") or "",
            }
        )

    return catalog


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


def count_by(datasets: list[dict], field: str) -> dict[str, int]:
    """
    Count the number of datasets per value of a catalog field.

    Parameters
    ----------
    datasets : list[dict]
        Datasets as returned by fetch_cellxgene_catalog.
    field : str
        Field to count, e.g. "organism", "tissue" or "disease".

    Returns
    -------
    dict[str, int]
        Number of datasets per value, sorted by descending count and name.
    """

    counts = Counter(value for dataset in datasets for value in dataset[field])

    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0].lower())))


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


def filter_datasets(
    datasets: list[dict],
    organism: str | None = None,
    tissues: list[str] | None = None,
    diseases: list[str] | None = None,
) -> list[dict]:
    """
    Filter datasets by organism, tissues and diseases.

    A dataset matches, if it contains the organism and at least one of the
    given tissues and at least one of the given diseases. Note that tissues and
    diseases are annotated per dataset, a matching dataset can therefore also
    contain cells of other tissues and diseases.

    Parameters
    ----------
    datasets : list[dict]
        Datasets as returned by fetch_cellxgene_catalog.
    organism : str | None = None
        Organism label, e.g. "Homo sapiens". None matches all organisms.
    tissues : list[str] | None = None
        Tissue labels. None or an empty list matches all tissues.
    diseases : list[str] | None = None
        Disease labels. None or an empty list matches all diseases.

    Returns
    -------
    list[dict]
        Datasets matching all given criteria.
    """

    def matches(values: list[str], selection: list[str] | None) -> bool:
        return not selection or any(value in selection for value in values)

    return [
        dataset
        for dataset in datasets
        if (organism is None or organism in dataset["organism"])
        and matches(dataset["tissue"], tissues)
        and matches(dataset["disease"], diseases)
    ]


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


def download_file(url: str, save_path: str | Path) -> Path:
    """
    Download a file from a given url and store it at a specified path.

    The file is first written to "<save_path>.part" and renamed after the
    download finished. If the download fails or is aborted, the partial file
    is removed.

    Parameters
    ----------
    url : str
        URL of the file.
    save_path : str | Path
        Path, where the downloaded file will be stored.

    Returns
    -------
    Path
        Path of the downloaded file.
    """

    save_path = Path(save_path)
    part_path = save_path.with_name(save_path.name + ".part")

    try:
        with requests.get(url, stream=True, timeout=DOWNLOAD_TIMEOUT) as response:
            response.raise_for_status()

            content_length = response.headers.get("Content-Length")
            total = int(content_length) if content_length else None

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
                    f"[cyan]Downloading {save_path.name}...", total=total
                )

                with open(part_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=DOWNLOAD_CHUNK_SIZE):
                        if chunk:
                            f.write(chunk)
                            progress.update(task, advance=len(chunk))

        part_path.replace(save_path)
    except BaseException:
        part_path.unlink(missing_ok=True)
        raise

    return save_path


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


def prepare_cellxgene_adata(
    path: str | Path,
    tissues: list[str] | None = None,
    diseases: list[str] | None = None,
    gene_id: str = "symbol",
    protein_coding_only: bool = True,
) -> Path:
    """
    Prepare a downloaded CELLxGENE AnnData file for HIDE-Deconv.

    Optionally subsets the cells to the given tissues and diseases. If raw counts
    are stored in adata.raw, these are moved to adata.X and adata.raw is removed.
    Spike-ins are always removed and optionally only protein coding genes are kept.

    CELLxGENE stores Ensembl IDs as var_names and gene symbols in adata.var["feature_name"].
    If gene symbols are selected, these are used as var_names and the Ensembl IDs are stored
    in adata.var["feature_id"]. Genes with the same gene symbol are summed up. If Ensembl IDs
    are selected, the index of adata.var is named "feature_id".
    The file is overwritten with the prepared AnnData object.

    Parameters
    ----------
    path : str | Path
        Path of the CELLxGENE AnnData file.
    tissues : list[str] | None = None
        Keep only cells with these labels in obs["tissue"]. None keeps all tissues.
    diseases : list[str] | None = None
        Keep only cells with these labels in obs["disease"]. None keeps all diseases.
    gene_id : str = "symbol"
        Gene identifier used as var_names. Either "symbol" or "ensembl".
    protein_coding_only : bool = True
        Keep only genes with var["feature_type"] == "protein_coding".

    Returns
    -------
    Path
        Path of the prepared AnnData file.

    Raises
    ------
    ValueError
        If a required obs column is missing, no cells match the selection or gene_id is invalid.
    """

    if gene_id not in ("symbol", "ensembl"):
        raise ValueError(f"gene_id must be 'symbol' or 'ensembl', not '{gene_id}'.")

    path = Path(path)

    # Open in backed mode to avoid loading cells into memory that are removed afterwards
    backed = ad.read_h5ad(path, backed="r")
    try:
        mask = np.ones(backed.n_obs, dtype=bool)
        for col, selection in (("tissue", tissues), ("disease", diseases)):
            if not selection:
                continue
            if col not in backed.obs.columns:
                raise ValueError(f"Column '{col}' not found in adata.obs.")
            mask &= backed.obs[col].astype(str).isin(selection).to_numpy()

        if not mask.any():
            raise ValueError("No cells match the selected tissues and diseases.")

        var = backed.var
        var_mask = np.ones(backed.n_vars, dtype=bool)
        if "feature_biotype" in var.columns:
            var_mask &= (var["feature_biotype"].astype(str) != "spike-in").to_numpy()

        if protein_coding_only:
            if "feature_type" in var.columns:
                var_mask &= (
                    var["feature_type"].astype(str) == "protein_coding"
                ).to_numpy()
            else:
                warnings.warn(
                    "Column 'feature_type' not found in adata.var. Genes are not restricted to protein coding genes."
                )

        if mask.all() and var_mask.all():
            adata = backed.to_memory()
        else:
            adata = backed[mask, var_mask].to_memory()
    finally:
        backed.file.close()

    if adata.raw is not None:
        adata.X = adata.raw[:, adata.var_names].X
        del adata.raw

    if gene_id == "symbol":
        adata = switch_adata_var_names(adata, "feature_name", "feature_id")
    elif "feature_id" not in adata.var.columns:
        # Name the index, so that switching the annotation later suggests a meaningful column name
        adata.var.index.name = "feature_id"

    tmp_path = path.with_name(path.stem + ".tmp.h5ad")
    try:
        adata.write_h5ad(tmp_path)
        tmp_path.replace(path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise

    return path


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


def write_cellxgene_citation(dataset: dict, ad_path: str | Path) -> Path:
    """
    Write the citation information of a CELLxGENE dataset to a text file.

    The file is stored next to the AnnData file as "<AnnData file name>_citation.txt".
    It contains the meta information of the dataset, the citation provided by
    CELLxGENE Discover and the citation of CZ CELLxGENE Discover itself.

    Parameters
    ----------
    dataset : dict
        Dataset entry as returned by fetch_cellxgene_catalog.
    ad_path : str | Path
        Path of the downloaded AnnData file.

    Returns
    -------
    Path
        Path of the citation file.
    """

    ad_path = Path(ad_path)
    citation_path = ad_path.with_name(f"{ad_path.stem}_citation.txt")

    doi = dataset.get("collection_doi")
    collection_id = dataset.get("collection_id")

    info = [
        ("Dataset", dataset.get("title")),
        ("Dataset ID", dataset.get("dataset_id")),
        ("Collection", dataset.get("collection_name")),
        ("Publication", dataset.get("collection_doi_label")),
        ("DOI", f"https://doi.org/{doi}" if doi else None),
        (
            "Collection URL",
            f"{CELLXGENE_COLLECTIONS_URL}/{collection_id}" if collection_id else None,
        ),
        ("Explorer URL", dataset.get("explorer_url")),
        ("Published", dataset.get("published_at")),
        ("Downloaded", date.today().isoformat()),
    ]

    # Skip entries without information
    lines = ["Citation information of the CELLxGENE Discover dataset", ""]
    lines += [f"{label + ':':<16}{value}" for label, value in info if value]

    if dataset.get("citation"):
        lines += ["", "Citation provided by CELLxGENE Discover:", dataset["citation"]]

    if not doi:
        lines += ["", "No publication DOI is available for this dataset."]

    lines += [
        "",
        "Please cite the original publication of the dataset and CZ CELLxGENE Discover:",
        CELLXGENE_PLATFORM_CITATION,
    ]

    citation_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return citation_path


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


def download_sc_cellxgene(
    dataset_id: str,
    save_dir: str | Path,
    tissues: list[str] | None = None,
    diseases: list[str] | None = None,
    catalog: list[dict] | None = None,
    gene_id: str = "symbol",
    protein_coding_only: bool = True,
) -> Path:
    """
    Download a single cell dataset from CELLxGENE Discover.

    The dataset is stored as "<dataset_id>.h5ad" in save_dir. Afterwards, the
    cells are optionally subset to the given tissues and diseases and raw counts
    stored in adata.raw are moved to adata.X. Spike-ins are removed and optionally
    only protein coding genes are kept. Either gene symbols or Ensembl IDs are used
    as var_names, the other identifier is kept in adata.var. The citation information
    of the dataset is stored as "<dataset_id>_citation.txt" in save_dir.

    Parameters
    ----------
    dataset_id : str
        CELLxGENE dataset ID.
    save_dir : str | Path
        Folder, where the AnnData file will be stored.
    tissues : list[str] | None = None
        Keep only cells of these tissues. None keeps all tissues.
    diseases : list[str] | None = None
        Keep only cells of these diseases. None keeps all diseases.
    catalog : list[dict] | None = None
        Catalog as returned by fetch_cellxgene_catalog. If None, the catalog is
        fetched from the CELLxGENE Discover API.
    gene_id : str = "symbol"
        Gene identifier used as var_names. Either "symbol" or "ensembl".
    protein_coding_only : bool = True
        Keep only protein coding genes.

    Returns
    -------
    Path
        Path of the downloaded AnnData file.

    Raises
    ------
    ValueError
        If the dataset ID is not found in the catalog.
    NotADirectoryError
        If save_dir is not an existing folder.
    """

    save_dir = Path(save_dir).expanduser()
    if not save_dir.is_dir():
        raise NotADirectoryError(f"'{save_dir}' is not an existing folder.")

    if catalog is None:
        catalog = fetch_cellxgene_catalog()

    dataset = next((d for d in catalog if d["dataset_id"] == dataset_id), None)
    if dataset is None:
        raise ValueError(f"Dataset '{dataset_id}' not found on CELLxGENE Discover.")

    save_path = download_file(dataset["url"], save_dir / f"{dataset_id}.h5ad")

    ad_path = prepare_cellxgene_adata(
        save_path,
        tissues=tissues,
        diseases=diseases,
        gene_id=gene_id,
        protein_coding_only=protein_coding_only,
    )
    write_cellxgene_citation(dataset, ad_path)

    return ad_path
