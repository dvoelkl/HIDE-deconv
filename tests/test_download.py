"""
=====================================================
Tests for the CELLxGENE download API and command
=====================================================
"""

import importlib

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import requests
from click.testing import CliRunner

import hide_deconv.cli_commands as cli_commands
import hide_deconv.cli_commands.download_command as download_command
from hide_deconv.cli import cli
from hide_deconv.constants import (
    CELLXGENE_PLATFORM_CITATION,
    MSG_FAILURE,
    MSG_SUCCESS,
    MSG_USER_ABORT,
)
from hide_deconv.download import (
    count_by,
    download_file,
    download_sc_cellxgene,
    fetch_cellxgene_catalog,
    filter_datasets,
    get_h5ad_asset,
    prepare_cellxgene_adata,
    write_cellxgene_citation,
)

# The module is shadowed by the function download_file in hide_deconv.download
download_file_module = importlib.import_module("hide_deconv.download.download_file")

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


def api_entry(
    dataset_id,
    organism,
    tissues,
    diseases,
    cell_count=100,
    h5ad=True,
    tombstone=False,
    doi="10.1000/test",
) -> dict:
    """
    Create a dataset entry in the format of the CELLxGENE Discover API.
    """

    assets = [{"filetype": "RDS", "url": f"https://x/{dataset_id}.rds", "filesize": 1}]
    if h5ad:
        assets.append(
            {
                "filetype": "H5AD",
                "url": f"https://x/{dataset_id}.h5ad",
                "filesize": 2_000_000,
            }
        )

    return {
        "dataset_id": dataset_id,
        "title": f"Dataset {dataset_id}",
        "collection_name": "Collection",
        "organism": [{"label": organism, "ontology_term_id": "NCBITaxon:0"}],
        "tissue": [{"label": t, "ontology_term_id": "UBERON:0"} for t in tissues],
        "disease": [{"label": d, "ontology_term_id": "MONDO:0"} for d in diseases],
        "assay": [{"label": "10x 3' v3", "ontology_term_id": "EFO:0"}],
        "suspension_type": ["cell"],
        "cell_count": cell_count,
        "published_at": "2025-01-31T10:00:00+00:00",
        "tombstone": tombstone,
        "assets": assets,
        "collection_id": f"collection_{dataset_id}",
        "collection_doi": doi,
        "collection_doi_label": "Doe et al. (2024) J Test" if doi else None,
        "citation": f"Publication: https://doi.org/{doi} Dataset Version: https://x/{dataset_id}.h5ad"
        if doi
        else None,
        "explorer_url": f"https://cellxgene.cziscience.com/e/{dataset_id}.cxg/",
    }


API_RESPONSE = [
    api_entry("d1", "Homo sapiens", ["lung"], ["normal"], cell_count=300),
    api_entry("d2", "Homo sapiens", ["lung", "liver"], ["normal", "COPD"], 500),
    api_entry("d3", "Homo sapiens", ["liver"], ["hepatitis"], cell_count=200, doi=None),
    api_entry("d4", "Mus musculus", ["lung"], ["normal"], cell_count=50),
    api_entry("d5", "Homo sapiens", ["lung"], ["normal"], h5ad=False),
    api_entry("d6", "Homo sapiens", ["lung"], ["normal"], tombstone=True),
]


class MockResponse:
    """
    Minimal replacement for requests.Response.
    """

    def __init__(self, json_data=None, content=b"", status=200, headers=None):
        self.json_data = json_data
        self.content = content
        self.status = status
        self.headers = headers if headers is not None else {}

    def json(self):
        return self.json_data

    def raise_for_status(self):
        if self.status >= 400:
            raise requests.HTTPError(f"{self.status} Error")

    def iter_content(self, chunk_size=1):
        for i in range(0, len(self.content), chunk_size):
            yield self.content[i : i + chunk_size]

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


@pytest.fixture
def catalog(monkeypatch) -> list[dict]:
    """
    Catalog created from the mocked API response.
    """

    monkeypatch.setattr(
        download_file_module.requests,
        "get",
        lambda *args, **kwargs: MockResponse(json_data=API_RESPONSE),
    )

    return fetch_cellxgene_catalog()


def create_cellxgene_anndata(with_raw: bool = True) -> ad.AnnData:
    """
    Create a small AnnData object in the CELLxGENE format.
    """

    obs = pd.DataFrame(
        {
            "tissue": pd.Categorical(["lung", "liver", "lung", "lung"]),
            "disease": pd.Categorical(["normal", "normal", "COPD", "normal"]),
            "cell_type": ["a", "b", "a", "b"],
        },
        index=[f"cell_{i}" for i in range(4)],
    )
    var = pd.DataFrame(
        {
            "feature_name": ["g1", "g2", "g3"],
            "feature_biotype": ["gene", "gene", "gene"],
            "feature_type": ["protein_coding", "protein_coding", "protein_coding"],
        },
        index=["ENSG1", "ENSG2", "ENSG3"],
    )
    counts = np.arange(12, dtype=np.float32).reshape(4, 3)

    adata = ad.AnnData(X=np.log1p(counts), obs=obs, var=var)
    if with_raw:
        adata.raw = ad.AnnData(X=counts, obs=obs, var=var)

    return adata


def create_cellxgene_gene_anndata() -> ad.AnnData:
    """
    Create a small AnnData object in the CELLxGENE format with different gene types.

    Genes: ENSG1 (g1), ENSG2 (g2), ENSG3 (g1, duplicate symbol), ENSG4 (lncRNA) and a spike-in.
    """

    obs = pd.DataFrame(
        {
            "tissue": pd.Categorical(["lung", "lung"]),
            "disease": pd.Categorical(["normal", "normal"]),
        },
        index=["cell_0", "cell_1"],
    )
    var = pd.DataFrame(
        {
            "feature_name": ["g1", "g2", "g1", "l1", "ERCC-00002"],
            "feature_biotype": ["gene", "gene", "gene", "gene", "spike-in"],
            "feature_type": [
                "protein_coding",
                "protein_coding",
                "protein_coding",
                "lncRNA",
                "synthetic",
            ],
        },
        index=["ENSG1", "ENSG2", "ENSG3", "ENSG4", "ERCC-00002"],
    )
    counts = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]], dtype=np.float32)

    return ad.AnnData(X=counts, obs=obs, var=var)


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


class TestCatalog:
    """
    Tests for fetching and filtering the CELLxGENE catalog.
    """

    def test_get_h5ad_asset(self) -> None:
        assert get_h5ad_asset(API_RESPONSE[0]) == ("https://x/d1.h5ad", 2_000_000)
        assert get_h5ad_asset(API_RESPONSE[4]) is None

    def test_fetch_skips_datasets_without_h5ad_and_tombstoned(self, catalog) -> None:
        assert [d["dataset_id"] for d in catalog] == ["d1", "d2", "d3", "d4"]

        d2 = catalog[1]
        assert d2["tissue"] == ["lung", "liver"]
        assert d2["disease"] == ["normal", "COPD"]
        assert d2["url"] == "https://x/d2.h5ad"
        assert d2["published_at"] == "2025-01-31"
        assert d2["collection_doi"] == "10.1000/test"
        assert d2["collection_id"] == "collection_d2"

        # Missing citation information is stored as empty string
        assert catalog[2]["collection_doi"] == ""
        assert catalog[2]["citation"] == ""

    def test_fetch_raises_without_internet(self, monkeypatch) -> None:
        def raise_connection_error(*args, **kwargs):
            raise requests.ConnectionError()

        monkeypatch.setattr(
            download_file_module.requests, "get", raise_connection_error
        )

        with pytest.raises(requests.ConnectionError):
            fetch_cellxgene_catalog()

    def test_fetch_raises_on_http_error(self, monkeypatch) -> None:
        monkeypatch.setattr(
            download_file_module.requests,
            "get",
            lambda *args, **kwargs: MockResponse(status=503),
        )

        with pytest.raises(requests.HTTPError):
            fetch_cellxgene_catalog()

    def test_count_by_sorts_by_count(self, catalog) -> None:
        assert count_by(catalog, "organism") == {"Homo sapiens": 3, "Mus musculus": 1}
        assert count_by(catalog, "tissue") == {"lung": 3, "liver": 2}

    def test_count_by_only_counts_preselection(self, catalog) -> None:
        human_lung = filter_datasets(catalog, "Homo sapiens", tissues=["lung"])
        assert count_by(human_lung, "disease") == {"normal": 2, "COPD": 1}

    def test_filter_single_values(self, catalog) -> None:
        result = filter_datasets(
            catalog, "Homo sapiens", tissues=["lung"], diseases=["normal"]
        )
        assert [d["dataset_id"] for d in result] == ["d1", "d2"]

    def test_filter_multiple_values(self, catalog) -> None:
        result = filter_datasets(
            catalog, "Homo sapiens", tissues=["liver"], diseases=["COPD", "hepatitis"]
        )
        assert [d["dataset_id"] for d in result] == ["d2", "d3"]

    def test_filter_none_matches_all(self, catalog) -> None:
        assert len(filter_datasets(catalog)) == 4
        assert len(filter_datasets(catalog, "Homo sapiens", None, None)) == 3

    def test_filter_without_match(self, catalog) -> None:
        assert filter_datasets(catalog, "Mus musculus", diseases=["COPD"]) == []


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


class TestDownloadFile:
    """
    Tests for downloading files.
    """

    def test_download_success(self, monkeypatch, tmp_path) -> None:
        content = b"x" * 5000
        monkeypatch.setattr(
            download_file_module.requests,
            "get",
            lambda *args, **kwargs: MockResponse(
                content=content, headers={"Content-Length": str(len(content))}
            ),
        )

        path = download_file("https://x/file.h5ad", tmp_path / "file.h5ad")

        assert path.read_bytes() == content
        assert not (tmp_path / "file.h5ad.part").exists()

    def test_download_without_content_length(self, monkeypatch, tmp_path) -> None:
        monkeypatch.setattr(
            download_file_module.requests,
            "get",
            lambda *args, **kwargs: MockResponse(content=b"abc"),
        )

        path = download_file("https://x/file.h5ad", tmp_path / "file.h5ad")

        assert path.read_bytes() == b"abc"

    def test_download_http_error_leaves_no_file(self, monkeypatch, tmp_path) -> None:
        monkeypatch.setattr(
            download_file_module.requests,
            "get",
            lambda *args, **kwargs: MockResponse(status=404),
        )

        with pytest.raises(requests.HTTPError):
            download_file("https://x/file.h5ad", tmp_path / "file.h5ad")

        assert list(tmp_path.iterdir()) == []

    def test_download_interrupted_removes_part_file(
        self, monkeypatch, tmp_path
    ) -> None:
        class InterruptedResponse(MockResponse):
            def iter_content(self, chunk_size=1):
                yield b"abc"
                raise requests.ConnectionError()

        monkeypatch.setattr(
            download_file_module.requests,
            "get",
            lambda *args, **kwargs: InterruptedResponse(),
        )

        with pytest.raises(requests.ConnectionError):
            download_file("https://x/file.h5ad", tmp_path / "file.h5ad")

        assert list(tmp_path.iterdir()) == []


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


class TestPrepareAdata:
    """
    Tests for preparing downloaded CELLxGENE AnnData files.
    """

    def test_raw_counts_are_moved_to_x(self, tmp_path) -> None:
        path = tmp_path / "data.h5ad"
        create_cellxgene_anndata().write_h5ad(path)

        prepare_cellxgene_adata(path)

        adata = ad.read_h5ad(path)
        assert adata.raw is None
        assert adata.n_obs == 4
        np.testing.assert_array_equal(
            adata.X, np.arange(12, dtype=np.float32).reshape(4, 3)
        )
        assert list(adata.var_names) == ["g1", "g2", "g3"]
        assert list(adata.var["feature_id"]) == ["ENSG1", "ENSG2", "ENSG3"]
        assert not (tmp_path / "data.tmp.h5ad").exists()

    def test_without_raw_keeps_x(self, tmp_path) -> None:
        path = tmp_path / "data.h5ad"
        original = create_cellxgene_anndata(with_raw=False)
        original.write_h5ad(path)

        prepare_cellxgene_adata(path)

        adata = ad.read_h5ad(path)
        np.testing.assert_allclose(adata.X, original.X)

    def test_subset_tissues_and_diseases(self, tmp_path) -> None:
        path = tmp_path / "data.h5ad"
        create_cellxgene_anndata().write_h5ad(path)

        prepare_cellxgene_adata(path, tissues=["lung"], diseases=["normal", "x"])

        adata = ad.read_h5ad(path)
        assert list(adata.obs_names) == ["cell_0", "cell_3"]
        np.testing.assert_array_equal(adata.X, np.array([[0, 1, 2], [9, 10, 11]]))

    def test_subset_without_matching_cells(self, tmp_path) -> None:
        path = tmp_path / "data.h5ad"
        create_cellxgene_anndata().write_h5ad(path)

        with pytest.raises(ValueError):
            prepare_cellxgene_adata(path, tissues=["kidney"])

    def test_symbol_sums_duplicate_genes(self, tmp_path) -> None:
        path = tmp_path / "data.h5ad"
        create_cellxgene_gene_anndata().write_h5ad(path)

        prepare_cellxgene_adata(path, gene_id="symbol")

        adata = ad.read_h5ad(path)
        assert list(adata.var_names) == ["g1", "g2"]
        assert list(adata.var["feature_id"]) == ["ENSG1;ENSG3", "ENSG2"]
        assert "feature_name" not in adata.var.columns
        np.testing.assert_array_equal(adata.X, np.array([[4, 2], [14, 7]]))

    def test_ensembl_keeps_feature_name(self, tmp_path) -> None:
        path = tmp_path / "data.h5ad"
        create_cellxgene_gene_anndata().write_h5ad(path)

        prepare_cellxgene_adata(path, gene_id="ensembl")

        adata = ad.read_h5ad(path)
        assert list(adata.var_names) == ["ENSG1", "ENSG2", "ENSG3"]
        assert adata.var.index.name == "feature_id"
        assert list(adata.var["feature_name"]) == ["g1", "g2", "g1"]
        np.testing.assert_array_equal(adata.X, np.array([[1, 2, 3], [6, 7, 8]]))

    def test_spike_ins_removed_without_protein_coding_filter(self, tmp_path) -> None:
        path = tmp_path / "data.h5ad"
        create_cellxgene_gene_anndata().write_h5ad(path)

        prepare_cellxgene_adata(path, gene_id="ensembl", protein_coding_only=False)

        adata = ad.read_h5ad(path)
        assert list(adata.var_names) == ["ENSG1", "ENSG2", "ENSG3", "ENSG4"]

    def test_missing_feature_type_warns(self, tmp_path) -> None:
        path = tmp_path / "data.h5ad"
        original = create_cellxgene_gene_anndata()
        original.var = original.var.drop(columns="feature_type")
        original.write_h5ad(path)

        with pytest.warns(UserWarning, match="feature_type"):
            prepare_cellxgene_adata(path, gene_id="ensembl")

        adata = ad.read_h5ad(path)
        assert list(adata.var_names) == ["ENSG1", "ENSG2", "ENSG3", "ENSG4"]

    def test_invalid_gene_id(self, tmp_path) -> None:
        path = tmp_path / "data.h5ad"
        create_cellxgene_gene_anndata().write_h5ad(path)

        with pytest.raises(ValueError):
            prepare_cellxgene_adata(path, gene_id="entrez")


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


def mock_download(monkeypatch, captured: dict) -> None:
    """
    Replace the file download by writing a CELLxGENE-like AnnData file.
    """

    def fake_download_file(url, save_path):
        captured["url"] = url
        create_cellxgene_anndata().write_h5ad(save_path)
        return save_path

    monkeypatch.setattr(download_file_module, "download_file", fake_download_file)


class TestCitation:
    """
    Tests for writing the citation information of a dataset.
    """

    def test_citation_file_contains_dataset_information(
        self, tmp_path, catalog
    ) -> None:
        path = write_cellxgene_citation(catalog[0], tmp_path / "d1.h5ad")

        assert path == tmp_path / "d1_citation.txt"
        text = path.read_text(encoding="utf-8")
        assert "Dataset:        Dataset d1" in text
        assert "Dataset ID:     d1" in text
        assert "Publication:    Doe et al. (2024) J Test" in text
        assert "DOI:            https://doi.org/10.1000/test" in text
        assert (
            "Collection URL: https://cellxgene.cziscience.com/collections/collection_d1"
            in text
        )
        assert "Citation provided by CELLxGENE Discover:" in text
        assert "Publication: https://doi.org/10.1000/test" in text
        assert "Downloaded:" in text
        assert CELLXGENE_PLATFORM_CITATION in text
        assert "No publication DOI" not in text

    def test_citation_file_without_doi(self, tmp_path, catalog) -> None:
        path = write_cellxgene_citation(catalog[2], tmp_path / "d3.h5ad")

        text = path.read_text(encoding="utf-8")
        assert "DOI:" not in text
        assert "Publication:" not in text
        assert "Citation provided by CELLxGENE Discover:" not in text
        assert "No publication DOI is available for this dataset." in text
        assert CELLXGENE_PLATFORM_CITATION in text


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


class TestDownloadScCellxgene:
    """
    Tests for the Python API function download_sc_cellxgene.
    """

    def test_download_and_subset(self, monkeypatch, tmp_path, catalog) -> None:
        captured = {}
        mock_download(monkeypatch, captured)

        path = download_sc_cellxgene("d2", tmp_path, tissues=["liver"], catalog=catalog)

        assert path == tmp_path / "d2.h5ad"
        assert captured["url"] == "https://x/d2.h5ad"
        assert (tmp_path / "d2_citation.txt").exists()
        adata = ad.read_h5ad(path)
        assert list(adata.obs["tissue"]) == ["liver"]
        assert adata.raw is None

    def test_fetches_catalog_if_not_given(self, monkeypatch, tmp_path) -> None:
        captured = {}
        mock_download(monkeypatch, captured)
        monkeypatch.setattr(
            download_file_module.requests,
            "get",
            lambda *args, **kwargs: MockResponse(json_data=API_RESPONSE),
        )

        path = download_sc_cellxgene("d1", tmp_path)

        assert path.exists()
        assert ad.read_h5ad(path).n_obs == 4

    def test_gene_options_are_forwarded(self, monkeypatch, tmp_path, catalog) -> None:
        mock_download(monkeypatch, {})

        path = download_sc_cellxgene(
            "d1",
            tmp_path,
            catalog=catalog,
            gene_id="ensembl",
            protein_coding_only=False,
        )

        adata = ad.read_h5ad(path)
        assert list(adata.var_names) == ["ENSG1", "ENSG2", "ENSG3"]

    def test_unknown_dataset(self, tmp_path, catalog) -> None:
        with pytest.raises(ValueError):
            download_sc_cellxgene("unknown", tmp_path, catalog=catalog)

    def test_invalid_folder(self, tmp_path, catalog) -> None:
        with pytest.raises(NotADirectoryError):
            download_sc_cellxgene("d1", tmp_path / "missing", catalog=catalog)


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


class MockPrompt:
    """
    Replacement for InquirerPy prompts returning predefined answers.
    """

    def __init__(self, answers: list):
        self.answers = answers

    def __call__(self, *args, **kwargs):
        self.kwargs = kwargs
        return self

    def execute(self):
        return self.answers.pop(0)


class TestDownloadCommand:
    """
    Tests for the interactive download command.
    """

    def setup_prompts(self, monkeypatch, fuzzy, confirm, filepath, select=None) -> None:
        select = ["symbol"] if select is None else select
        monkeypatch.setattr(download_command.inquirer, "fuzzy", MockPrompt(fuzzy))
        monkeypatch.setattr(download_command.inquirer, "confirm", MockPrompt(confirm))
        monkeypatch.setattr(download_command.inquirer, "filepath", MockPrompt(filepath))
        monkeypatch.setattr(download_command.inquirer, "select", MockPrompt(select))
        monkeypatch.setattr(
            download_command.console, "print_exception", lambda *args, **kwargs: None
        )

    def test_command_downloads_selected_dataset(
        self, monkeypatch, tmp_path, catalog
    ) -> None:
        monkeypatch.setattr(
            download_command, "fetch_cellxgene_catalog", lambda: catalog
        )

        captured = {}

        def fake_download_sc_cellxgene(dataset_id, save_dir, **kwargs):
            captured["dataset_id"] = dataset_id
            captured.update(kwargs)
            return tmp_path / f"{dataset_id}.h5ad"

        monkeypatch.setattr(
            download_command, "download_sc_cellxgene", fake_download_sc_cellxgene
        )

        # Organism, tissues, diseases, dataset index (sorted by cell count: d2, d1)
        self.setup_prompts(
            monkeypatch,
            fuzzy=["Homo sapiens", ["lung"], [download_command.ALL_VALUES], 1],
            confirm=[True, True],
            filepath=[str(tmp_path)],
        )

        result = download_command.download_cellxgene_command()

        assert result == MSG_SUCCESS
        assert captured["dataset_id"] == "d1"
        assert captured["tissues"] == ["lung"]
        assert captured["diseases"] is None
        assert captured["catalog"] is catalog
        assert captured["gene_id"] == "symbol"
        assert captured["protein_coding_only"] is True

    def test_command_forwards_gene_options(
        self, monkeypatch, tmp_path, catalog
    ) -> None:
        monkeypatch.setattr(
            download_command, "fetch_cellxgene_catalog", lambda: catalog
        )

        captured = {}
        monkeypatch.setattr(
            download_command,
            "download_sc_cellxgene",
            lambda dataset_id, save_dir, **kwargs: captured.update(kwargs) or "x",
        )

        self.setup_prompts(
            monkeypatch,
            fuzzy=["Homo sapiens", [download_command.ALL_VALUES], ["COPD"], 0],
            confirm=[True, False],
            filepath=[str(tmp_path)],
            select=["ensembl"],
        )

        result = download_command.download_cellxgene_command(protein_coding_only=False)

        assert result == MSG_SUCCESS
        assert captured["gene_id"] == "ensembl"
        assert captured["protein_coding_only"] is False

    @pytest.mark.parametrize("args, expected", [([], True), (["-pc", "False"], False)])
    def test_cli_protein_coding_option(self, monkeypatch, args, expected) -> None:
        captured = {}
        monkeypatch.setattr(
            cli_commands,
            "download_cellxgene_command",
            lambda protein_coding_only: captured.update(value=protein_coding_only),
        )

        result = CliRunner().invoke(cli, ["download", *args])

        assert result.exit_code == 0
        assert captured["value"] is expected

    def test_command_without_subset(self, monkeypatch, tmp_path, catalog) -> None:
        monkeypatch.setattr(
            download_command, "fetch_cellxgene_catalog", lambda: catalog
        )

        captured = {}
        monkeypatch.setattr(
            download_command,
            "download_sc_cellxgene",
            lambda dataset_id, save_dir, **kwargs: captured.update(kwargs) or "x",
        )

        self.setup_prompts(
            monkeypatch,
            fuzzy=["Homo sapiens", ["lung"], ["COPD"], 0],
            confirm=[True, False],
            filepath=[str(tmp_path)],
        )

        assert download_command.download_cellxgene_command() == MSG_SUCCESS
        assert captured["tissues"] is None
        assert captured["diseases"] is None

    def test_command_change_filters_and_cancel(
        self, monkeypatch, tmp_path, catalog
    ) -> None:
        monkeypatch.setattr(
            download_command, "fetch_cellxgene_catalog", lambda: catalog
        )

        self.setup_prompts(
            monkeypatch,
            fuzzy=[
                "Homo sapiens",
                [download_command.ALL_VALUES],
                [download_command.ALL_VALUES],
                download_command.CHANGE_FILTERS,
                ["liver"],
                ["hepatitis"],
                download_command.CANCEL,
            ],
            confirm=[],
            filepath=[],
        )

        assert download_command.download_cellxgene_command() == MSG_USER_ABORT

    def test_command_without_internet(self, monkeypatch) -> None:
        def raise_connection_error():
            raise requests.ConnectionError()

        monkeypatch.setattr(
            download_command, "fetch_cellxgene_catalog", raise_connection_error
        )
        printed = []
        monkeypatch.setattr(
            download_command.console, "print", lambda msg, *a, **k: printed.append(msg)
        )

        assert download_command.download_cellxgene_command() == MSG_FAILURE
        assert any("Internet access is necessary" in str(m) for m in printed)
