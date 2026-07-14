"""
=====================================================
Utility functions for mtx files
=====================================================
"""

from scipy.io import mmread
import pandas as pd
import anndata as ad
from pathlib import Path


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def mtx_to_csv(mtx_path: str, barcodes_path: str, genes_path: str) -> pd.DataFrame:
    """
    Read a mtx file and convert it to a pandas DataFrame

    Parameters
    -----------
    mtx_path : str
        Path to mtx file
    barcodes_path : str
        Path to mtx barcodes
    gene_path : str
        Path to mtx genes

    Returns
    --------
    pd.DataFrame
        Dataframe containing the converted mtx file

    """

    # Read barcodes, separate between tsv and csv table
    if Path(barcodes_path).suffix == ".tsv":
        barcodes = pd.read_csv(barcodes_path, header=None, sep="\t")
    elif Path(barcodes_path).suffix == ".csv":
        barcodes = pd.read_csv(barcodes_path, header=None)
    else:
        raise ValueError("Barcode file must either be a csv or tsv table.")

    # Read genes
    if Path(genes_path).suffix == ".tsv":
        genes = pd.read_csv(genes_path, header=None, sep="\t")
    elif Path(genes_path).suffix == ".csv":
        genes = pd.read_csv(genes_path, header=None)
    else:
        raise ValueError("Features file must either be a csv or tsv table.")

    # Read matrix
    data_mtx = mmread(mtx_path)
    data_df = pd.DataFrame.sparse.from_spmatrix(
        data_mtx
    )  # , index=genes.iloc[:,0], columns=barcodes.iloc[:,0])
    data_df = data_df.sparse.to_dense()

    data_df = data_df.set_index(genes.iloc[:, 0])
    data_df.columns = barcodes.iloc[:, 0]

    return data_df


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def mtx_to_adata(mtx_path: str, barcodes_path: str, genes_path: str) -> ad.AnnData:
    """
    Read a mtx file and convert it to a anndata

    Parameters
    -----------
    mtx_path : str
        Path to mtx file
    barcodes_path : str
        Path to mtx barcodes
    gene_path : str
        Path to mtx genes

    Returns
    --------
    ad.AnnData
        AnnData object containing the converted mtx file

    """

    # Read barcodes
    if Path(barcodes_path).suffix == ".tsv":
        barcodes = pd.read_csv(barcodes_path, header=None, sep="\t")
    elif Path(barcodes_path).suffix == ".csv":
        barcodes = pd.read_csv(barcodes_path, header=None)
    else:
        raise ValueError("Barcode file must either be a csv or tsv table.")

    # Read genes
    if Path(genes_path).suffix == ".tsv":
        genes = pd.read_csv(genes_path, header=None, sep="\t")
    elif Path(genes_path).suffix == ".csv":
        genes = pd.read_csv(genes_path, header=None)
    else:
        raise ValueError("Features file must either be a csv or tsv table.")

    # Read matrix
    data_mtx = mmread(mtx_path).tocsr()
    X = data_mtx.T

    obs = pd.DataFrame(index=barcodes.iloc[:, 0].astype(str))
    obs.index.name = "barcode"

    var = pd.DataFrame(index=genes.iloc[:, 0].astype(str))
    var.index.name = "gene"

    adata = ad.AnnData(
        X=X,
        obs=obs,
        var=var,
    )

    return adata
