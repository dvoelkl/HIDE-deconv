"""
=====================================================
Pipeline for preprocessing of AnnData files
=====================================================
"""

import scanpy as sc
import numpy as np
import anndata as ad

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


def preprocess_anndata_file(
    adata: ad.AnnData,
    celltype_col: str,
    min_cell: int = 100,
    min_gene_count: int = 200,
    max_gene_count: int = 5000,
    mt_percentage_per_cell: float = 0.05,
    hb_percentage_per_cell: float = 0.05,
    malat1_percentile_per_cell: float = 0.99,
    exclude_mito_ribo_rna: bool = True,
    exclude_hemoglobine: bool = True,
    filter_low_expressed_genes: bool = True,
    low_expressed_gene_cell_min: int = 10,
) -> ad.AnnData:
    """
    Applies a standard AnnData preprocessing pipeline to a given AnnData file. Removes cells with low quality or high mitochondrial rna expression.
    Additionally excludes celltypes, that are below the min_cell threshold and removes genes that are either ribosomal, mitochondrial or have a very low expression level.

    Parameters:
    ----------
    adata : ad.AnnData
        AnnData single cell file, that will be preprocessed.
    celltype_col : str
        Column of the AnnData file, that holds the cell type labels
    min_cell : int = 100
        Minimum number of cells per cell type to be included.
    min_gene_count : int = 100
        Minimum summed gene count below which a cell is considered as low quality.
    max_gene_count : int = 5000
        Max summed gene count above which a cell is considered to be low quality.
    mt_percentage_per_cell: float = 0.05
        Percentage of mitochondrial gene expression, above which a cell is considered low quality.
    hb_percentage_per_cell: float = 0.05
        Percentage of haemoglobine gene expression, above which a cell is considered low quality.
    malat1_percentile_per_cell: float = 0.99
        Percentile of MALAT1 expression, above which a cell is considered low quality.
    exclude_mito_ribo_rna : bool = True
        Remove genes that are either mitochondrial or ribosomal.
    exclude_hemoglobine: bool = True
        Remove genes that are related to haemoglobine.
    filter_low_expressed_genes : bool = True
        Remove genes that have a low overall expression level and can thus be considered non-informative.
    low_expressed_gene_cell_min : int = 10
        If filter_low_expressed_genes is True => Number of cells in which a gene has to be expressed at least.

    Returns
    -------
    ad.AnnData
        Preprocessed single cells
    """

    # Calculate QC metrics
    adata.var["mt"] = adata.var_names.str.startswith("MT-")
    hb_genes = adata.var_names.str.startswith(
        ("HBA", "HBB", "HBD", "HBE", "HBG", "HBM", "HBQ", "HBZ")
    )
    adata.var["hb"] = hb_genes
    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt", "hb"], inplace=True)

    # create filters
    cell_mask = np.ones(adata.n_obs, dtype=bool)

    cell_mask &= adata.obs["n_genes_by_counts"] >= min_gene_count
    cell_mask &= adata.obs["n_genes_by_counts"] < max_gene_count
    cell_mask &= adata.obs["pct_counts_mt"] < mt_percentage_per_cell
    cell_mask &= adata.obs["pct_counts_hb"] < hb_percentage_per_cell

    if "MALAT1" in adata.var_names:
        malat1_values = adata[:, "MALAT1"].X

        if hasattr(malat1_values, "toarray"):
            malat1_values = malat1_values.toarray().flatten()
        else:
            malat1_values = malat1_values.flatten()

        malat1_thresh = np.quantile(malat1_values, malat1_percentile_per_cell)
        cell_mask &= malat1_values < malat1_thresh

    ct_counts = adata.obs[celltype_col].value_counts()
    ct_to_keep = set(ct_counts[ct_counts >= min_cell].index)
    cell_mask &= adata.obs[celltype_col].isin(ct_to_keep)

    # Apply filters
    adata = adata[cell_mask].copy()

    gene_mask = np.ones(adata.n_vars, dtype=bool)
    gene_names = adata.var_names

    if exclude_mito_ribo_rna:
        gene_mask &= ~(gene_names.str.startswith(("MT-", "RPS", "RPL")))

    if exclude_hemoglobine:
        gene_mask &= ~(
            gene_names.str.startswith(
                ("HBA", "HBB", "HBD", "HBE", "HBG", "HBM", "HBQ", "HBZ")
            )
        )

    adata = adata[:, gene_mask].copy()

    if filter_low_expressed_genes:
        sc.pp.filter_genes(adata, min_cells=low_expressed_gene_cell_min)

    return adata
