"""
=====================================================
Utility functions for anndata
=====================================================
"""

from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


def get_adata_var_info(adata: ad.AnnData) -> dict[str, list[str]]:
    """
    Collects information on adata.var and processes them.

    Parameters
    ----------
    adata : ad.AnnData
        AnnData dataframe that should be described.

    Returns
    -------
    dict[str, list[str]]
        Dictionary with each entry of adata.var as key. Each key contains a list with first: Number unique values, first unique values
    """

    var_dict = {}

    # Add var_names first, as this is the index
    len_var_names = len(adata.var_names)
    var_names_examples = ", ".join(
        [name for name in adata.var_names[0 : min(5, len_var_names)]]
    )

    var_dict["var_names"] = [str(len_var_names), var_names_examples]

    # Loop through each entry of adata var
    for var in adata.var.columns.unique():
        len_var = len(adata.var[var].unique())
        var_examples = ", ".join(
            [str(name) for name in adata.var[var].unique()[0 : min(5, len_var)]]
        )

        var_dict[var] = [str(len_var), var_examples]

    return var_dict


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


def get_adata_obs_info(adata: ad.AnnData) -> dict[str, list[str]]:
    """
    Collects information on adata.obs and processes them.

    Parameters
    ----------
    adata : ad.AnnData
        AnnData dataframe that should be described.

    Returns
    -------
    dict[str, list[str]]
        Dictionary with each entry of adata.obs as key. Each key contains a list with first: Number unique values, datatype, first unique values
    """

    obs_dict = {}

    # Loop through each entry of adata obs
    for obs in adata.obs.columns.unique():
        len_obs = len(adata.obs[obs].unique())
        obs_examples = ", ".join(
            [str(name) for name in adata.obs[obs].unique()[0 : min(5, len_obs)]]
        )

        obs_dict[obs] = [str(len_obs), obs_examples]

    return obs_dict


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


def get_adata_uns_info(adata: ad.AnnData) -> dict[str, Any]:
    """
    Collects information on adata.uns and processes them.

    Parameters
    ----------
    adata : ad.AnnData
        AnnData dataframe that should be described.

    Returns
    -------
    dict[str, Any]
        Dictionary with each entry of adata.uns as key. Each key contains the corresponding entry.
    """

    uns_dict = {}

    for key in adata.uns.keys():
        uns_dict[key] = adata.uns[key]

    return uns_dict


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


def subset_adata_obs(adata: ad.AnnData, obs_col: str, values: list) -> ad.AnnData:
    """
    Subset a AnnData dataframe, such that a specified observation column only contains certain values.

    Parameters
    ----------
    adata : ad.AnnData
        AnnData dataframe, that should be subsetted.
    obs_col : str
        Name of observation that should be subsetted.
    values : str
        List of values, that should be kept in observation.

    Returns
    -------
    ad.AnnData
        Subsetted AnnData dataframe.
    """

    adata = adata[adata.obs[obs_col].isin(values)]

    return adata


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


def create_annotation_template(
    adata: ad.AnnData,
    obs_col: str,
    example_col_name: str = "new_cell_layer_example",
) -> pd.DataFrame:
    """
    Create a template for adding higher annotation layers.

    Parameters
    ----------
    adata : ad.AnnData
        AnnData dataframe.
    obs_col : str
        Observation column with the sub cell types layer.
    example_col_name : str, default="new_cell_layer_example"
        Name of the example column that is initialized with the same values.

    Returns
    -------
    pd.DataFrame
        Template dataframe with the selected annotation layer and an example column.
    """

    template = pd.DataFrame(adata.obs[obs_col].drop_duplicates()).copy()
    template[example_col_name] = template[obs_col]

    return template


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


def add_annotation_columns_from_template(
    adata: ad.AnnData,
    annotation_df: pd.DataFrame,
    sub_ct_col: str,
) -> tuple[ad.AnnData, list[str]]:
    """
    Add higher annotation layers from an edited template dataframe.

    Parameters
    ----------
    adata : ad.AnnData
        AnnData dataframe that should receive new observation columns.
    annotation_df : pd.DataFrame
        Edited annotation template.
    sub_ct_col : str
        Observation column that matches the original cell type layer.

    Returns
    -------
    tuple[ad.AnnData, list[str]]
        Updated AnnData object and a list with the newly added observation columns.
    """

    if sub_ct_col not in annotation_df.columns:
        raise ValueError(
            f"Cell type column '{sub_ct_col}' is missing from the annotation template."
        )

    template_sub_ct = annotation_df[sub_ct_col].astype(str)
    if template_sub_ct.duplicated().any():
        raise ValueError(
            "The cell type column in the annotation template must contain each cell type only once."
        )

    added_columns = [col for col in annotation_df.columns if col != sub_ct_col]

    updated_adata = adata.copy()
    anchor_values = updated_adata.obs[sub_ct_col].astype(str)
    template_index = annotation_df.set_index(sub_ct_col)

    for column in added_columns:
        mapping = template_index[column].to_dict()
        updated_adata.obs[column] = anchor_values.map(mapping)

    return updated_adata, added_columns


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


def get_var_names_from_column(adata: ad.AnnData, var_col: str) -> pd.Index:
    """
    Get the gene identifiers that result from using a column of adata.var as var_names.

    Missing or empty values are replaced by the current var_names.

    Parameters
    ----------
    adata : ad.AnnData
        AnnData dataframe.
    var_col : str
        Column of adata.var, that should be used as var_names.

    Returns
    -------
    pd.Index
        New gene identifiers in the order of adata.var_names. Can contain duplicates.

    Raises
    ------
    ValueError
        If var_col is not a column of adata.var.
    """

    if var_col not in adata.var.columns:
        raise ValueError(f"Column '{var_col}' not found in adata.var.")

    values = adata.var[var_col].astype(object)
    missing = values.isna() | (values.astype(str).str.strip() == "")

    # Keep the current identifier for genes without a value in var_col
    new_names = values.where(~missing, pd.Series(adata.var_names, index=values.index))

    return pd.Index(new_names.astype(str), name=var_col)


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


def switch_adata_var_names(
    adata: ad.AnnData, var_col: str, old_var_col: str
) -> ad.AnnData:
    """
    Switch the gene annotation used as adata.var_names, e.g. from Ensembl IDs to gene symbols.

    The current var_names are stored in adata.var[old_var_col], the column var_col is
    used as new var_names. Genes that share the same new identifier are combined by summing
    their expression (as done for duplicate gene symbols in bulk data). For combined genes,
    adata.var[old_var_col] contains all previous identifiers separated by ";" and all other
    columns of adata.var contain the value of the first gene.

    Parameters
    ----------
    adata : ad.AnnData
        AnnData dataframe, whose var_names should be switched.
    var_col : str
        Column of adata.var, that should be used as new var_names.
    old_var_col : str
        Name of the column in adata.var, where the current var_names are stored.

    Returns
    -------
    ad.AnnData
        AnnData dataframe with switched var_names.

    Raises
    ------
    ValueError
        If var_col is not a column of adata.var or old_var_col is already used by another column.
    """

    new_names = get_var_names_from_column(adata, var_col)

    if old_var_col == var_col or old_var_col in adata.var.columns:
        raise ValueError(f"Column '{old_var_col}' already exists in adata.var.")

    old_names = pd.Series(adata.var_names.astype(str), index=adata.var_names)
    codes, unique_names = pd.factorize(new_names)

    # No genes have to be combined, exchange only annotation
    if len(unique_names) == adata.n_vars:
        switched = adata.copy()
        switched.var[old_var_col] = old_names.to_numpy()
        switched.var = switched.var.drop(columns=var_col)
        switched.var_names = new_names
        return switched

    indicator = sp.csr_matrix(
        (np.ones(adata.n_vars), (np.arange(adata.n_vars), codes)),
        shape=(adata.n_vars, len(unique_names)),
    )

    def combine(matrix):
        combined = matrix @ indicator.astype(matrix.dtype)
        return sp.csr_matrix(combined) if sp.issparse(matrix) else np.asarray(combined)

    # Keep annotation of the first gene of each new index
    _, first_idx = np.unique(codes, return_index=True)
    var = adata.var.iloc[first_idx].drop(columns=var_col).copy()
    var[old_var_col] = old_names.groupby(codes).agg(";".join).to_numpy()
    var.index = pd.Index(unique_names, name=var_col)

    switched = ad.AnnData(
        X=combine(adata.X),
        obs=adata.obs.copy(),
        var=var,
        uns=adata.uns.copy(),
        obsm=adata.obsm.copy(),
        obsp=adata.obsp.copy(),
        layers={
            key: combine(layer)
            for key, layer in adata.layers.items()
            if key is not None  # Fix for newer AnnData version
        },
    )

    if adata.raw is not None:
        switched.raw = adata.raw.to_adata()

    return switched
