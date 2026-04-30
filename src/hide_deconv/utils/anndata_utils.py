"""
=====================================================
Utility functions for anndata
=====================================================
"""

import anndata as ad
import pandas as pd


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


def get_adata_uns_info(adata: ad.AnnData) -> dict[str, str]:
    """
    Collects information on adata.uns and processes them.

    Parameters
    ----------
    adata : ad.AnnData
        AnnData dataframe that should be described.

    Returns
    -------
    dict[str, list[str]]
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
