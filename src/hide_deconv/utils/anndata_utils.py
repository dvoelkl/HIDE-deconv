"""
=====================================================
Utility functions for anndata
=====================================================
"""

import anndata as ad


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
