"""
=====================================================
Functions for Clustering
=====================================================
"""

import pandas as pd
import scanpy as sp
import networkx as nx
import scipy.sparse as sps

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


def run_clustering(data: pd.DataFrame, is_bulk: bool = False) -> pd.DataFrame:
    """
    Performs a clustering of the entered bulk or composition data.

    Parameters
    ----------
    data : pd.DataFrame
        Dataframe containing either bulks or composition (celltype/genes x samples)
    is_bulk : bool = False
        If set to true, uses correlation as distance measure for the neighborhood calculation.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the sample ids and the assigned clusters

    """

    # AnnData
    obs = pd.DataFrame(index=data.columns)
    obs["id"] = data.columns
    var = pd.DataFrame(index=data.index)
    adata = sp.AnnData(X=data.T.to_numpy(), obs=obs, var=var)

    if is_bulk:
        sp.pp.neighbors(adata, metric="correlation")
    else:
        sp.pp.neighbors(adata, metric="braycurtis")

    # Sparse arrays have distances instead of connectivities property
    if "connectivities" in adata.obsp:
        conn = adata.obsp["connectivities"]
    elif "distances" in adata.obsp:
        conn = adata.obsp["distances"]
    else:
        raise ValueError("Calculation of neighborhood graph has failed!")

    if not sps.issparse(conn):
        conn = sps.csr_matrix(conn)

    coo = conn.tocoo()
    G = nx.Graph()
    for u, v, w in zip(coo.row, coo.col, coo.data):
        if w != 0:
            G.add_edge(int(u), int(v), weight=float(w))

    if G.number_of_nodes() == 0:
        labels = []
    else:
        communities = nx.algorithms.community.greedy_modularity_communities(G)
        labels = [None] * G.number_of_nodes()
        for i, comm in enumerate(communities):
            for node in comm:
                labels[node] = i

    # Return DataFrame with sample ids and assigned cluster
    return pd.DataFrame({"id": list(adata.obs_names), "assigned_cluster": labels})
