"""
=====================================================
Shared fixtures for all tests
=====================================================
"""

import pytest
import pandas as pd
import numpy as np

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


@pytest.fixture
def sample_bulk_data() -> pd.DataFrame:
    """
    Provides sample bulk RNA-seq data for testing.

    Returns
    -------
    pd.DataFrame
        Sample bulk data with genes as rows and samples as columns.
    """
    np.random.seed(42)

    n_genes = 100
    n_samples = 50

    data = np.random.poisson(10, size=(n_genes, n_samples)) + 1

    bulk_df = pd.DataFrame(
        data,
        index=[f"gene_{i}" for i in range(1, n_genes + 1)],
        columns=[f"sample_{i}" for i in range(1, n_samples + 1)],
    )

    return bulk_df


@pytest.fixture
def sample_composition_data() -> pd.DataFrame:
    """
    Provides sample cell type composition data for testing.

    Returns
    -------
    pd.DataFrame
        Sample composition data with cell types as rows and samples as columns.
    """
    np.random.seed(42)

    n_celltypes = 10
    n_samples = 50

    # Generate random compositions that sum to 1
    compositions = np.random.dirichlet(np.ones(n_celltypes), n_samples).T

    comp_df = pd.DataFrame(
        compositions,
        index=[f"celltype_{i}" for i in range(1, n_celltypes + 1)],
        columns=[f"sample_{i}" for i in range(1, n_samples + 1)],
    )

    return comp_df


@pytest.fixture
def sample_reference_data() -> pd.DataFrame:
    """
    Provides sample single-cell reference data for testing.

    Returns
    -------
    pd.DataFrame
        Sample reference data with genes as rows and cell types as columns.
    """
    np.random.seed(42)

    n_genes = 100
    n_celltypes = 10

    # Create reference profiles
    data = np.random.poisson(5, size=(n_genes, n_celltypes)) + 1

    for ct in range(n_celltypes):
        gene_indices = np.random.choice(n_genes, size=20, replace=False)
        data[gene_indices, ct] = np.random.poisson(30, size=20) + 10

    ref_df = pd.DataFrame(
        data,
        index=[f"gene_{i}" for i in range(1, n_genes + 1)],
        columns=[f"celltype_{i}" for i in range(1, n_celltypes + 1)],
    )

    return ref_df


@pytest.fixture
def sample_gene_weights() -> list:
    """
    Provides sample gene weight matrices for testing.

    Returns
    -------
    list
        List of gene weight vectors for testing.
    """
    np.random.seed(42)

    n_genes = 100

    weights = np.random.uniform(0.5, 1.5, size=n_genes)
    weights = weights / np.sum(weights) * n_genes

    # G_l sollte ein Vektor sein, nicht eine Diagonalmatrix!
    gene_weight_df = pd.DataFrame(
        weights,
        index=[f"gene_{i}" for i in range(1, n_genes + 1)],
        columns=["weight"],
    )

    return [gene_weight_df]


@pytest.fixture
def sample_hierarchy() -> list:
    """
    Provides sample hierarchy matrices for testing.

    Returns
    -------
    list
        List of hierarchy matrices.
    """

    hierarchy_df = pd.DataFrame(
        np.eye(10),
        index=[f"celltype_{i}" for i in range(1, 11)],
        columns=[f"celltype_{i}" for i in range(1, 11)],
    )

    return [hierarchy_df]
