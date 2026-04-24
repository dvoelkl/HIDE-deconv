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


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


@pytest.fixture
def sample_bulks_two_cohorts() -> pd.DataFrame:
    """
    Provides synthetic bulk compositions for two cohorts.

    Returns
    -------
    pd.DataFrame
        Cell type composition matrix (celltypes x samples).
    """

    columns = [f"sample_{i}" for i in range(1, 13)]

    # First six samples are cohort A, last six are cohort B.
    ct_a_high = [0.82, 0.79, 0.81, 0.77, 0.80, 0.78, 0.21, 0.24, 0.23, 0.20, 0.22, 0.19]
    ct_balanced = [
        0.50,
        0.52,
        0.49,
        0.51,
        0.50,
        0.48,
        0.47,
        0.49,
        0.51,
        0.50,
        0.48,
        0.52,
    ]

    return pd.DataFrame(
        [ct_a_high, ct_balanced],
        index=["ct_a_high", "ct_balanced"],
        columns=columns,
    )


@pytest.fixture
def sample_sheet_two_cohorts() -> pd.DataFrame:
    """
    Provides a sample sheet for two cohorts with one additional unrelated sample.

    Returns
    -------
    pd.DataFrame
        Sample metadata table.
    """

    sample_ids = [f"sample_{i}" for i in range(1, 13)] + ["sample_extra"]
    cohorts = ["A"] * 6 + ["B"] * 6 + ["A"]

    return pd.DataFrame(
        {
            "SampleID": sample_ids,
            "Cohort": cohorts,
        }
    )


@pytest.fixture
def sample_bulks_three_cohorts() -> pd.DataFrame:
    """
    Provides synthetic bulk compositions for three cohorts.

    Returns
    -------
    pd.DataFrame
        Cell type composition matrix (celltypes x samples).
    """

    columns = [f"sample_{i}" for i in range(1, 16)]

    # Cohorts are encoded as A (1-5), B (6-10), C (11-15).
    ct_shifted = [
        0.18,
        0.20,
        0.21,
        0.19,
        0.22,
        0.43,
        0.45,
        0.47,
        0.42,
        0.44,
        0.73,
        0.75,
        0.72,
        0.74,
        0.71,
    ]
    ct_stable = [
        0.50,
        0.49,
        0.51,
        0.48,
        0.50,
        0.49,
        0.50,
        0.51,
        0.49,
        0.50,
        0.50,
        0.49,
        0.50,
        0.51,
        0.50,
    ]

    return pd.DataFrame(
        [ct_shifted, ct_stable],
        index=["ct_shifted", "ct_stable"],
        columns=columns,
    )


@pytest.fixture
def sample_sheet_three_cohorts() -> pd.DataFrame:
    """
    Provides a sample sheet for three cohorts.

    Returns
    -------
    pd.DataFrame
        Sample metadata table.
    """

    sample_ids = [f"sample_{i}" for i in range(1, 16)]
    cohorts = ["A"] * 5 + ["B"] * 5 + ["C"] * 5

    return pd.DataFrame(
        {
            "SampleID": sample_ids,
            "Cohort": cohorts,
        }
    )


@pytest.fixture
def sample_survival_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Provides synthetic compositions and clinical data for Cox regression tests.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        Composition matrix and matching sample sheet.
    """

    np.random.seed(42)

    n_samples = 40
    sample_ids = [f"sample_{i}" for i in range(1, n_samples + 1)]

    ct_risk = np.linspace(0.05, 0.95, n_samples)
    ct_background = np.linspace(0.95, 0.05, n_samples)

    bulks = pd.DataFrame(
        [ct_risk, ct_background],
        index=["ct_risk", "ct_background"],
        columns=sample_ids,
    )

    # Survival time decreases with ct_risk to create a clear association.
    survival_time = 30 - (ct_risk * 20) + np.random.normal(0.0, 0.7, n_samples)
    survival_time = np.clip(survival_time, a_min=1.0, a_max=None)

    # Most samples have an observed event.
    event = (np.random.rand(n_samples) > 0.2).astype(int)

    age = np.random.randint(45, 80, size=n_samples)
    sex = np.where(np.random.rand(n_samples) > 0.5, "M", "F")

    sample_sheet = pd.DataFrame(
        {
            "SampleID": sample_ids,
            "time": survival_time,
            "event": event,
            "age": age,
            "sex": sex,
        }
    )

    # Introduce missing values to test automatic filtering.
    sample_sheet.loc[0, "time"] = np.nan
    sample_sheet.loc[1, "age"] = np.nan

    return bulks, sample_sheet
