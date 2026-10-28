"""
=====================================================
Functions for creating synthetic data
=====================================================
"""

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Settings for the synthetic datasets

EXAMPLE_HIERARCHY = {
    "T_CD4": "T",
    "T_CD8": "T",
    "B_naive": "B",
    "B_memory": "B",
    "Mono_classical": "Mono",
    "Mono_nonclassical": "Mono",
}


EXAMPLE_SC_FRACTIONS = {
    "T_CD4": 0.25,
    "T_CD8": 0.20,
    "B_naive": 0.12,
    "B_memory": 0.10,
    "Mono_classical": 0.23,
    "Mono_nonclassical": 0.10,
}


EXAMPLE_LIBRARY_SIZES = {
    "T_CD4": 2500,
    "T_CD8": 2200,
    "B_naive": 1800,
    "B_memory": 2000,
    "Mono_classical": 4000,
    "Mono_nonclassical": 3500,
}


EXAMPLE_HEALTHY_COMPOSITION = {
    "T_CD4": 0.30,
    "T_CD8": 0.20,
    "B_naive": 0.10,
    "B_memory": 0.08,
    "Mono_classical": 0.25,
    "Mono_nonclassical": 0.07,
}

EXAMPLE_DISEASE_EFFECT = {"T_CD8": 0.5, "Mono_classical": 1.8}
EXAMPLE_MEDICATION_EFFECT = {
    "Drug_A": {"B_memory": 2.0},
    "Drug_B": {},
    "Drug_C": {"T_CD4": 0.6},
}

EXAMPLE_MT_GENES = ["MT-ND1", "MT-CO1", "MT-CO2", "MT-ATP6", "MT-CYB"]
EXAMPLE_RIBO_GENES = ["RPS3", "RPS6", "RPL3", "RPL7"]
EXAMPLE_HB_GENES = ["HBA1", "HBB"]

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


def simulate_profiles(rng: np.random.Generator, n_genes: int) -> pd.DataFrame:
    """
    Simulate reference profiles for all subtypes.

    Parameters
    ----------
    rng : np.random.Generator
        Random number generator.
    n_genes : int
        Total number of genes.

    Returns
    -------
    pd.DataFrame
        Reference profiles (genes x subtypes), every column sums to 1.
    """

    qc_genes = EXAMPLE_MT_GENES + EXAMPLE_RIBO_GENES + EXAMPLE_HB_GENES + ["MALAT1"]
    n_regular = n_genes - len(qc_genes)
    gene_names = [f"GENE{i:04d}" for i in range(1, n_regular + 1)] + qc_genes

    subtypes = list(EXAMPLE_HIERARCHY.keys())
    majors = sorted(set(EXAMPLE_HIERARCHY.values()))

    baseline = rng.gamma(0.4, 1.0, n_regular)

    # Disjoint marker genes
    marker_pool = rng.permutation(n_regular)
    major_markers = {
        major: marker_pool[i * 60 : (i + 1) * 60] for i, major in enumerate(majors)
    }
    offset = len(majors) * 60
    sub_markers = {
        sub: marker_pool[offset + i * 30 : offset + (i + 1) * 30]
        for i, sub in enumerate(subtypes)
    }

    profiles = np.zeros((n_genes, len(subtypes)))
    for j, sub in enumerate(subtypes):
        profile = baseline * rng.lognormal(0.0, 0.2, n_regular)
        profile[major_markers[EXAMPLE_HIERARCHY[sub]]] *= rng.lognormal(2.0, 0.5, 60)
        profile[sub_markers[sub]] *= rng.lognormal(1.5, 0.4, 30)
        profile = profile / profile.sum()

        # Fractions of mitochondrial, ribosomal, hemoglobin and MALAT1 counts,
        # that can be removed by the preprocessing
        qc_profile = np.concatenate(
            [
                np.full(len(EXAMPLE_MT_GENES), 0.02 / len(EXAMPLE_MT_GENES)),
                np.full(len(EXAMPLE_RIBO_GENES), 0.05 / len(EXAMPLE_RIBO_GENES)),
                np.full(len(EXAMPLE_HB_GENES), 0.0005 / len(EXAMPLE_HB_GENES)),
                [0.03],
            ]
        )
        profiles[:, j] = np.concatenate([profile * (1 - qc_profile.sum()), qc_profile])

    return pd.DataFrame(profiles, index=gene_names, columns=subtypes)


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


def simulate_single_cells(
    rng: np.random.Generator,
    profiles: pd.DataFrame,
    n_cells: int,
    dispersion: float = 0.3,
) -> ad.AnnData:
    """
    Simulate raw single cell counts from the reference profiles using a negative binomial.

    Some cells are simulated to have a bad quality.

    Parameters
    ----------
    rng : np.random.Generator
        Random number generator.
    profiles : pd.DataFrame
        Reference profiles (genes x subtypes).
    n_cells : int
        Number of cells.
    dispersion : float, default=0.3
        Dispersion of the negative binomial distribution.

    Returns
    -------
    ad.AnnData
        Raw counts with the annotation columns cell_type and major in obs.
    """

    subtypes = list(profiles.columns)
    fractions = np.array([EXAMPLE_SC_FRACTIONS[sub] for sub in subtypes])
    celltypes = rng.choice(subtypes, size=n_cells, p=fractions / fractions.sum())

    library_sizes = np.array([EXAMPLE_LIBRARY_SIZES[ct] for ct in celltypes])
    library_sizes = library_sizes * rng.lognormal(0.0, 0.3, n_cells)

    mean = profiles[celltypes].to_numpy().T * library_sizes[:, None]

    # Low quality cells
    is_mt = profiles.index.isin(EXAMPLE_MT_GENES)
    high_mt = rng.random(n_cells) < 0.03
    mean[np.ix_(high_mt, is_mt)] *= 15
    low_count = rng.random(n_cells) < 0.02
    mean[low_count] *= 0.05

    counts = rng.poisson(rng.gamma(1 / dispersion, mean * dispersion))

    obs = pd.DataFrame(
        {
            "cell_type": pd.Categorical(celltypes, categories=subtypes),
            "major": pd.Categorical(
                [EXAMPLE_HIERARCHY[ct] for ct in celltypes],
                categories=sorted(set(EXAMPLE_HIERARCHY.values())),
            ),
        },
        index=[f"cell_{i:05d}" for i in range(1, n_cells + 1)],
    )

    return ad.AnnData(
        X=sp.csr_matrix(counts.astype(np.float32)),
        obs=obs,
        var=pd.DataFrame(index=profiles.index),
    )


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


def simulate_compositions(
    rng: np.random.Generator, sample_sheet: pd.DataFrame, concentration: float = 40.0
) -> pd.DataFrame:
    """
    Simulate compositions depending on disease and medication.

    Parameters
    ----------
    rng : np.random.Generator
        Random number generator.
    sample_sheet : pd.DataFrame
        Sample sheet containing the columns SampleID, disease and medication.
    concentration : float, default=40.0
        Concentration of the Dirichlet distribution.

    Returns
    -------
    pd.DataFrame
        Cell type compositions (subtypes x samples).
    """

    subtypes = list(EXAMPLE_HEALTHY_COMPOSITION.keys())
    compositions = []

    for _, sample in sample_sheet.iterrows():
        mean = pd.Series(EXAMPLE_HEALTHY_COMPOSITION)

        if sample["disease"] == "disease":
            for ct, effect in EXAMPLE_DISEASE_EFFECT.items():
                mean[ct] *= effect

        for ct, effect in EXAMPLE_MEDICATION_EFFECT.get(
            sample["medication"], {}
        ).items():
            mean[ct] *= effect

        mean = mean / mean.sum()
        compositions.append(rng.dirichlet(concentration * mean.to_numpy()))

    return pd.DataFrame(
        np.array(compositions).T, index=subtypes, columns=sample_sheet["SampleID"]
    )


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


def simulate_bulk(
    rng: np.random.Generator,
    profiles: pd.DataFrame,
    compositions: pd.DataFrame,
    is_disease: np.ndarray,
    n_deg: int = 40,
    dispersion: float = 0.05,
) -> pd.DataFrame:
    """
    Simulate raw bulk counts from cell type compositions.

    Cell types contribute proportional to their library size. A gene wise
    domain shift between single cell and bulk data is applied and a set of
    genes is differentially expressed in disease patients independent of the
    composition.

    Parameters
    ----------
    rng : np.random.Generator
        Random number generator.
    profiles : pd.DataFrame
        Relative expression profiles (genes x subtypes).
    compositions : pd.DataFrame
        Cell type compositions (subtypes x samples).
    is_disease : np.ndarray
        Boolean mask of disease samples.
    n_deg : int, default=40
        Number of genes differentially expressed in disease samples.
    dispersion : float, default=0.05
        Dispersion of the negative binomial distribution.

    Returns
    -------
    pd.DataFrame
        Raw bulk counts (genes x samples).
    """

    library_sizes = pd.Series(EXAMPLE_LIBRARY_SIZES)[profiles.columns]
    weights = compositions.loc[profiles.columns].mul(library_sizes, axis=0)

    mean = profiles.to_numpy() @ weights.to_numpy()

    # Gene wise domain shift between single cell and bulk RNA-seq
    mean *= rng.lognormal(0.0, 0.5, profiles.shape[0])[:, None]

    # Composition independent differentially expressed genes
    regular_genes = np.flatnonzero(profiles.index.str.startswith("GENE"))
    deg_idx = rng.choice(regular_genes, size=n_deg, replace=False)
    deg_effect = rng.choice([-1.0, 1.0], size=n_deg) * rng.uniform(0.8, 1.5, n_deg)
    mean[np.ix_(deg_idx, is_disease)] *= np.exp(deg_effect)[:, None]

    depth = rng.lognormal(np.log(2e7), 0.2, mean.shape[1])
    mean = mean / mean.sum(axis=0) * depth

    counts = rng.poisson(rng.gamma(1 / dispersion, mean * dispersion))

    return pd.DataFrame(counts, index=profiles.index, columns=compositions.columns)


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


def simulate_survival(
    rng: np.random.Generator,
    sample_sheet: pd.DataFrame,
    compositions: pd.DataFrame,
    max_follow_up: float = 60.0,
) -> pd.DataFrame:
    """
    Simulate survival times in months.

    The hazard increases with disease and age and decreases with the
    abundance of T_CD8 cells.

    Parameters
    ----------
    rng : np.random.Generator
        Random number generator.
    sample_sheet : pd.DataFrame
        Sample sheet containing the columns SampleID, disease and age.
    compositions : pd.DataFrame
        Cell type compositions (subtypes x samples).
    max_follow_up : float, default=60.0
        Maximum follow up time in months.

    Returns
    -------
    pd.DataFrame
        Sample sheet with the additional columns survival_time and event.
    """

    t_cd8 = compositions.loc["T_CD8", sample_sheet["SampleID"]].to_numpy()
    t_cd8 = (t_cd8 - t_cd8.mean()) / t_cd8.std()

    log_hazard = (
        np.log(1 / 80)
        + 1.0 * (sample_sheet["disease"] == "disease").to_numpy()
        + 0.03 * (sample_sheet["age"].to_numpy() - 60)
        - 0.8 * t_cd8
    )

    event_time = rng.exponential(1 / np.exp(log_hazard))
    censor_time = np.minimum(rng.exponential(120, len(sample_sheet)), max_follow_up)

    sample_sheet = sample_sheet.copy()
    sample_sheet["survival_time"] = np.round(np.minimum(event_time, censor_time), 1)
    sample_sheet["event"] = (event_time <= censor_time).astype(int)

    return sample_sheet


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


def generate_example(
    seed: int = 42,
    n_cells: int = 3000,
    n_genes: int = 2000,
    n_patients: int = 60,
) -> tuple[ad.AnnData, pd.DataFrame, pd.DataFrame]:
    """
    Generate a synthetic example dataset to test the functionality of HIDE-Deconv.

    The single cell data contains six immune cell subtypes (column cell_type),
    which are grouped into three major cell types (column major). The bulk
    samples are split into healthy and disease patients. Disease patients
    receive one of three medications, which together with the disease state
    affect the cell type composition. Survival depends on disease, age and the
    abundance of T_CD8 cells.

    Note: The data is **synthetic**.
    Parameters
    ----------
    seed : int, default=42
        Random seed for reproducible data.
    n_cells : int, default=3000
        Number of single cells.
    n_genes : int, default=2000
        Number of genes.
    n_patients : int, default=60
        Number of bulk samples. Half of the patients are healthy.

    Returns
    -------
    tuple[ad.AnnData, pd.DataFrame, pd.DataFrame]
        Single cells with raw counts, bulk with raw counts (genes x samples) and
        sample sheet with the columns SampleID, disease, medication, age, sex,
        survival_time and event.
    """

    rng = np.random.default_rng(seed)

    profiles = simulate_profiles(rng, n_genes)
    adata = simulate_single_cells(rng, profiles, n_cells)

    # Shuffle healthy and disease patients
    n_healthy = n_patients // 2
    disease = rng.permutation(
        ["healthy"] * n_healthy + ["disease"] * (n_patients - n_healthy)
    )
    medication = np.where(
        disease == "disease",
        rng.choice(list(EXAMPLE_MEDICATION_EFFECT.keys()), size=n_patients),
        "none",
    )

    sample_sheet = pd.DataFrame(
        {
            "SampleID": [f"P{i:03d}" for i in range(1, n_patients + 1)],
            "disease": disease,
            "medication": medication,
            "age": np.clip(rng.normal(60, 10, n_patients), 30, 90).astype(int),
            "sex": rng.choice(["female", "male"], size=n_patients),
        }
    )

    compositions = simulate_compositions(rng, sample_sheet)
    bulk = simulate_bulk(
        rng, profiles, compositions, (sample_sheet["disease"] == "disease").to_numpy()
    )
    bulk.columns.name = None
    sample_sheet = simulate_survival(rng, sample_sheet, compositions)

    return adata, bulk, sample_sheet
