"""
=====================================================
PLS-DA analysis
=====================================================
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler

from ..visualization import plot_plsda_loading, plot_plsda_score, plot_plsda_vip


def prepare_plsda_inputs(
    C_est: pd.DataFrame,
    sample_sheet: pd.DataFrame,
    sample_id_col: str,
    cohort_col: str,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Prepares a given sample sheet in order to be used for PLS-DA
    """
    C_est = C_est.copy()
    C_est.columns = C_est.columns.astype(str)

    samples = sample_sheet[[sample_id_col, cohort_col]].copy()
    samples = samples.dropna(subset=[sample_id_col, cohort_col])
    samples[sample_id_col] = samples[sample_id_col].astype(str)
    samples = samples.drop_duplicates(subset=[sample_id_col], keep="first")
    samples = samples[samples[sample_id_col].isin(C_est.columns)]

    if len(samples) == 0:
        raise ValueError("No matching sample ids were found for PLS-DA.")

    samples = samples.set_index(sample_id_col).reindex(C_est.columns)
    samples = samples.dropna(subset=[cohort_col])

    if samples[cohort_col].nunique() < 2:
        raise ValueError("PLS-DA requires at least two cohort values.")

    C_est = C_est.loc[:, samples.index]
    labels = samples[cohort_col].astype(str)

    return C_est, labels


def calculate_vip(model: PLSRegression) -> np.ndarray:
    """
    Calculates the variable importance in projections
    """
    weights = model.x_weights_
    scores = model.x_scores_
    y_loadings = model.y_loadings_

    n_features, n_components = weights.shape
    weight_norm = np.sum(weights**2, axis=0)
    weight_norm[weight_norm == 0] = 1.0

    explained_y = np.array(
        [
            np.sum(scores[:, idx] ** 2) * np.sum(y_loadings[:, idx] ** 2)
            for idx in range(n_components)
        ],
        dtype=float,
    )

    total_explained_y = explained_y.sum()
    if total_explained_y == 0:
        return np.ones(n_features, dtype=float)

    vip = np.sqrt(
        n_features
        * ((weights**2 / weight_norm) * explained_y).sum(axis=1)
        / total_explained_y
    )

    return vip


def run_plsda(
    data: pd.DataFrame,
    sample_sheet: pd.DataFrame,
    sample_id_col: str,
    cohort_col: str,
    out_path: Path,
) -> pd.DataFrame:
    """
    Runs a PLS-DA (PLS2) model and saves the corresponding plots


    Parameters
    ----------
    C_est : pd.DataFrame
        Estimated composition to be used for PLS-DA
    sample_sheet : pd.DataFrame
        Samples sheet holding clinical metainformation
    sample_id_col : str
        Column name linking the sample sheet with the estimated compositions
    cohort_col : str
        Column name holding the different cohorts
    out_path : Path
        Path, where the created plots will be stored

    Returns
    -------
    pd.DataFrame
        Estimated Scores
    """

    data, labels = prepare_plsda_inputs(data, sample_sheet, sample_id_col, cohort_col)

    classes = sorted(labels.unique())
    y = pd.get_dummies(labels, dtype=float).reindex(columns=classes, fill_value=0.0)

    x = data.T.to_numpy(dtype=float)
    x = StandardScaler().fit_transform(x)

    n_components = min(2, x.shape[0], x.shape[1], y.shape[1])
    if n_components < 2:
        raise ValueError("PLS-DA requires at least two samples and two features.")

    model = PLSRegression(n_components=2, scale=False)
    model.fit(x, y.to_numpy(dtype=float))

    scores = pd.DataFrame(
        model.x_scores_[:, :2],
        index=data.columns,
        columns=["PLS1", "PLS2"],
    )
    scores[cohort_col] = labels.reindex(scores.index).values

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    scores.to_csv(out_path.with_suffix(".csv"))

    vip = pd.Series(calculate_vip(model), index=data.index, name="VIP")
    loading = pd.Series(model.x_loadings_[:, 0], index=data.index, name="Loading")

    plot_plsda_score(scores, out_path.with_suffix(".png"), cohort_col)
    plot_plsda_vip(vip, out_path.with_name(f"{out_path.stem}_vip.png"))
    plot_plsda_loading(loading, out_path.with_name(f"{out_path.stem}_loading.png"))

    return scores
