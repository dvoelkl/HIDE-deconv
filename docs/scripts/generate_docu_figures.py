"""
=====================================================
Generate the figures used in the documentation
=====================================================

Runs HIDE-Deconv on the synthetic example data and stores all plots shown in
the documentation in docs/assets/plots
"""

from pathlib import Path
import tempfile

import numpy as np
import pandas as pd
import scanpy as sc
from PIL import Image

from hide_deconv import generate_example
from hide_deconv.pipelines.deconvolve_hide_pipeline import (
    normalize_bulk_to_cpm,
    predict_deconvolution_results,
)
from hide_deconv.pipelines.lazy_deconvolution_pipeline import setup_model
from hide_deconv.preprocessing import get_common_genes
from hide_deconv.statistic import (
    pydeseq2_preprocess,
    run_cox_regression,
    run_mann_whitney_u,
    run_plsda,
    run_pydeseq2,
)
from hide_deconv.visualization import (
    plot_anndata_umap,
    plot_celltype_bar_scatter,
    plot_cox_forest,
    plot_genemap,
    plot_hier_heat,
    plot_kaplan_meier_cohort,
    plot_kaplan_meier_comp,
    plot_kmeans_pca,
    plot_pca,
    plot_umap,
)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

OUT_DIR = Path(__file__).resolve().parents[1] / "assets" / "plots"
MAX_WIDTH = 1400

# Generated file -> file name in the documentation
GALLERY = {
    "pca.png": "pca.png",
    "umap.png": "umap.png",
    "kmeans_pca.png": "kmeans_pca.png",
    "celltype_scatter.png": "celltype_scatter.png",
    "hier_heat.png": "hier_heat.png",
    "genemap.png": "genemap.png",
    "plsda.png": "plsda_score.png",
    "plsda_vip.png": "plsda_vip.png",
    "plsda_loading.png": "plsda_loading.png",
    "plsda_biplot.png": "plsda_biplot.png",
    "deg_volcano.png": "volcano.png",
    "km_composition.png": "km_composition.png",
    "km_cohort.png": "km_cohort.png",
    "cox_forest.png": "cox_forest.png",
    "anndata_umap.png": "anndata_umap.png",
}

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


def deconvolve_example(
    adata, bulk, celltype_cols: list[str]
) -> tuple[list[pd.DataFrame], object]:
    """
    Deconvolve the example data analogous to the lazy API, but return the model as well.
    """

    bulk = normalize_bulk_to_cpm(bulk)
    adata.obs["library_size"] = np.asarray(adata.X.sum(axis=1)).ravel()
    library_sizes = adata.obs.groupby(celltype_cols[0], observed=True)[
        "library_size"
    ].median()

    adata = sc.pp.normalize_total(adata, target_sum=1e4, copy=True)
    adata = adata[:, get_common_genes(adata, bulk)].copy()

    # Reduced settings for the example data, same as in the documentation examples
    model, adata = setup_model(
        adata,
        celltype_cols,
        n_genes=1000,
        n_train_bulks=5000,
        n_cells_per_bulk=100,
        n_iter=1000,
        seed=42,
    )

    predictions, _ = predict_deconvolution_results(
        model,
        adata,
        bulk.loc[model.gene_labels],
        celltype_cols[0],
        100,
        library_size_correction=library_sizes,
    )

    return predictions, model


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


def create_plots(out: Path) -> None:
    """
    Create all gallery plots in the given folder.
    """

    adata, bulk, sample_sheet = generate_example()

    (C_sub, C_major), model = deconvolve_example(
        adata.copy(), bulk, ["cell_type", "major"]
    )

    labels = sample_sheet.set_index("SampleID").loc[C_sub.columns, "disease"].tolist()

    # Composition plots
    plot_pca(C_sub, str(out / "pca.png"), labeling=labels, group_name="disease")
    plot_umap(C_sub, str(out / "umap.png"), labeling=labels, group_name="disease")
    plot_kmeans_pca(
        C_sub,
        str(out / "kmeans_pca.png"),
        n_clusters=2,
        labeling=labels,
        group_name="disease",
    )
    plot_celltype_bar_scatter(
        C_sub, str(out / "celltype_scatter.png"), labeling=labels, group_name="disease"
    )

    # Hierarchical differences
    mwu_sub = run_mann_whitney_u(C_sub, sample_sheet, "SampleID", "disease")
    mwu_major = run_mann_whitney_u(C_major, sample_sheet, "SampleID", "disease")
    A_ls = [
        pd.DataFrame(
            A.numpy(),
            index=model.celltype_layer_labels[layer],
            columns=model.celltype_layer_labels[0],
        )
        for layer, A in enumerate(model.A_l)
    ]
    mean_columns = [col for col in mwu_sub.columns if col.startswith("mean[")]
    plot_hier_heat(
        mwu_sub,
        [mwu_major],
        ["cell_type", "major"],
        A_ls,
        mean_columns[0][5:-1],
        mean_columns[1][5:-1],
        out / "hier_heat.png",
    )

    # Genes relevant for deconvolution
    X = pd.DataFrame(
        model.X_l[0].numpy(),
        index=model.gene_labels,
        columns=model.celltype_layer_labels[0],
    )
    gene_weights = pd.Series(model.g_l[0].detach().numpy(), index=model.gene_labels)
    relevant_genes = gene_weights.mul(X.var(axis=1)).sort_values(ascending=False)
    plot_genemap(
        X,
        relevant_genes.index[0:30],
        "Most relevant genes of cell_type layer",
        out / "genemap.png",
    )

    # PLS-DA
    run_plsda(C_sub, sample_sheet, "SampleID", "disease", out / "plsda")

    # Differential gene expression
    counts, metadata = pydeseq2_preprocess(
        bulk, sample_sheet, "SampleID", "disease", None
    )
    run_pydeseq2(counts, metadata, "disease", "disease", "healthy", None, out / "deg")

    # Survival
    cox_results = run_cox_regression(
        C_sub, sample_sheet, "SampleID", "survival_time", "event", ["age"]
    )
    plot_cox_forest(cox_results, str(out / "cox_forest.png"))
    plot_kaplan_meier_comp(
        C_sub,
        sample_sheet,
        "SampleID",
        "survival_time",
        "event",
        "T_CD8",
        str(out / "km_composition.png"),
        show_censors=True,
        show_risk_table=True,
    )
    plot_kaplan_meier_cohort(
        sample_sheet,
        "disease",
        "survival_time",
        "event",
        str(out / "km_cohort.png"),
        show_censors=True,
        show_risk_table=True,
    )

    # Single cells
    plot_anndata_umap(adata, str(out / "anndata_umap.png"), obs_col="cell_type")


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        create_plots(tmp)

        # Downscale plots
        for src_name, dst_name in GALLERY.items():
            img = Image.open(tmp / src_name)
            if img.width > MAX_WIDTH:
                img = img.resize(
                    (MAX_WIDTH, round(img.height * MAX_WIDTH / img.width)),
                    Image.LANCZOS,
                )
            img.save(OUT_DIR / dst_name, optimize=True)
            print(f"Saved {OUT_DIR / dst_name}")


if __name__ == "__main__":
    main()
