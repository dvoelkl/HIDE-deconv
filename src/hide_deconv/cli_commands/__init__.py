from .config_command import setup_config, show_config
from .preprocess_command import preprocess
from .setup_command import init, load_anndata, load_bulk, setup_project
from .train_command import train_model
from .help_command import show_help
from .deconvolve_command import deconvolve_hide, deconvolve_command
from .simulate_command import create_simulation
from .cohort_command import combine_cohorts, plot_km_cohort
from .analyze_command import (
    analyze_differences,
    benchmark_result,
    create_hdiff_plot,
    create_kmean_plot,
    create_pca_plot,
    create_plsda_plot,
    survival_analysis,
    create_umap_plot,
    cell_type_clustering,
    gene_markerplot,
)
from .download_command import download_single_cells
from .anndata_command import (
    preprocess_anndata,
    inspect_anndata,
    subset_anndata,
    add_annotation,
    create_anndata_umap_plot,
)
from .bulk_command import (
    create_bulk_pca_plot,
    create_bulk_umap_plot,
    merge_bulks,
    subset_bulk,
    create_bulk_clustering,
    create_bulk_deg,
)

__all__ = [
    "setup_config",
    "show_config",
    "preprocess",
    "setup_project",
    "init",
    "load_anndata",
    "load_bulk",
    "train_model",
    "show_help",
    "deconvolve_hide",
    "deconvolve_command",
    "create_simulation",
    "combine_cohorts",
    "plot_km_cohort",
    "analyze_differences",
    "benchmark_result",
    "create_hdiff_plot",
    "create_kmean_plot",
    "create_pca_plot",
    "create_plsda_plot",
    "download_single_cells",
    "preprocess_anndata",
    "survival_analysis",
    "create_umap_plot",
    "create_bulk_pca_plot",
    "create_bulk_umap_plot",
    "merge_bulks",
    "subset_bulk",
    "inspect_anndata",
    "subset_anndata",
    "add_annotation",
    "create_anndata_umap_plot",
    "cell_type_clustering",
    "create_bulk_clustering",
    "create_bulk_deg",
    "gene_markerplot",
]
