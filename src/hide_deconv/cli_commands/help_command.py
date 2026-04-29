"""
=====================================================
ViewModel functions for CLI help command
=====================================================
"""

from rich.console import Console
from rich.panel import Panel

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

console = Console()

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


def show_help():
    """
    Display a quick start guide for the HIDE-Deconv package.
    """
    console.print(
        Panel(
            "[bold]HIDE-Deconv Package[/bold]\n"
            "[dim]Bulk RNA-seq deconvolution with[/dim] [cyan]HIDE[/cyan]\n\n"
            "[bold cyan]Algorithms[/bold cyan]\n"
            "• [cyan]HIDE[/cyan]: Hierarchical deconvolution with gene-weight learning.\n"
            ""
            "  and cell type-specific gene regulation.\n\n"
            "[bold cyan]Required Input[/bold cyan]\n"
            "• [bold]Single-cell training data[/bold]: AnnData (.h5ad)\n"
            "  - Cell type annotations in [i]adata.obs[/i]\n"
            "  - Gene identifiers in [i]adata.var_names[/i] (alias: [i]adata.var.index[/i])\n"
            "  - Raw counts in [i]adata.X[/i] are recommended\n"
            "• [bold]Bulk deconvolution data[/bold]: CSV\n"
            "  - Gene identifiers as row index (same type as in Single Cells)\n"
            "  - Bulk sample n ames as columns\n"
            "  - Use raw counts or compatible normalization across datasets\n\n"
            "[bold cyan]Full workflow[/bold cyan]\n"
            "HIDE-Deconv offers a guided workflow that runs all required steps in order.\n"
            "Run in an empty target directory:\n"
            "[cyan]$ hide-deconv run[/cyan]\n\n"
            "This will initialize the project, preprocess data, account for domain transfer,\n"
            "learn gene weights and run deconvolution with your selected model.\n"
            "Results are written to the results directory.",
            title="HIDE-Deconv Package Help",
            padding=(1, 2),
        )
    )
    commands = Panel(
        "[bold]hide-deconv init[/bold]              Initialize project structure\n"
        "[bold]hide-deconv preprocess[/bold]        Run preprocessing\n"
        "[bold]hide-deconv train[/bold]             Learn optimal gene weights\n"
        "[bold]hide-deconv deconv[/bold]            Open deconvolution menu\n"
        "[bold]hide-deconv deconv hide[/bold]       Run HIDE directly\n"
        ""
        "[bold]hide-deconv config edit[/bold]       Edit configuration\n"
        "[bold]hide-deoncv config show[/bold]       Show configuration\n"
        "[bold]hide-deconv analyze benchmark[/bold] Benchmarkes results vs known ground truth\n"
        "[bold]hide-deconv analyze diff[/bold]      Analyze differences between cohorts\n"
        "[bold]hide-deconv analyze pca[/bold]       Plots compositions as PCA plot\n"
        "[bold]hide-deconv analyze umap[/bold]      Plots compositions as UMAP plot\n"
        "[bold]hide-deconv analyze survival[/bold]  Runs Cox regression\n"
        "[bold]hide-deconv download[/bold]          Download example AnnData Single Cells\n"
        "[bold]hide-deconv simulate[/bold]          Split AnnData in train and create test data\n"
        "[bold]hide-deconv bulk pca[/bold]          Plots bulk RNA-seq data as PCA plot\n"
        "[bold]hide-deconv bulk umap[/bold]         Plots bulk RNA-seq data as UMAP plot\n"
        "[bold]hide-deconv bulk merge[/bold]        Merges multiple RNA-seq data sources\n"
        "[bold]hide-deconv anndata inspect[/bold]   Summarizes an AnnData file\n"
        "[bold]hide-deconv anndata subset[/bold]    Subsets AnnData file to certain subsets in observations\n"
        "[bold]hide-deconv anndata preprocess[/bold] Preprocesses an AnnData file\n\n"
        "Run a command with the [i]--help[/i] flag for further information.",
        title="Commands",
        padding=(1, 2),
    )
    console.print(commands)
