"""
=====================================================
ViewModel functions for CLI anndata commands
=====================================================
"""

from rich.console import Console
import anndata as ad
from pathlib import Path

from ..preprocessing import get_adata_info
from ..pipelines import preprocess_anndata_file

from InquirerPy import inquirer, prompt
from InquirerPy.validator import PathValidator

from ..constants import MSG_FAILURE, MSG_SUCCESS

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

console = Console()

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


def preprocess_anndata() -> int:
    """
    Selects and preprocesses an AnnData file.
    """
    console.print("[bold blue]AnnData Single Cell Preprocessing[/bold blue]")

    ad_path = Path(
        inquirer.filepath(
            message="Enter path to anndata single cell file:",
            default=str("."),
            mandatory=True,
            mandatory_message="An AnnData file (.h5ad) must be selected to continue setup.",
            validate=PathValidator(is_file=True, message="Input is not a file."),
        ).execute()
    )

    try:
        with console.status(
            "[bold blue]Loading AnnData File...[/bold blue]", spinner="dots"
        ):
            dict = get_adata_info(ad_path)
            adata = ad.read_h5ad(ad_path)
    except Exception:
        console.print_exception()
        return MSG_FAILURE

    obs_levels = sorted(dict["obs"])

    message_subtype = {
        "type": "list",
        "message": "Select the column of the cell type annotation:",
        "choices": obs_levels,
        "mandatory": True,
    }

    # Save subtype column to config and remove from list of available annotations
    celltype_col = str(prompt(message_subtype)[0])

    ### Parameter questions
    if not inquirer.confirm(
        message="Use standard preprocessing parameters?", default=True
    ).execute():
        min_cell = int(
            inquirer.number(
                "Enter minimum number of cells for a cell type to be included.",
                min_allowed=0,
                default=100,
                mandatory=True,
                float_allowed=False,
                mandatory_message="A positive integer must be entered.",
            ).execute()
        )

        min_gene_count = int(
            inquirer.number(
                "Enter minimum summed gene count below which a cell is considered as low quality.",
                min_allowed=0,
                default=100,
                mandatory=True,
                float_allowed=False,
                mandatory_message="A positive integer must be entered.",
            ).execute()
        )

        max_gene_count = int(
            inquirer.number(
                "Enter maximum summed gene count above which a cell is considered to be low quality.",
                min_allowed=0,
                default=100,
                mandatory=True,
                float_allowed=False,
                mandatory_message="A positive integer must be entered.",
            ).execute()
        )

        mt_percentage_per_cell = float(
            inquirer.number(
                "Enter percentage of mitochondrial gene expression, above which a cell is considered low quality.",
                min_allowed=0.0,
                max_allowed=1.0,
                default=0.05,
                mandatory=True,
                float_allowed=True,
                mandatory_message="A percentage between 0.0 and 1.0 must be entered.",
            ).execute()
        )

        hb_percentage_per_cell = float(
            inquirer.number(
                "Enter percentage of haemoglobine gene expression, above which a cell is considered low quality.",
                min_allowed=0.0,
                max_allowed=1.0,
                default=0.05,
                mandatory=True,
                float_allowed=True,
                mandatory_message="A percentage between 0.0 and 1.0 must be entered.",
            ).execute()
        )

        malat1_percentile_per_cell = float(
            inquirer.number(
                "Enter percentile of MALAT1 expression, above which a cell is considered low quality.",
                min_allowed=0.0,
                max_allowed=1.0,
                default=0.99,
                mandatory=True,
                float_allowed=True,
                mandatory_message="A percentile between 0.0 and 1.0 must be entered.",
            ).execute()
        )

        exclude_mito_ribo_rna = inquirer.confirm(
            message="Remove mitochondrial and ribosomal genes?", default=True
        ).execute()

        exclude_hemoglobine = inquirer.confirm(
            message="Remove haemoglobin-related genes?", default=True
        ).execute()

        filter_low_expressed_genes = inquirer.confirm(
            message="Filter low-expressed genes?", default=True
        ).execute()

        low_expressed_gene_cell_min = int(
            inquirer.number(
                "Enter minimum number of cells a gene must be expressed in to be kept.",
                min_allowed=0,
                default=10,
                mandatory=True,
                float_allowed=False,
                mandatory_message="A positive integer must be entered.",
            ).execute()
        )

        try:
            with console.status(
                "[bold blue]Preprocessing single cells...[/bold blue]",
                spinner="dots",
            ):
                adata = preprocess_anndata_file(
                    adata,
                    celltype_col,
                    min_cell,
                    min_gene_count,
                    max_gene_count,
                    mt_percentage_per_cell,
                    hb_percentage_per_cell,
                    malat1_percentile_per_cell,
                    exclude_mito_ribo_rna,
                    exclude_hemoglobine,
                    filter_low_expressed_genes,
                    low_expressed_gene_cell_min,
                )
        except Exception:
            console.print_exception()
            return MSG_FAILURE
    else:
        try:
            with console.status(
                "[bold blue]Preprocessing single cells...[/bold blue]",
                spinner="dots",
            ):
                adata = preprocess_anndata_file(adata, celltype_col)

        except Exception:
            console.print_exception()
            return MSG_FAILURE

    ad_name = ad_path.stem
    processed_path = ad_path.with_name(f"{ad_name}_preprocessed.h5ad")

    with console.status(
        "[bold blue]Saving processed file...[/bold blue]",
        spinner="dots",
    ):
        adata.write_h5ad(processed_path)

    console.print(f"[bold green]Saved processed file to {processed_path}[/bold green]")

    return MSG_SUCCESS
