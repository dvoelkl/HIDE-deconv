# Example data

HIDE-deconv comes with a function to generate a small synthetic dataset. It can be used to try out both the command line workflow and the Python API.

All examples in the documentation were executed using the synthetic data.

## Generate the files

```bash
hide-deconv example -o <OutputFolder>
```

The command stores the following files in the output folder:

| File | Content |
|---|---|
| `single_cells.h5ad` | 3000 single cells x 2000 genes, raw counts |
| `bulk.csv` | 60 bulk samples, raw counts (genes x samples) |
| `sample_sheet.csv` | Meta information of the bulk samples |

The same data is available in Python:

```python
from hide_deconv import generate_example

adata, bulk, sample_sheet = generate_example()
```

## Single cells

The AnnData file contains two cell type annotation layers in `adata.obs`,
which can be used for hierarchical deconvolution:

| `cell_type` (finest layer) | `major` |
|---|---|
| `T_CD4`, `T_CD8` | `T` |
| `B_naive`, `B_memory` | `B` |
| `Mono_classical`, `Mono_nonclassical` | `Mono` |

The cell types differ in their library size. Additionally, the data contains
mitochondrial, ribosomal and hemoglobin genes as well as a small number of
low quality cells, so that the single cell preprocessing
(`hide-deconv anndata preprocess`) can be tested.

## Bulk and sample sheet

The bulk samples are linked to the sample sheet by the column `SampleID`.
Compared to the single cells, the bulk data contains a gene wise domain shift.

| Column | Content |
|---|---|
| `SampleID` | Bulk sample ID |
| `disease` | `healthy` (30 patients) or `disease` (30 patients) |
| `medication` | `none` for healthy patients, `Drug_A`, `Drug_B` or `Drug_C` for disease patients |
| `age`, `sex` | Covariates |
| `survival_time` | Time to event or last follow-up in months |
| `event` | 0: censored, 1: event |

The simulated effects can be recovered with the analysis functions:

- Disease patients have fewer `T_CD8` and more `Mono_classical` cells.
- `Drug_A` increases `B_memory` cells, `Drug_C` decreases `T_CD4` cells.
- A high abundance of `T_CD8` cells improves survival.
- A set of genes is differentially expressed in disease patients independent of the cell type composition.
