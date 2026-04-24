# HIDE-deconv

**Interactive command line tool and python package for hierarchical deconvolution and analysis of bulk RNA-seq data.**

---

## Features

- Designed for AnnData single cell datasets
- Open Source package, that can be run on safe servers
- Hierarchical cell type deconvolution for any number of cell type annotation layers
- Includes methods for post-deconvolution analysis
- Usable via command line interface and Python API
- Provides a guided workflow that allows users without programming experience to perform deconvolution

![Workflow Summary](/figures/HIDE-deconv%20Graphical%20summary.png)

## Installation

```bash
# Create and activate a new virtual environment (recommended)
python3 -m venv .venv
source .venv/bin/activate

# Install HIDE-deconv
pip install hide-deconv
```

---

## Necessary Data

- **Single-cell data:** Annotated AnnData (.h5ad) file with gene names in `adata.var_names` and cell type annotations for each desired layer in `adata.obs` (at least one layer of cell type annotations is necessary).
- **Bulk RNA-seq data:** CSV file, genes as row index, samples as columns. Gene IDs must match single-cell data.
- **Sample sheet (optional):** CSV with sample meta-information (e.g., cohort, survival time, event).
- **Data Normalization:** We recommend to use raw counts for all datasets.

---

## Command Line Workflow

**Deconvolution (standard workflow):**
```bash
hide-deconv run --path <project_dir>
```
This initializes the project, preprocesses data, trains the model, and runs deconvolution.

**Difference in composition & survival analysis:**
```bash
hide-deconv analyze diff --path <project_dir>
hide-deconv analyze survival --path <project_dir>
```
- For difference analysis, the sample sheet must contain columns for sample ID and cohort.
- For survival analysis, the sample sheet must contain columns for sample ID, survival time, and event.

**Command overview:**
```bash
hide-deconv help
```

This displays a short introduction to the command line interface and gives an overview of all available commands.

---

## API Example

```python
import anndata as ad
import pandas as pd
import numpy as np
from hide_deconv.preprocessing import (
	train_test_split_adata,
	create_reference,
	create_hierarchy,
	create_bulks,
)
from hide_deconv.models import HIDE
from hide_deconv.statistic import run_mann_whitney_u

# 1. Load AnnData
adata = ad.read_h5ad("single_cells.h5ad")

# 2. Split into training and test set
adata_train, adata_test = train_test_split_adata(adata, celltype_col="cell_type", train_frac=0.7)

# 3. Create reference profiles and hierarchy (single layer example)
X_sub = create_reference(adata_train, celltype_col="cell_type")
A_l = [pd.DataFrame(np.eye(X_sub.shape[1]), index=X_sub.columns, columns=X_sub.columns)]
X_l = [X_sub]

# 4. Simulate training bulks
Y_train, C_train = create_bulks(adata_train, n_bulks=1000, n_cells_per_bulk=100, celltype_col="cell_type")

# 5. Simulate test bulks
Y_test, C_test = create_bulks(adata_test, n_bulks=100, n_cells_per_bulk=100, celltype_col="cell_type")

# 6. Initialize and train model
hide = HIDE(X_l, A_l)
hide.train(Y_train, C_train, iter=1000)

# 7. Deconvolution on test data
results = hide.predict(Y_test, norm=True)["prediction"]

# 8. Optional: Difference in composition analysis
# (requires a sample sheet with columns 'SampleID' and 'Cohort')

# sample_sheet = read_csv("sample_sheet.csv")
# diff = run_mann_whitney_u(results[0], sample_sheet, sample_id_col="SampleID", cohort_col="Cohort")
```

---

## Citation
HIDE-deconv's deconvolution algorithm is based on HIDE: Hierarchical Cell Type Deconvolution. If you use HIDE-deconv, please cite the following article.

Dennis Völkl, Malte Mensching-Buhr, Thomas Sterr, Sarah Bolz, Andreas Schäfer, Nicole Seifert, Jana Tauschke, Austin Rayford, Oddbjørn Straume, Helena U Zacharias, Sushma Nagaraja Grellscheid, Tim Beissbarth, Michael Altenbuchinger, Franziska Görtler, HIDE: hierarchical cell-type deconvolution, Bioinformatics, Volume 41, Issue Supplement_1, July 2025, Pages i207–i216, https://doi.org/10.1093/bioinformatics/btaf179

---

## License

This project is licensed under the MIT License.

---

## Contact

For questions, support or scientific collaboration:
- Dennis Voelkl: dennis.k.voelkl(at)uib.no
- Franziska Goertler: Franziska.Gortler(at)uib.no

---
