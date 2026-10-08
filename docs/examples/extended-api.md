# Extended API example

The extended API exposes the individual preprocessing, training and prediction
steps. It is useful when domain transfer or sample filtering needs to be
customized.

The example below uses healthy bulk samples as the reference domain and
deconvolves the remaining samples.

```python
import anndata as ad
import pandas as pd
import numpy as np
from hide_deconv.preprocessing import create_reference, create_bulks
from hide_deconv.models import HIDE
from hide_deconv.statistic import run_mann_whitney_u
from hide_deconv.preprocessing.bulk_preprocessing import get_domain_transfer_factor
from hide_deconv.pipelines.deconvolve_hide_pipeline import normalize_bulk_to_cpm

# 1. Load single-cell, bulk and sample sheet data
adata = ad.read_h5ad("single_cells.h5ad")
bulk = pd.read_csv("bulk.csv", index_col=0)  # genes x samples
sample_sheet = pd.read_csv("sample_sheet.csv", index_col=0)

# The sample sheet index must contain the corresponding bulk sample IDs.
sample_sheet = sample_sheet.reindex(bulk.columns)
bulk = normalize_bulk_to_cpm(bulk)
healthy_bulk = bulk.loc[:, sample_sheet["disease"] == "healthy"]
bulk = bulk.loc[:, sample_sheet["disease"] != "healthy"]
common_genes = adata.var_names.intersection(bulk.index).intersection(healthy_bulk.index)
adata = adata[:, common_genes].copy()
healthy_bulk = healthy_bulk.loc[common_genes]
bulk = bulk.loc[common_genes]

# 2. Create reference profiles and hierarchy (single layer example)
X_sub = create_reference(adata, celltype_col="cell_type")
A_l = [pd.DataFrame(np.eye(X_sub.shape[1]), index=X_sub.columns, columns=X_sub.columns)]
X_l = [X_sub]

# 3. Simulate training bulks
Y_train, C_train = create_bulks(adata, n_bulks=1000, n_cells_per_bulk=100, celltype_col="cell_type")

# 4. Initialize and train model
hide = HIDE(X_l, A_l)
hide.train(Y_train, C_train, iter=1000)

# 5. Calculate domain-transfer factors from the healthy bulk reference
Y_domain, _ = create_bulks(
	adata,
	n_bulks=1000,
	n_cells_per_bulk=100,
	celltype_col="cell_type",
)
Y_domain = Y_domain.loc[common_genes]
alpha = get_domain_transfer_factor(healthy_bulk, Y_domain)
alpha_inv = (1 / alpha).replace([np.inf, -np.inf], 0).fillna(1.0)
bulk = bulk.mul(alpha_inv, axis=0)

# 6. Calculate library sizes on the complete single-cell gene set
library_sizes = (
	adata.obs.assign(library_size=np.asarray(adata.X.sum(axis=1)).ravel())
	.groupby("cell_type")["library_size"]
	.median()
)

# 7. Deconvolution on the batch-corrected bulk data
results = hide.predict(
	bulk.loc[hide.gene_labels],
	norm=True,
	library_sizes=library_sizes,
)["prediction"]

# 8. Optional: Difference in composition analysis
# diff = run_mann_whitney_u(results[0], sample_sheet, sample_id_col="SampleID", cohort_col="Cohort")
```
