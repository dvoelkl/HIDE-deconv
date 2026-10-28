# Extended API example

The extended API exposes the individual preprocessing, training and prediction
steps. It is useful when domain transfer or sample filtering needs to be
customized.

The example below uses healthy bulk samples as the reference domain and
deconvolves the remaining samples. The example runs on the [example data](example-data.md), which can be created with `hide-deconv example`.

```python
import anndata as ad
import pandas as pd
import numpy as np
import scanpy as sc
from hide_deconv.preprocessing import create_reference, create_bulks, reduce_genes
from hide_deconv.models import HIDE
from hide_deconv.statistic import run_kruskal_wallis
from hide_deconv.preprocessing.bulk_preprocessing import get_domain_transfer_factor
from hide_deconv.pipelines.deconvolve_hide_pipeline import normalize_bulk_to_cpm

# 1. Load single-cell, bulk and sample sheet data
adata = ad.read_h5ad("single_cells.h5ad")
bulk = pd.read_csv("bulk.csv", index_col=0)  # genes x samples
sample_sheet = pd.read_csv("sample_sheet.csv")  # sample IDs in column "SampleID"

# Calculate library sizes on the raw counts of the complete single-cell gene set
library_sizes = (
	adata.obs.assign(library_size=np.asarray(adata.X.sum(axis=1)).ravel())
	.groupby("cell_type")["library_size"]
	.median()
)

# Normalize single cells on the complete gene set before simulating bulks
sc.pp.normalize_total(adata, target_sum=1e4)

# Use healthy samples as reference domain and deconvolve the remaining samples
healthy_ids = sample_sheet.loc[sample_sheet["disease"] == "healthy", "SampleID"]
bulk = normalize_bulk_to_cpm(bulk)
healthy_bulk = bulk.loc[:, healthy_ids]
bulk = bulk.drop(columns=healthy_ids)
common_genes = adata.var_names.intersection(bulk.index).intersection(healthy_bulk.index)
adata = adata[:, common_genes].copy()

# Reduce to the most informative genes (for real data we recommend 5000 genes)
adata = reduce_genes(adata, 1000, ct_col="cell_type")
healthy_bulk = healthy_bulk.loc[adata.var_names]
bulk = bulk.loc[adata.var_names]

# 2. Create reference profiles and hierarchy (single layer example)
X_sub = create_reference(adata, celltype_col="cell_type")
A_l = [pd.DataFrame(np.eye(X_sub.shape[1]), index=X_sub.columns, columns=X_sub.columns)]
X_l = [X_sub]

# 3. Simulate training bulks (for real data we recommend at least 10000 bulks)
Y_train, C_train = create_bulks(adata, n_bulks=5000, n_cells_per_bulk=100, celltype_col="cell_type")

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
alpha = get_domain_transfer_factor(healthy_bulk, Y_domain)
alpha_inv = (1 / alpha).replace([np.inf, -np.inf], 0).fillna(1.0)
bulk = bulk.mul(alpha_inv, axis=0)

# 6. Deconvolution on the batch-corrected bulk data
results = hide.predict(
	bulk.loc[hide.gene_labels],
	norm=True,
	library_sizes=library_sizes,
)["prediction"]

# 7. Optional: Difference in composition between medications
diff = run_kruskal_wallis(results[0], sample_sheet, sample_id_col="SampleID", cohort_col="medication")
```
