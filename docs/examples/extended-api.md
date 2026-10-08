# Extended API example

The extended API exposes the individual preprocessing, training and prediction
steps. It is useful when domain transfer or sample filtering needs to be
customized.

The example below uses healthy bulk samples as the reference domain and
deconvolves the remaining samples.

```python
import anndata as ad
import numpy as np
import pandas as pd

from hide_deconv.models import HIDE
from hide_deconv.preprocessing import create_bulks, create_reference
from hide_deconv.preprocessing.bulk_preprocessing import get_domain_transfer_factor
from hide_deconv.pipelines.deconvolve_hide_pipeline import normalize_bulk_to_cpm

adata = ad.read_h5ad("single_cells.h5ad")
bulk = pd.read_csv("bulk.csv", index_col=0)
sample_sheet = pd.read_csv("sample_sheet.csv", index_col=0).reindex(bulk.columns)

bulk = normalize_bulk_to_cpm(bulk)
healthy_bulk = bulk.loc[:, sample_sheet["disease"] == "healthy"]
bulk = bulk.loc[:, sample_sheet["disease"] != "healthy"]

common_genes = adata.var_names.intersection(bulk.index).intersection(healthy_bulk.index)
adata = adata[:, common_genes].copy()
healthy_bulk = healthy_bulk.loc[common_genes]
bulk = bulk.loc[common_genes]

reference = create_reference(adata, celltype_col="cell_type")
projection = pd.DataFrame(
    np.eye(reference.shape[1]),
    index=reference.columns,
    columns=reference.columns,
)

training_bulk, training_composition = create_bulks(
    adata,
    n_bulks=1000,
    n_cells_per_bulk=100,
    celltype_col="cell_type",
)

model = HIDE([reference], [projection])
model.train(training_bulk, training_composition, iter=1000)

domain_bulk, _ = create_bulks(
    adata,
    n_bulks=1000,
    n_cells_per_bulk=100,
    celltype_col="cell_type",
)
domain_bulk = domain_bulk.loc[common_genes]
alpha = get_domain_transfer_factor(healthy_bulk, domain_bulk)
alpha_inverse = (1 / alpha).replace([np.inf, -np.inf], 0).fillna(1.0)
bulk = bulk.mul(alpha_inverse, axis=0)

results = model.predict(
    bulk.loc[model.gene_labels],
    norm=True,
)["prediction"]
```
