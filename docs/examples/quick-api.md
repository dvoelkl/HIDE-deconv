# Quick API example

The package-level API runs preprocessing, training and deconvolution in one
call. The example runs on the [example data](example-data.md), which can be
created with `hide-deconv example`.

```python
import anndata as ad
import pandas as pd

from hide_deconv import deconvolution

adata = ad.read_h5ad("single_cells.h5ad")
bulk = pd.read_csv("bulk.csv", index_col=0)

# Reduced settings for the example data. For real data we recommend the
# defaults of 5000 genes and at least 10000 training bulks.
results = deconvolution(
    adata,
    bulk,
    celltype_cols=["cell_type", "major"],
    n_genes=1000,
    n_train_bulks=5000,
)
```

`results` is a list of composition DataFrames, one for each requested cell
type layer. The first element represents the finest cell type layer.
