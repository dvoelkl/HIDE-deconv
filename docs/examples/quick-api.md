# Quick API example

The package-level API runs preprocessing, training and deconvolution in one
call.

```python
import anndata as ad
import pandas as pd

from hide_deconv import deconvolution

adata = ad.read_h5ad("single_cells.h5ad")
bulk = pd.read_csv("bulk.csv", index_col=0)

results = deconvolution(
    adata,
    bulk,
    celltype_cols=["cell_type", "major"],
)
```

`results` is a list of composition DataFrames, one for each requested cell
type layer. The first element represents the finest cell type layer.
