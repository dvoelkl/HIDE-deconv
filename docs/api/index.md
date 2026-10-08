# Python API

The Python API is organized into preprocessing, model, pipeline, statistical
and visualization components.

The main workflow is:

1. Load annotated single-cell data and bulk expression data.
2. Create reference profiles and simulated training bulks.
3. Train the HIDE model.
4. Apply domain-transfer and library-size corrections where required.
5. Predict cell-type proportions.
6. Analyze and visualize the resulting compositions.
