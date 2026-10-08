# Python API

The Python API is organized into preprocessing, deconvolution, model,
statistical and visualization components.

The main workflow is:

1. Load annotated single-cell data and bulk expression data.
2. Create reference profiles and simulated training bulks.
3. Train the HIDE model.
4. Apply domain-transfer and library-size corrections where required.
5. Predict cell-type proportions.
6. Analyze and visualize the resulting compositions.

## Public API

The package-level convenience function is:

::: hide_deconv.deconvolution

The lower-level API is grouped by responsibility:

- [Preprocessing](preprocessing.md)
- [Deconvolution](deconvolution.md)
- [Models](models.md)
- [Statistics](statistics.md)
- [Visualization](visualization.md)
