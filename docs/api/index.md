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

To run the deconvolution with a single command, you can use:

::: hide_deconv.deconvolution

To generate synthetic data, you can use:

::: hide_deconv.generate_example

The API is grouped in modules:

- [Preprocessing](preprocessing.md)
- [Deconvolution](deconvolution.md)
- [Models](models.md)
- [Statistics](statistics.md)
- [Visualization](visualization.md)
- [Utilities](utils.md)
