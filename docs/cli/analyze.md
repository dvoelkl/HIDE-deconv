# Analyze commands

All analysis commands use a trained HIDE-deconv project.

## PCA, UMAP and PLS-DA

```bash
hide-deconv analyze pca --path <project_dir>
hide-deconv analyze umap --path <project_dir>
hide-deconv analyze plsda --path <project_dir>
```

Existing composition datasets can be projected into the fitted analysis:

```bash
hide-deconv analyze pca \
  --path <project_dir> \
  --map-others <composition_1.csv> \
  --map-others <composition_2.csv>
```

The mapped CSV files must use the same cell type labels as the selected
composition. They are transformed after fitting and do not influence the
calculated components.

## Cohort differences

```bash
hide-deconv analyze diff --path <project_dir>
hide-deconv analyze hdiff --path <project_dir>
```

## Other analyses

```bash
hide-deconv analyze benchmark --path <project_dir>
hide-deconv analyze cluster --path <project_dir>
hide-deconv analyze survival --path <project_dir>
```
