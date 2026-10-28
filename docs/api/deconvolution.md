# Deconvolution

## High-level API

`deconvolution` runs preprocessing, model training and prediction in one
call. It returns one composition DataFrame for each requested cell type
layer.

::: hide_deconv.deconvolution

## Prediction with domain transfer

`predict_deconvolution_results` applies a trained model to bulk samples and
corrects for domain transfer between simulated and real bulks.

::: hide_deconv.pipelines.deconvolve_hide_pipeline.predict_deconvolution_results
