# E-Nose Gas Classification & H2S Regression

This repo contains analysis scripts for an electronic-nose (16-sensor) dataset. There are two main tracks:

- Multi-gas classification on run/window features (baseline classical + deep models).
- H2S concentration regression with diagnostic-first baselines and model comparisons.

Most scripts are designed as standalone entry points that read CSV/text files and write figures/CSVs to an output directory.

## Contents

- `run_h2s_concentration_regression.py`: end-to-end H2S regression pipeline (feature extraction, LOCO + within-concentration evaluation, diagnostics, model artifacts).
- `fig_h2s_regression_main4.py`: builds the main H2S regression figure (4 representative models) from `baseline_comparison.csv`.
- `baseline_run_window_pipeline_rf.py`: run-level split, windowed augmentation, SVM/KNN/RF + optional CNN/LSTM baselines for multi-gas classification.
- `figure_fingerprint_pca.py`: fingerprint heatmap + PCA scatter for run-level features.
- `plot_metrics_summary.py`: publication-quality metrics summary heatmaps.
- `plot_confmats_5baselines_rasterpanels.py`: rasterized confusion-matrix panels with vector text.
- `tools/`: small data preparation and visualization utilities.

## Environment

Install Python dependencies from `requirements.txt`:

```bash
pip install -r requirements.txt
```

Notes:
- Torch is only required if you enable deep models in `baseline_run_window_pipeline_rf.py`.
- Many scripts use absolute default paths (see “Paths & data locations”). Override them or edit the constants for your environment.

## H2S concentration regression

### Expected input

`run_h2s_concentration_regression.py` expects a directory of `*ppm.txt` files. Each file should contain:

- Column 0: time (ms or s)
- Columns 1..K: sensor resistance values (K sensors)
- Filename contains `XXppm` (e.g., `H2S_10ppm.txt`) so the ppm can be parsed

### Run

```bash
python run_h2s_concentration_regression.py \
  --data_dir /path/to/h2s_txt \
  --out_dir /path/to/results_h2s_regression \
  --exposure_s 60 \
  --purge_s 120 \
  --n_cycles 10
```

Outputs (written to `--out_dir`):

- `dataset_cycles.csv`: per-cycle features and targets
- `metrics.csv`: global metrics for LOCO + within-concentration schemes
- `fold_predictions.csv`: per-sample predictions for all models
- `per_concentration_errors.csv`: MAE by true concentration
- `baseline_comparison.csv`: baseline vs ML comparisons
- `feature_monotonicity.csv`: Spearman monotonicity per feature
- `diagnostics_dose_response.png`
- `diagnostics_parity_log.png`
- `diagnostics_residuals_by_conc.png`
- `best_model.joblib`
- `feature_schema.json`


### Main H2S regression figure

Once `baseline_comparison.csv` exists:

1. Update paths at the top of `fig_h2s_regression_main4.py` or place the CSV at the default location.
2. Run:

```bash
python fig_h2s_regression_main4.py
```

Outputs:

- `fig_h2s_regression_main4_v3.pdf`
- `fig_h2s_regression_main4_v3.png`
- `tab_h2s_main4.tex`
- `summary_h2s_ranked.csv`

## Multi-gas classification baselines

### Expected input

`baseline_run_window_pipeline_rf.py` expects a frame-wise CSV (e.g., `all_gases_framewise.csv`) with at least:

- `time`
- `16 sensor columns`
- `gas_id`, `run_id`
- `t_exposure_start_ms`, `t_exposure_end_ms`

### Run

```bash
python baseline_run_window_pipeline_rf.py \
  --csv /path/to/all_gases_framewise.csv \
  --outdir outputs_baseline \
  --phase_mode 3phase
```

Outputs:

- Console summary of mean±std metrics (window-level and run-level)
- Confusion-matrix PNGs under `--outdir`
- Metrics summary CSV (if enabled by your configuration)

### Plotting summaries

```bash
python plot_metrics_summary.py \
  --csv /path/to/metrics_summary.csv \
  --out_dir /path/to/outputs_baseline
```

```bash
python plot_confmats_5baselines_rasterpanels.py \
  --input_dir /path/to/outputs_baseline \
  --out_pdf confmats_panels.pdf
```

## Fingerprint + PCA figure

`figure_fingerprint_pca.py` reads a frame-wise CSV and produces a fingerprint heatmap + PCA scatter. It uses hard-coded paths by default:

- `DATA_PATH = /home/chunzhi/Desktop/nose/Network/all_gases_framewise.csv`
- `OUTPUT_PNG = figure_fingerprint_pca.png`
- `OUTPUT_PDF = figure_fingerprint_pca.pdf`

Update the constants or place your CSV accordingly, then run:

```bash
python figure_fingerprint_pca.py
```

## Tools

Utilities under `tools/` help with data preparation and inspection:

- `tools/fuse_data.py`: add metadata columns to per-gas files.
- `tools/fuse_csv.py`: merge per-gas CSVs into `all_gases_framewise.csv`.
- `tools/plot_16_sensors.py`: quick 16-sensor time-series plots.
- `tools/plot_16_sensors_skip_flush600s.py`: same as above with initial flush trimming.
- `tools/drop_flush_prefix.py`: drop initial flush seconds from a raw file.
- `tools/trim_tail.py`: truncate a file to a target duration.
- `tools/strip_time00.py`: fix time tokens with leading zeros.
- `tools/test_csv.py`: sanity-check run-level stratified split counts.

## Paths & data locations

Several scripts use absolute default paths under `/home/chunzhi/Desktop/nose/Network/...`. If you run elsewhere:

1. Pass CLI arguments where available (recommended).
2. Otherwise edit the constants at the top of the script.

## Repro tips

- Set `--seed` on the pipelines for repeatable splits and model behavior.
- For LOCO evaluation in H2S regression, per-fold R² is invalid (targets are constant per fold). The pipeline computes global metrics on concatenated predictions instead.

## License

MIT
