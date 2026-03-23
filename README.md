# SoundProcessing

A collection of Python scripts and modules for acoustic data loading, calibration, spectral analysis, and batch processing.

## Repository structure

- `Soundtrap_summary.py` - Main summary script for Soundtrap data; likely computes aggregate metrics and/or exports summary tables.
- `WavBatchProcessing/` - Batch processing utilities for WAV files (e.g., LTSA, spectrogram segmentation, and summary features).
- `Py_functions/` - Reusable function library with submodules:
  - `calibration.py` - Calibration routines (e.g., hydrophone sensitivity, SPL correction).
  - `data_loading/` - Data loading helpers (CSV, metadata, sensor streams).
  - `LTSA_Jomopans/` - Long-term spectral analysis functions.
  - `spectral_quantiles.py` - Spectral quantile metrics (e.g., percentile-based noise levels).
  - `spl_calculation.py` - Sound pressure level computation helpers.

## Goals

- Support automated acoustic/soundscape analysis workflows
- Standardize calibration and SPL/comparison metrics
- Enable batch processing of long recordings and WAV datasets

## Quick start

1. Ensure Python 3.8+ is installed.
2. Install required libraries:

```bash
pip install -r requirements.txt
```

3. Correct processing workflow is:

- Step 1: Convert WAV files into 1-second third-octave-band SPL estimates with `WavBatchProcessing`.

```bash
cd WavBatchProcessing
python process_wav_batch.py --input /path/to/wav --output /path/to/wavbatch-out
```

- Step 2: Use `Soundtrap_summary.py` on the `WavBatchProcessing` output to produce LTSA, spectral quantile plots, and third-octave-band SPL estimates.

```bash
cd ..
python Soundtrap_summary.py --input /path/to/wavbatch-out --output /path/to/soundtrap-summary
```

- Optional: Calibration utilities from `Py_functions` can be used in pre/post processing:

```bash
python -m Py_functions.calibration --help
```

(Replace script names/CLI args with actual ones in your repo; inspect `WavBatchProcessing` and `Soundtrap_summary.py`.)

## How to discover available scripts and options

- List Python files:

```bash
find . -name "*.py" -maxdepth 3
```

- Inspect docstrings or CLI help:

```bash
python Soundtrap_summary.py --help
python -m Py_functions.spectral_quantiles --help
```

## Output expectations

- CSV/TSV or JSON summaries of acoustic metrics.
- Figures (e.g., spectrograms or LTSA plots) if the scripts generate plots.
- Calibrated SPL data.

## Notes

- Update the commands above if specific argument names differ.
- Dependencies are listed in `requirements.txt` for easy installation.

