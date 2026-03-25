# SoundProcessing

A collection of Python scripts and modules for acoustic data loading, calibration, spectral analysis, and batch processing.

## Goals

- Support automated acoustic/soundscape analysis workflows
- Standardize calibration and SPL/comparison metrics
- Enable batch processing of long recordings and WAV datasets


## Repository structure

- `WavBatchProcessing/` - Batch processing utilities for WAV files. Transforms wav files into 1-second, TOB sound pressure level matrices saved to .tab files for post-processing. 
TOB sound pressure level estimation follows guidelines from project JOMOPANS (http://northsearegion.eu/jomopans/about/index.html), outlined in their Standards for Data Processing report (https://vb.northsearegion.eu/public/files/repository/20190329144007_Jomopans_WP3standardDataProcessing_v15.pdf).

  User can select the output time window and maximum frequency of output (default 50k Hz)

  Options for 1-second, 1Hz resolution and variable temporal resolution windows to be implemented.

- `Soundtrap_summary.py` - Main summary script for audio data pre-processed through WavBatchProcessing. 
Computes and exports summary tables including:
  - LTSA (long format for R viz)
  - Quantile Spectral plots
  - SPL timeseries for user-selected TOBs (long format for R viz)
  - Loud Event detections

  Users should define the summary window and select which outputs are created. Other psecs for each output are also editable in set up.

- `Py_functions/` - Reusable function library with submodules:
  - `calibration.py` - Calibration routines (e.g., hydrophone sensitivity, SPL correction).
  - `data_loading/` - Data loading helpers (CSV, metadata, sensor streams).
  - `LTSA_Jomopans/` - Long-term spectral analysis functions.
  - `spectral_quantiles.py` - Spectral quantile metrics (e.g., percentile-based noise levels).
  - `spl_calculation.py` - Sound pressure level computation helpers.

## Quick start

1. Ensure Python 3.8+ is installed.
2. Install required libraries:

```bash
pip install -r requirements.txt
```

3. Convert WAV files into 1-second third-octave-band SPL estimates with `WavBatchProcessing`.

```bash
cd WavBatchProcessing
python process_wav_batch.py --input /path/to/wav --output /path/to/wavbatch-out
```

- Use `Soundtrap_summary.py` on the `WavBatchProcessing` output to produce LTSA, spectral quantile plots, and third-octave-band SPL estimates.

```bash
cd ..
python Soundtrap_summary.py --input /path/to/wavbatch-out --output /path/to/soundtrap-summary
```

(Replace script names/CLI args with actual ones in your repo; inspect `WavBatchProcessing` and `Soundtrap_summary.py`.)


