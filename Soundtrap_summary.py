"""
Acoustic Summary Script for SoundTrap TOB LTSA outputs
=======================================================
Reads *_ltsa_TOB_JOMOPANS.tab files produced by Soundtrap_batch.py and
produces three CSV files covering the full deployment period, clipped to
1 hour after deployment and 1 hour before retrieval using a metadata file.

Output files
------------
  1. ltsa_<N>h.csv
       Long-format: DateTime | TOB | SPL
       One row per (window, TOB band). SPL = median dB re 1 µPa across
       all 1-second bins in that window.

  2. spl_timeseries_<N>h.csv
       Long-format: DateTime | band_hz | mean | median | std | min |
                    p05 | p25 | p50 | p75 | p95 | max | n_samples
       Per-window statistics for each band listed in SPL_BANDS_HZ.

  3. quantile_spectral_density_<N>h.csv
       Long-format: percentile | TOB | SPL
       Percentiles computed across ALL data (not per window).
       One row per (percentile, TOB band).

Usage
-----
Edit the CONFIGURATION block below, then run:
    python summarise_acoustic.py

Dependencies: numpy, pandas, openpyxl  (no matplotlib required)
"""

import os
import glob
import warnings
import numpy as np
import pandas as pd
from datetime import timedelta

warnings.filterwarnings("ignore")

# ============================================================================
# CONFIGURATION  –  edit these before running
# ============================================================================

INPUT_PATH  = "E:\\S3\\Processed\\SoundTrap\\S21\\S21_9471_260213"   # folder with .tab files
OUTPUT_PATH = "C:\\Users\\fg363\\OneDrive - University of Exeter\\Projects\\S3 Project - Documents\\Data\\SoundTrap_Summaries\\S21\\S21_9471_260213"              # where CSVs are saved

# Deployment metadata file (Excel)
METADATA_FILE = (
   "C:\\Users\\fg363\\OneDrive - University of Exeter\\Projects\\S3 Project - Documents\\Data\\MetaData.xlsx"
)

# Station and hydrophone serial to look up in the metadata file
# These must match the Station and Hydrophone columns in METADATA_FILE
STATION    = "S21"    # e.g. "S21"
HYDROPHONE = "9471"   # serial number as string

# Buffer applied to deployment/retrieval times (hours)
DEPLOY_BUFFER_HOURS   = 1   # clip start = DateTime_deploy_UTC   + this
RETRIEVE_BUFFER_HOURS = 1   # clip end   = DateTime_retrieve_UTC - this

# Summary window duration in hours.
# Sub-daily: must divide 24 evenly (1, 2, 3, 4, 6, 8, 12, 24).
# Multi-day: any multiple of 24 (48, 72, …).
WINDOW_HOURS = 12

# Percentiles written to spl_timeseries and quantile_spectral_density CSVs
PERCENTILES = [1, 5, 25, 50, 75, 95, 99]

# TOB bands (Hz) to include in the SPL time-series summary
SPL_BANDS_HZ = [63, 125, 200, 500]

# ============================================================================
# DEPLOYMENT CLIPPING
# ============================================================================

def load_deployment_window(metadata_file, station, hydrophone,
                           deploy_buffer_h, retrieve_buffer_h):
    """
    Read MetaData.xlsx and return (clip_start, clip_end) for the given
    Station / Hydrophone combination.

    Expected columns: Station, Hydrophone,
                      DateTime_deploy_UTC, DateTime_retrieve_UTC
    Datetime format:  YYYY-MM-DD HH:MM:SS
    """
    try:
        meta = pd.read_excel(metadata_file, dtype=str)
    except Exception as exc:
        raise RuntimeError(f"Could not read metadata file: {exc}")

    # Normalise column names (strip whitespace)
    meta.columns = meta.columns.str.strip()

    # Match station and hydrophone (case-insensitive, strip whitespace)
    mask = (
        meta["Station"].str.strip().str.upper()    == station.upper() ) & (
        meta["Hydrophone"].str.strip().str.upper() == str(hydrophone).upper()
    )
    matches = meta[mask]

    if matches.empty:
        raise ValueError(
            f"No metadata row found for Station='{station}', "
            f"Hydrophone='{hydrophone}'. Check STATION / HYDROPHONE config."
        )
    if len(matches) > 1:
        warnings.warn(
            f"Multiple metadata rows found for Station='{station}', "
            f"Hydrophone='{hydrophone}'. Using the first row."
        )

    row = matches.iloc[0]
    deploy   = pd.to_datetime(row["DateTime_deploy_UTC"])
    retrieve = pd.to_datetime(row["DateTime_retrieve_UTC"])

    clip_start = deploy   + timedelta(hours=deploy_buffer_h)
    clip_end   = retrieve - timedelta(hours=retrieve_buffer_h)

    if clip_start >= clip_end:
        raise ValueError(
            f"Clip window is empty or negative after applying buffers: "
            f"{clip_start} → {clip_end}"
        )

    return clip_start, clip_end


# ============================================================================
# LOADING HELPERS
# ============================================================================

def find_tab_files(folder):
    """Return sorted list of *_ltsa_TOB_JOMOPANS.tab files."""
    files = sorted(glob.glob(
        os.path.join(folder, "**", "*_ltsa_TOB_JOMOPANS.tab"), recursive=True
    ))
    if not files:
        files = sorted(glob.glob(
            os.path.join(folder, "**", "*.tab"), recursive=True
        ))
    return files


def load_tab_file(filepath):
    """
    Load one .tab file.
    Returns a DataFrame with a DatetimeIndex and one column per TOB frequency,
    or None on failure.
    """
    try:
        df = pd.read_csv(filepath, sep="\t", low_memory=False)

        dt_col = next(
            (c for c in df.columns if "datetime" in c.lower()), None
        )
        if dt_col is None:
            warnings.warn(f"No datetime column in {filepath} – skipping.")
            return None

        df[dt_col] = pd.to_datetime(df[dt_col])
        df = df.set_index(dt_col).sort_index()

        freq_cols = [c for c in df.columns if c.endswith("Hz")]
        if not freq_cols:
            warnings.warn(f"No frequency columns in {filepath} – skipping.")
            return None

        return df[freq_cols].apply(pd.to_numeric, errors="coerce")

    except Exception as exc:
        warnings.warn(f"Could not load {filepath}: {exc}")
        return None


def extract_freq_hz(columns):
    """Parse numeric Hz values from column names like '63.0Hz'."""
    return np.array([float(c.replace("Hz", "")) for c in columns])


def nearest_band_col(target_hz, freq_array):
    """Return the index of the frequency closest to target_hz."""
    return int(np.argmin(np.abs(freq_array - target_hz)))


# ============================================================================
# WINDOW ASSIGNMENT
# ============================================================================

def assign_window(dt_index, window_hours):
    """
    Floor each timestamp to the nearest window_hours boundary.
    Works for sub-daily windows (must divide 24 evenly) and multi-day
    multiples of 24. Windows anchor to midnight UTC.
    """
    return dt_index.floor(f"{window_hours}h")


# ============================================================================
# SUMMARY COMPUTATIONS
# ============================================================================

def compute_ltsa(grouped, freq_cols):
    """
    Median SPL per TOB per window.
    Returns long-format DataFrame: DateTime | TOB | SPL
    """
    rows = []
    for window_start, grp in grouped:
        medians = np.nanmedian(grp[freq_cols].values.astype(float), axis=0)
        for col, spl in zip(freq_cols, medians):
            rows.append({
                "DateTime": window_start,
                "TOB":      float(col.replace("Hz", "")),
                "SPL":      round(float(spl), 3),
            })
    return (pd.DataFrame(rows, columns=["DateTime", "TOB", "SPL"])
              .sort_values(["DateTime", "TOB"])
              .reset_index(drop=True))


def compute_spl_timeseries(grouped, freq_cols, freqs_hz,
                           target_bands_hz, percentiles):
    """
    Per-window descriptive statistics for each target TOB band.
    Returns long-format DataFrame:
        DateTime | band_hz | mean | median | std | min | p05…p95 | max | n_samples
    """
    pct_labels = [f"p{p:02d}" for p in percentiles]
    rows = []

    for window_start, grp in grouped:
        data = grp[freq_cols].values.astype(float)
        for target in target_bands_hz:
            idx   = nearest_band_col(target, freqs_hz)
            valid = data[:, idx]
            valid = valid[~np.isnan(valid)]
            if len(valid) == 0:
                continue
            pct_vals = np.nanpercentile(valid, percentiles)
            row = {
                "DateTime":  window_start,
                "band_hz":   freqs_hz[idx],
                "mean":      np.nanmean(valid),
                "median":    np.nanmedian(valid),
                "std":       np.nanstd(valid),
                "min":       np.nanmin(valid),
                "max":       np.nanmax(valid),
                "n_samples": len(valid),
            }
            for label, val in zip(pct_labels, pct_vals):
                row[label] = val
            rows.append(row)

    col_order = (["DateTime", "band_hz", "mean", "median", "std", "min"]
                 + pct_labels + ["max", "n_samples"])
    return (pd.DataFrame(rows, columns=col_order)
              .sort_values(["DateTime", "band_hz"])
              .reset_index(drop=True))


def compute_qsd(full_df, freq_cols, percentiles):
    """
    Quantile Spectral Density computed across ALL data (not per window).
    Returns long-format DataFrame: percentile | TOB | SPL
    """
    data = full_df[freq_cols].values.astype(float)   # (N_total, N_freq)
    pct_matrix = np.nanpercentile(data, percentiles, axis=0)  # (n_pct, n_freq)

    rows = []
    for i, pct in enumerate(percentiles):
        for j, col in enumerate(freq_cols):
            rows.append({
                "percentile": pct,
                "TOB":        float(col.replace("Hz", "")),
                "SPL":        round(float(pct_matrix[i, j]), 3),
            })

    return (pd.DataFrame(rows, columns=["percentile", "TOB", "SPL"])
              .sort_values(["percentile", "TOB"])
              .reset_index(drop=True))


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 70)
    print(f"SoundTrap {WINDOW_HOURS}-Hour Acoustic Summary  →  CSV outputs")
    print("=" * 70)

    if not os.path.isdir(INPUT_PATH):
        print(f"\nERROR: INPUT_PATH not found:\n  {INPUT_PATH}")
        return

    os.makedirs(OUTPUT_PATH, exist_ok=True)
    print(f"\nInput    : {INPUT_PATH}")
    print(f"Output   : {OUTPUT_PATH}")
    print(f"Metadata : {METADATA_FILE}")

    # ------------------------------------------------------------------
    # Load deployment window from metadata
    # ------------------------------------------------------------------
    print(f"\nLooking up deployment window for "
          f"Station={STATION}, Hydrophone={HYDROPHONE} …")
    try:
        clip_start, clip_end = load_deployment_window(
            METADATA_FILE, STATION, HYDROPHONE,
            DEPLOY_BUFFER_HOURS, RETRIEVE_BUFFER_HOURS
        )
    except Exception as exc:
        print(f"\nERROR: {exc}")
        return

    print(f"  Deploy + {DEPLOY_BUFFER_HOURS}h  : {clip_start}")
    print(f"  Retrieve - {RETRIEVE_BUFFER_HOURS}h : {clip_end}")

    # ------------------------------------------------------------------
    # Load all .tab files
    # ------------------------------------------------------------------
    tab_files = find_tab_files(INPUT_PATH)
    if not tab_files:
        print(f"\nERROR: No *_ltsa_TOB_JOMOPANS.tab files found in {INPUT_PATH}")
        return

    print(f"\nFound {len(tab_files)} .tab file(s). Loading …")
    frames = []
    for i, fp in enumerate(tab_files, 1):
        df = load_tab_file(fp)
        if df is not None:
            frames.append(df)
        if i % 100 == 0 or i == len(tab_files):
            print(f"  Loaded {i}/{len(tab_files)} …")

    if not frames:
        print("\nERROR: No data could be loaded. Check file format.")
        return

    print("\nMerging …")
    full_df = pd.concat(frames).sort_index()
    full_df = full_df[~full_df.index.duplicated(keep="first")]

    print(f"  Raw span        : {full_df.index.min()}  →  {full_df.index.max()}")
    print(f"  Raw 1-s bins    : {len(full_df):,}")

    # ------------------------------------------------------------------
    # Clip to deployment window
    # ------------------------------------------------------------------
    full_df = full_df.loc[
        (full_df.index >= clip_start) & (full_df.index <= clip_end)
    ]

    if full_df.empty:
        print("\nERROR: No data remains after clipping to deployment window.")
        return

    freq_cols = list(full_df.columns)
    freqs_hz  = extract_freq_hz(freq_cols)

    print(f"  Clipped span    : {full_df.index.min()}  →  {full_df.index.max()}")
    print(f"  Clipped 1-s bins: {len(full_df):,}")
    print(f"  TOB bands       : {freqs_hz[0]:.0f}–{freqs_hz[-1]:.0f} Hz  "
          f"({len(freq_cols)} bands)")

    # ------------------------------------------------------------------
    # Assign every row to a window
    # ------------------------------------------------------------------
    print(f"\nAssigning {WINDOW_HOURS}-hour windows …")
    full_df["window_start"] = assign_window(full_df.index, WINDOW_HOURS)
    grouped   = full_df.groupby("window_start")
    n_windows = len(grouped)
    print(f"  {n_windows} windows")

    suffix = f"{WINDOW_HOURS}h"

    # ------------------------------------------------------------------
    # 1. LTSA CSV  –  long format: DateTime | TOB | SPL
    # ------------------------------------------------------------------
    print(f"\n[1/3] Computing LTSA (median SPL per window per TOB) …")
    ltsa_df   = compute_ltsa(grouped, freq_cols)
    ltsa_path = os.path.join(OUTPUT_PATH, f"ltsa_{suffix}.csv")
    ltsa_df.to_csv(ltsa_path, index=False)
    print(f"  Saved: ltsa_{suffix}.csv  ({len(ltsa_df):,} rows)")

    # ------------------------------------------------------------------
    # 2. SPL time-series CSV
    # ------------------------------------------------------------------
    print(f"\n[2/3] Computing SPL summaries for {SPL_BANDS_HZ} Hz bands …")
    spl_df   = compute_spl_timeseries(
        grouped, freq_cols, freqs_hz, SPL_BANDS_HZ, PERCENTILES
    )
    spl_path = os.path.join(OUTPUT_PATH, f"spl_timeseries_{suffix}.csv")
    spl_df.round(3).to_csv(spl_path, index=False)
    print(f"  Saved: spl_timeseries_{suffix}.csv  ({len(spl_df):,} rows)")

    # ------------------------------------------------------------------
    # 3. Quantile Spectral Density CSV  –  computed across ALL data
    # ------------------------------------------------------------------
    print(f"\n[3/3] Computing Quantile Spectral Density "
          f"(P{PERCENTILES}) across all data …")
    qsd_df   = compute_qsd(full_df, freq_cols, PERCENTILES)
    qsd_path = os.path.join(OUTPUT_PATH, f"quantile_spectral_density_{suffix}.csv")
    qsd_df.to_csv(qsd_path, index=False)
    print(f"  Saved: quantile_spectral_density_{suffix}.csv  ({len(qsd_df):,} rows)")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Complete.  Output files:")
    print(f"  {ltsa_path}")
    print(f"  {spl_path}")
    print(f"  {qsd_path}")
    print()
    print("CSV schemas:")
    print(f"  ltsa_{suffix}.csv")
    print("    DateTime | TOB | SPL")
    print()
    print(f"  spl_timeseries_{suffix}.csv")
    print("    DateTime | band_hz | mean | median | std | min | "
          + " | ".join(f"p{p:02d}" for p in PERCENTILES)
          + " | max | n_samples")
    print()
    print(f"  quantile_spectral_density_{suffix}.csv")
    print("    percentile | TOB | SPL  (computed across full deployment)")
    print("=" * 70)


if __name__ == "__main__":
    main()