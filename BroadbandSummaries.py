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

    4. loud_events_<STATION>.csv
       One row per detected loud event (broadband):
       event_id | start_time | end_time | duration_s | peak_spl_dB |
       leq_dB | delta_peak_dB | baseline_spl_dB | impulsivity_dB |
       spectral_centroid_hz
       See LOUD EVENT DETECTION section below for parameter descriptions.

Output selection
----------------
  Set CREATE_LTSA_CSV, CREATE_SPL_TIMESERIES_CSV, CREATE_QSD_CSV
  in the CONFIGURATION block to choose which files to generate.

Usage
-----
Edit the CONFIGURATION block below, then run:
    python summarise_acoustic.py

Dependencies: numpy, pandas, openpyxl
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

# Output selection (set to False to skip writing that file)
CREATE_LTSA_CSV             = True
CREATE_SPL_TIMESERIES_CSV   = True
CREATE_QSD_CSV              = True
CREATE_LOUD_CSV             = True 

# ============================================================================
# LOUD EVENT DETECTION  –  edit these before running
# ============================================================================

# Frequency range (Hz) used to compute broadband SPL via power summation.
# Set to None to use all available bands, or restrict to exclude noisy
# ends of the spectrum (e.g. wind noise below 50 Hz).
#   Example: BROADBAND_FMIN_HZ = 50.0 ; BROADBAND_FMAX_HZ = 8000.0
BROADBAND_FMIN_HZ = 100.0     # lower bound (Hz); set None for no lower limit
BROADBAND_FMAX_HZ = 50000.0   # upper bound (Hz); set None for no upper limit

# Baseline method: rolling low-percentile of broadband SPL.
# A rolling window tracks the "quiet floor" of the soundscape and
# naturally adapts to each station's ambient level — making the detector
# transferable across sites without manual re-tuning.
BASELINE_WINDOW_HOURS  = 1     # rolling window length (hours → seconds internally)
BASELINE_PERCENTILE    = 10    # low percentile that represents the ambient floor
                               # 10th pct is robust to transient events

# Detection threshold: a second is flagged if
#   L_broadband(t)  >  L_baseline(t)  +  DETECTION_THRESHOLD_DB
# Suggested starting values:
#   3 dB  → sensitive  (catches moderate rises, more false positives)
#   6 dB  → moderate   (roughly double the acoustic power above ambient)
#  10 dB  → conservative (clearly prominent events only)
DETECTION_THRESHOLD_DB = 10.0

# Post-processing: minimum number of *consecutive* flagged seconds to
# retain as a real event (rejects single-sample spikes / impulses).
EVENT_MIN_DURATION_S = 3

# Post-processing: merge two detections separated by fewer than this many
# quiet seconds into a single event (prevents one event being fragmented
# by brief dips below threshold).
EVENT_MERGE_GAP_S = 30

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
# LOUD EVENT DETECTION
# ============================================================================

def compute_broadband_spl(full_df, freq_cols, freqs_hz,
                          fmin_hz=None, fmax_hz=None):
    """
    Collapse third-octave band SPLs into a single broadband SPL time series
    using incoherent power summation:

        L_broad(t) = 10 * log10( sum_i( 10^(L_i(t) / 10) ) )

    Parameters
    ----------
    full_df   : DataFrame with DatetimeIndex, columns = freq_cols
    freq_cols : list of column names ending in 'Hz'
    freqs_hz  : numpy array of corresponding frequencies
    fmin_hz   : lower frequency limit for summation (None = no limit)
    fmax_hz   : upper frequency limit for summation (None = no limit)

    Returns
    -------
    broadband : pd.Series (DatetimeIndex) of broadband SPL in dB re 1 µPa
    bb_cols   : list of column names that were summed (for reference)
    """
    mask = np.ones(len(freqs_hz), dtype=bool)
    if fmin_hz is not None:
        mask &= freqs_hz >= fmin_hz
    if fmax_hz is not None:
        mask &= freqs_hz <= fmax_hz

    bb_cols = [c for c, m in zip(freq_cols, mask) if m]
    if not bb_cols:
        raise ValueError(
            f"No TOB bands remain after applying frequency limits "
            f"[{fmin_hz}, {fmax_hz}] Hz. Check BROADBAND_FMIN_HZ / BROADBAND_FMAX_HZ."
        )

    data = full_df[bb_cols].values.astype(float)          # (N, n_bands)
    power = np.power(10.0, data / 10.0)                    # linear pressure²
    broadband_linear = np.nansum(power, axis=1)            # sum across bands
    broadband_db = 10.0 * np.log10(
        np.where(broadband_linear > 0, broadband_linear, np.nan)
    )
    return pd.Series(broadband_db, index=full_df.index, name="broadband_spl"), bb_cols


def compute_rolling_baseline(broadband_spl, window_hours, percentile):
    """
    Compute a rolling low-percentile baseline of the broadband SPL.

    The rolling window is applied symmetrically (centre=True) so the baseline
    is not biased towards past values.  At the edges of the series the window
    is automatically narrowed (min_periods=1).

    Parameters
    ----------
    broadband_spl : pd.Series with DatetimeIndex at 1-second resolution
    window_hours  : rolling window length in hours
    percentile    : percentile to use as the baseline (e.g. 10)

    Returns
    -------
    baseline : pd.Series aligned with broadband_spl
    """
    window_s = int(window_hours * 3600)   # hours → seconds

    # pandas rolling quantile expects q in [0, 1]
    baseline = (
        broadband_spl
        .rolling(window=window_s, center=True, min_periods=1)
        .quantile(percentile / 100.0)
    )
    return baseline.rename("baseline_spl")


def detect_loud_events(broadband_spl, baseline,
                       threshold_db,
                       min_duration_s,
                       merge_gap_s,
                       full_df,
                       freq_cols,
                       freqs_hz):
    """
    Detect loud events from the broadband SPL and rolling baseline.

    A second is flagged when:
        broadband_spl(t) > baseline(t) + threshold_db

    Short detections are discarded and nearby detections are merged before
    event-level statistics are computed.

    Parameters
    ----------
    broadband_spl  : pd.Series (1-s broadband SPL)
    baseline       : pd.Series (rolling low-percentile baseline)
    threshold_db   : dB excess above baseline required to flag a second
    min_duration_s : minimum consecutive flagged seconds for a valid event
    merge_gap_s    : gap in seconds below which two events are merged
    full_df        : original DataFrame (used to compute spectral centroid)
    freq_cols      : list of TOB column names
    freqs_hz       : numpy array of TOB frequencies (Hz)

    Returns
    -------
    events_df : DataFrame with one row per loud event
    """

    # --- 1. Flag individual seconds ---
    flagged = (broadband_spl > (baseline + threshold_db)).fillna(False)

    # Convert to integer array for run-length logic
    flag_arr = flagged.values.astype(int)
    times    = broadband_spl.index

    # --- 2. Identify contiguous runs of flagged seconds ---
    # Pad with zeros so diff picks up edges at start/end
    padded = np.concatenate([[0], flag_arr, [0]])
    diff   = np.diff(padded.astype(int))
    starts = np.where(diff ==  1)[0]   # index into original array
    ends   = np.where(diff == -1)[0]   # exclusive end

    # Each run spans times[starts[i]] .. times[ends[i]-1]
    runs = list(zip(starts, ends))   # (start_idx, end_idx_exclusive)

    if not runs:
        return pd.DataFrame()

    # --- 3. Discard runs shorter than min_duration_s ---
    runs = [(s, e) for s, e in runs if (e - s) >= min_duration_s]
    if not runs:
        return pd.DataFrame()

    # --- 4. Merge runs with inter-event gap < merge_gap_s ---
    merged = [runs[0]]
    for s, e in runs[1:]:
        prev_s, prev_e = merged[-1]
        gap_s = s - prev_e          # seconds between end of last and start of this
        if gap_s < merge_gap_s:
            merged[-1] = (prev_s, e)
        else:
            merged.append((s, e))

    # --- 5. Compute per-event statistics ---
    rows = []
    tob_data = full_df[freq_cols].values.astype(float)

    for event_id, (s_idx, e_idx) in enumerate(merged, start=1):
        t_start = times[s_idx]
        t_end   = times[e_idx - 1]          # inclusive last second

        bb_seg    = broadband_spl.iloc[s_idx:e_idx].values
        base_seg  = baseline.iloc[s_idx:e_idx].values
        tob_seg   = tob_data[s_idx:e_idx, :]  # (dur, n_bands)

        valid_bb = bb_seg[~np.isnan(bb_seg)]
        if len(valid_bb) == 0:
            continue

        peak_spl   = float(np.nanmax(bb_seg))
        base_mean  = float(np.nanmean(base_seg))
        delta_peak = round(peak_spl - base_mean, 3)

        # Leq over the event (energy-average)
        leq = 10.0 * np.log10(np.nanmean(10.0 ** (bb_seg / 10.0)))

        # Impulsivity: peak − Leq  (>6 dB suggests impulsive character)
        impulsivity = round(peak_spl - leq, 3)

        # Spectral centroid: frequency-weighted mean of mean power per band
        band_means  = np.nanmean(tob_seg, axis=0)   # (n_bands,)
        band_linear = 10.0 ** (band_means / 10.0)
        total_power = np.nansum(band_linear)
        if total_power > 0:
            spectral_centroid = float(
                np.nansum(freqs_hz * band_linear) / total_power
            )
        else:
            spectral_centroid = np.nan

        rows.append({
            "event_id":            event_id,
            "start_time":          t_start,
            "end_time":            t_end,
            "duration_s":          int(e_idx - s_idx),
            "peak_spl_dB":         round(peak_spl, 3),
            "leq_dB":              round(leq, 3),
            "delta_peak_dB":       delta_peak,
            "baseline_spl_dB":     round(base_mean, 3),
            "impulsivity_dB":      impulsivity,
            "spectral_centroid_hz": round(spectral_centroid, 1),
        })

    col_order = [
        "event_id", "start_time", "end_time", "duration_s",
        "peak_spl_dB", "leq_dB", "delta_peak_dB", "baseline_spl_dB",
        "impulsivity_dB", "spectral_centroid_hz",
    ]
    return pd.DataFrame(rows, columns=col_order)

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
    ltsa_path = os.path.join(OUTPUT_PATH, f"ltsa_{suffix}.csv")
    if CREATE_LTSA_CSV:
        print(f"\n[1/4] Computing LTSA (median SPL per window per TOB) …")
        ltsa_df = compute_ltsa(grouped, freq_cols)
        ltsa_df.to_csv(ltsa_path, index=False)
        print(f"  Saved: ltsa_{suffix}.csv  ({len(ltsa_df):,} rows)")
    else:
        ltsa_df = None
        print("\n[1/4] Skipping LTSA output (CREATE_LTSA_CSV=False)")

    # ------------------------------------------------------------------
    # 2. SPL time-series CSV
    # ------------------------------------------------------------------
    spl_path = os.path.join(OUTPUT_PATH, f"spl_timeseries_{suffix}.csv")
    if CREATE_SPL_TIMESERIES_CSV:
        print(f"\n[2/4] Computing SPL summaries for {SPL_BANDS_HZ} Hz bands …")
        spl_df = compute_spl_timeseries(
            grouped, freq_cols, freqs_hz, SPL_BANDS_HZ, PERCENTILES
        )
        spl_df.round(3).to_csv(spl_path, index=False)
        print(f"  Saved: spl_timeseries_{suffix}.csv  ({len(spl_df):,} rows)")
    else:
        spl_df = None
        print("\n[2/4] Skipping SPL time-series output (CREATE_SPL_TIMESERIES_CSV=False)")

    # ------------------------------------------------------------------
    # 3. Quantile Spectral Density CSV  –  computed across ALL data
    # ------------------------------------------------------------------
    qsd_path = os.path.join(OUTPUT_PATH, f"quantile_spectral_density_{suffix}.csv")
    if CREATE_QSD_CSV:
        print(f"\n[3/4] Computing Quantile Spectral Density "
              f"(P{PERCENTILES}) across all data …")
        qsd_df = compute_qsd(full_df, freq_cols, PERCENTILES)
        qsd_df.to_csv(qsd_path, index=False)
        print(f"  Saved: quantile_spectral_density_{suffix}.csv  ({len(qsd_df):,} rows)")
    else:
        qsd_df = None
        print("\n[3/4] Skipping quantile spectral density output (CREATE_QSD_CSV=False)")

    # ------------------------------------------------------------------
    # 4. Loud Event Detection
    # ------------------------------------------------------------------
    if CREATE_LOUD_CSV:
        print(f"\n[4/4] Detecting loud events …")
        print(f"  Broadband range : {BROADBAND_FMIN_HZ} – {BROADBAND_FMAX_HZ} Hz")
        print(f"  Baseline        : rolling {BASELINE_WINDOW_HOURS}h P{BASELINE_PERCENTILE}")
        print(f"  Threshold       : baseline + {DETECTION_THRESHOLD_DB} dB")
        print(f"  Min duration    : {EVENT_MIN_DURATION_S} s")
        print(f"  Merge gap       : {EVENT_MERGE_GAP_S} s")

        # Drop the window_start helper column before SPL maths
        tob_df = full_df.drop(columns=["window_start"])

        try:
            broadband_spl, bb_cols = compute_broadband_spl(
                tob_df, freq_cols, freqs_hz,
                fmin_hz=BROADBAND_FMIN_HZ,
                fmax_hz=BROADBAND_FMAX_HZ,
            )
        except ValueError as exc:
            print(f"\nERROR in broadband computation: {exc}")
            return

        print(f"  Broadband SPL computed across {len(bb_cols)} TOB bands "
            f"({bb_cols[0]} – {bb_cols[-1]})")

        baseline = compute_rolling_baseline(
            broadband_spl,
            window_hours=BASELINE_WINDOW_HOURS,
            percentile=BASELINE_PERCENTILE,
        )

        events_df = detect_loud_events(
            broadband_spl, baseline,
            threshold_db   = DETECTION_THRESHOLD_DB,
            min_duration_s = EVENT_MIN_DURATION_S,
            merge_gap_s    = EVENT_MERGE_GAP_S,
            full_df        = tob_df,
            freq_cols      = freq_cols,
            freqs_hz       = freqs_hz,
        )

        events_path = os.path.join(OUTPUT_PATH, f"loud_events.csv")

        if events_df.empty:
            print("  No loud events detected with current parameters.")
            # Write an empty file with headers so downstream pipelines don't break
            pd.DataFrame(columns=[
                "event_id", "start_time", "end_time", "duration_s",
                "peak_spl_dB", "leq_dB", "delta_peak_dB", "baseline_spl_dB",
                "impulsivity_dB", "spectral_centroid_hz",
            ]).to_csv(events_path, index=False)
        else:
            events_df.to_csv(events_path, index=False)
            print(f"  {len(events_df)} event(s) detected.")
            print(f"  Duration   : {events_df['duration_s'].min()}–"
                f"{events_df['duration_s'].max()} s  "
                f"(median {events_df['duration_s'].median():.0f} s)")
            print(f"  Peak SPL   : {events_df['peak_spl_dB'].min():.1f}–"
                f"{events_df['peak_spl_dB'].max():.1f} dB re 1 µPa")
            print(f"  ΔL peak    : {events_df['delta_peak_dB'].min():.1f}–"
                f"{events_df['delta_peak_dB'].max():.1f} dB above baseline")

        print(f"  Saved: loud_events_{STATION}.csv  ({len(events_df)} rows)")
    else:
        events_df = None
        print("\n[4/4] Skipping loud events detection (CREATE_LOUD_CSV=False)")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Complete. Output files (generated):")
    if CREATE_LTSA_CSV:
        print(f"  {ltsa_path}")
    if CREATE_SPL_TIMESERIES_CSV:
        print(f"  {spl_path}")
    if CREATE_QSD_CSV:
        print(f"  {qsd_path}")
    if CREATE_LOUD_CSV:
        print(f"  {events_path}")
    if not (CREATE_LTSA_CSV or CREATE_SPL_TIMESERIES_CSV or CREATE_QSD_CSV or CREATE_LOUD_CSV):
        print("  (none selected; set output flags in config to True)")

    print() 
    print("CSV schemas (for enabled outputs):")
    if CREATE_LTSA_CSV:
        print(f"  ltsa_{suffix}.csv")
        print("    DateTime | TOB | SPL")
        print()
    if CREATE_SPL_TIMESERIES_CSV:
        print(f"  spl_timeseries_{suffix}.csv")
        print("    DateTime | band_hz | mean | median | std | min | "
              + " | ".join(f"p{p:02d}" for p in PERCENTILES)
              + " | max | n_samples")
        print()
    if CREATE_QSD_CSV:
        print(f"  quantile_spectral_density_{suffix}.csv")
        print("    percentile | TOB | SPL  (computed across full deployment)")
        print()
    if CREATE_LOUD_CSV:
        print(f"  loud_events_{STATION}.csv")
        print("    event_id | start_time | end_time | duration_s | "
              "peak_spl_dB | leq_dB | delta_peak_dB | baseline_spl_dB | "
              "impulsivity_dB | spectral_centroid_hz")
    print("=" * 70)


if __name__ == "__main__":
    main()