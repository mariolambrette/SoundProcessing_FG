"""
SPL (Sound Pressure Level) calculation module for SoundTrap audio analysis.

Provides functions to calculate SPL in third-octave bands using various methods.
"""

import numpy as np
import pandas as pd
from scipy import signal


def third_octave_filter(center_freq, sample_rate):
    """
    Create IEC 61260 compliant third-octave bandpass filter
    
    Parameters
    ----------
    center_freq : float
        Center frequency in Hz
    sample_rate : int
        Sample rate in Hz
    
    Returns
    -------
    b, a : array
        Numerator and denominator of the IIR filter
    """
    # Third-octave bandwidth
    f_lower = center_freq / (2**(1/6))
    f_upper = center_freq * (2**(1/6))
    
    # Design Butterworth bandpass filter
    nyquist = sample_rate / 2
    low = f_lower / nyquist
    high = f_upper / nyquist
    
    # Ensure frequencies are valid
    low = max(0.001, min(low, 0.999))
    high = max(0.001, min(high, 0.999))
    
    if low >= high:
        return None, None
    
    b, a = signal.butter(3, [low, high], btype='band')
    return b, a


def calculate_spl(audio_data, sample_rate, center_freq, ref_pressure=1e-6):
    """
    Calculate SPL in dB re 1 µPa for a third-octave band
    
    Parameters
    ----------
    audio_data : array
        Audio signal in Pa
    sample_rate : int
        Sample rate in Hz
    center_freq : float
        Center frequency of third-octave band in Hz
    ref_pressure : float
        Reference pressure (default 1e-6 Pa = 1 µPa)
    
    Returns
    -------
    float
        SPL in dB re 1 µPa or NaN if calculation failed
    """
    b, a = third_octave_filter(center_freq, sample_rate)
    if b is None or a is None:
        return np.nan

    # Ensure input is numeric and has enough samples for filtfilt padding
    audio = np.asarray(audio_data, dtype=float)
    padlen = 3 * max(len(a), len(b))
    if audio.size <= padlen:
        # not enough samples to apply filtfilt safely
        return np.nan

    # Filter the signal
    try:
        filtered = signal.filtfilt(b, a, audio)
    except Exception:
        return np.nan

    # Calculate RMS pressure
    rms = np.sqrt(np.mean(filtered**2))
    
    # Convert to dB re 1 µPa
    if rms > 0:
        spl = 20 * np.log10(rms / ref_pressure)
    else:
        spl = np.nan
    
    return spl


def process_spl_timeseries(audio_data, sample_rate, tob_freqs, window_duration=60.0, 
                          downsample_sr=2000, fft_size=2048):
    """
    Calculate SPL time series in specified third-octave bands using Welch PSD.
    
    Downsamples the calibrated audio (Pa) to `downsample_sr` using linear FIR resampling,
    splits into non-overlapping windows of `window_duration` seconds, and for each window
    computes Welch PSD and integrates PSD across the third-octave band to obtain band power
    (Pa^2), then RMS and dB re 1 µPa.

    Parameters
    ----------
    audio_data : array
        Audio signal in Pa
    sample_rate : int
        Sample rate in Hz
    tob_freqs : list
        List of third-octave band center frequencies in Hz
    window_duration : float
        Duration of each window in seconds (default 60)
    downsample_sr : int
        Downsample target sample rate in Hz (default 2000)
    fft_size : int
        FFT size for Welch method (default 2048)
    
    Returns
    -------
    pd.DataFrame
        Dataframe with columns 'time_offset_s' and 'SPL_<freq>Hz' for each frequency
    """
    # If downsample_sr >= sample_rate, skip resampling
    if downsample_sr >= sample_rate:
        audio_ds = audio_data
        ds_rate = sample_rate
    else:
        # Resample using polyphase (FIR) to avoid aliasing (linear-phase)
        gcd = np.gcd(sample_rate, downsample_sr)
        up = downsample_sr // gcd
        down = sample_rate // gcd
        try:
            audio_ds = signal.resample_poly(audio_data, up, down)
            ds_rate = downsample_sr
        except Exception:
            # fallback: use simple decimation (less ideal)
            factor = int(round(sample_rate / downsample_sr))
            audio_ds = audio_data[::factor]
            ds_rate = int(sample_rate / factor)

    window_samples = int(window_duration * ds_rate)
    if window_samples <= 0:
        raise ValueError("window_duration too small for downsampled rate")

    n_windows = len(audio_ds) // window_samples
    results = []

    for i in range(n_windows):
        start_idx = i * window_samples
        end_idx = start_idx + window_samples
        window = audio_ds[start_idx:end_idx]

        time_offset = i * window_duration
        spl_values = {'time_offset_s': time_offset}

        # Compute Welch PSD for this window
        # choose nperseg conservatively
        nperseg = min(fft_size, max(256, window_samples // 8))
        try:
            f, psd = signal.welch(window, ds_rate, nperseg=nperseg, scaling='density')
        except Exception:
            # if welch fails, fill NaNs
            for freq in tob_freqs:
                spl_values[f'SPL_{freq}Hz'] = np.nan
            results.append(spl_values)
            continue

        for center in tob_freqs:
            # third-octave band edges
            f_lower = center / (2**(1/6))
            f_upper = center * (2**(1/6))
            # mask frequencies
            mask = (f >= f_lower) & (f <= f_upper)
            if not np.any(mask):
                spl_values[f'SPL_{center}Hz'] = np.nan
                continue

            # integrate PSD over band to obtain power (Pa^2)
            band_power = np.trapz(psd[mask], f[mask])
            if band_power <= 0 or np.isnan(band_power):
                spl_values[f'SPL_{center}Hz'] = np.nan
                continue

            rms = np.sqrt(band_power)
            spl_db = 20 * np.log10(rms / 1e-6)
            spl_values[f'SPL_{center}Hz'] = spl_db

        results.append(spl_values)

    return pd.DataFrame(results)
