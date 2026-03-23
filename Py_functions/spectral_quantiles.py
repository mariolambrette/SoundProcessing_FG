"""
Spectral quantiles computation module for SoundTrap audio analysis.

Provides functions to compute percentile-based spectral analysis across time windows.
"""

import numpy as np
import pandas as pd
from scipy import signal
from scipy.interpolate import interp1d



def compute_spectral_quantiles(audio_data, sample_rate, fft_size=2048, 
                               frame_duration=10.0, max_freq=50000):
    """
    Compute spectral quantiles across time using Welch PSD computed per frame.
    
    Splits the audio into consecutive frames of `frame_duration` seconds and computes
    a Welch PSD for each frame (nperseg=fft_size). Then converts to dB re 1 µPa^2/Hz
    and computes percentiles across frames for each frequency.

    Parameters
    ----------
    audio_data : array
        Audio signal in Pa
    sample_rate : int
        Sample rate in Hz
    fft_size : int
        FFT size for Welch method (default 2048)
    frame_duration : float
        Duration of each frame in seconds (default 10.0)
    max_freq : float
        Maximum frequency for analysis in Hz (default 50000)
    
    Returns
    -------
    pd.DataFrame
        Dataframe with columns: frequency_hz, p05, p25, p50, p75, p95, mean, std
    """
    frame_samples = int(frame_duration * sample_rate)
    if frame_samples < fft_size:
        frame_samples = fft_size

    n_frames = len(audio_data) // frame_samples

    # target frequency vector using log-spacing
    nyq = sample_rate / 2.0
    max_freq_use = min(nyq, max_freq)
    target_freqs = make_freq_vector(max_freq_use)

    if n_frames == 0:
        # fallback to single-welch over entire file and interpolate
        f, psd = signal.welch(audio_data, sample_rate, nperseg=fft_size, scaling='density')
        psd_interp = np.interp(target_freqs, f, psd, left=0.0, right=0.0)
        Sxx_db = 10 * np.log10(psd_interp + 1e-20) + 120.0
        quantiles = {
            'frequency_hz': target_freqs,
            'p05': Sxx_db,
            'p25': Sxx_db,
            'p50': Sxx_db,
            'p75': Sxx_db,
            'p95': Sxx_db,
            'mean': Sxx_db,
            'std': np.zeros_like(Sxx_db)
        }
        return pd.DataFrame(quantiles)

    psd_list = []
    for i in range(n_frames):
        start = i * frame_samples
        end = start + frame_samples
        frame = audio_data[start:end]
        f, psd = signal.welch(frame, sample_rate, nperseg=fft_size, scaling='density')
        # interpolate onto target_freqs
        psd_interp = np.interp(target_freqs, f, psd, left=0.0, right=0.0)
        psd_list.append(psd_interp)

    psd_stack = np.vstack(psd_list)  # shape (n_frames, n_freqs)
    # convert to dB re 1 µPa^2/Hz
    Sxx_db = 10 * np.log10(psd_stack + 1e-20) + 120.0

    quantiles = {
        'frequency_hz': target_freqs,
        'p05': np.percentile(Sxx_db, 5, axis=0),
        'p25': np.percentile(Sxx_db, 25, axis=0),
        'p50': np.percentile(Sxx_db, 50, axis=0),
        'p75': np.percentile(Sxx_db, 75, axis=0),
        'p95': np.percentile(Sxx_db, 95, axis=0),
        'mean': np.mean(Sxx_db, axis=0),
        'std': np.std(Sxx_db, axis=0)
    }

    return pd.DataFrame(quantiles)
