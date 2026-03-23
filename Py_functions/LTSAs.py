def make_freq_vector_tob(max_freq, min_freq=10.0):
        """
        Create a target frequency vector with third-octave band (TOB) centers.
        
        Parameters
        ----------
        max_freq : float
            Maximum frequency in Hz
        min_freq : float
            Minimum frequency in Hz (default 10.0)
        
        Returns
        -------
        array
            Third-octave band center frequencies within [min_freq, max_freq]
        """
        # Standard TOB centers (ISO 266)
        tob_centers = np.array([10, 12.5, 16, 20, 25, 31.5, 40, 50, 63, 80, 100, 125, 160, 200, 
                                250, 315, 400, 500, 630, 800, 1000, 1250, 1600, 2000, 2500, 3150, 
                                4000, 5000, 6300, 8000, 10000, 12500, 16000, 20000, 25000, 31500, 40000, 50000])
        
        # Filter to range [min_freq, max_freq]
        tob_filtered = tob_centers[(tob_centers >= min_freq) & (tob_centers <= max_freq)]
        return tob_filtered

# ============================================================================
# LTSA GENERATION
# ============================================================================
def compute_ltsa_tob_jomopans(audio_data, sample_rate, time_bin_duration, max_freq, chunk_duration=300):
    """
    Compute TOB LTSA following JOMOPANS standard (Section 3.3.2 & 3.4)
    Processes audio in chunks to avoid memory issues with large files.
   
    Parameters
    ----------
    audio_data : array
        Audio signal in Pa
    sample_rate : int
        Sample rate in Hz
    time_bin_duration : float
        Duration of each time bin in seconds
    max_freq : float
        Maximum frequency
    chunk_duration : float
        Duration of each processing chunk in seconds (default 300 = 5 min)
    
    Returns
    -------
    tob_centers, times, ltsa_db
    """
    time_bin_samples = int(time_bin_duration * sample_rate)
    n_time_bins = len(audio_data) // time_bin_samples
    
    if n_time_bins == 0:
        return np.array([]), np.array([]), np.zeros((0, 0))
    
    tob_centers = make_freq_vector_tob(max_freq)
    
    # TOB band edges (IEC 1260-1:2014)
    band_edges = np.zeros((len(tob_centers), 2))
    for i, fc in enumerate(tob_centers):
        band_edges[i, 0] = fc * 10**(-1/20)  # f_lower
        band_edges[i, 1] = fc * 10**(1/20)   # f_upper
    
    # Trim audio to exact number of time bins
    audio_trimmed = audio_data[:n_time_bins * time_bin_samples]
    
    # FFT parameters
    nperseg = time_bin_samples
    n_fft = 4 * nperseg  # 4x zero-padding as per JOMOPANS
    
    # Work out chunk size in number of time bins
    bins_per_chunk = max(1, int(chunk_duration / time_bin_duration))
    n_chunks = int(np.ceil(n_time_bins / bins_per_chunk))
    
    #print(f"Computing TOB LTSA (JOMOPANS standard) with {n_time_bins} time bins in {n_chunks} chunks...")
    
    # Pre-allocate full output array
    ltsa = np.zeros((len(tob_centers), n_time_bins))
    
    for chunk_idx in range(n_chunks):
        bin_start = chunk_idx * bins_per_chunk
        bin_end = min(bin_start + bins_per_chunk, n_time_bins)
        n_bins_this_chunk = bin_end - bin_start
        
        # Slice audio for this chunk
        sample_start = bin_start * time_bin_samples
        sample_end = bin_end * time_bin_samples
        audio_chunk = audio_trimmed[sample_start:sample_end]
        
        #print(f"  Chunk {chunk_idx+1}/{n_chunks} ({bin_start}-{bin_end} of {n_time_bins} bins)...")
        
        # Compute spectrogram for this chunk only
        freqs, _, Sxx = signal.spectrogram(
            audio_chunk,
            fs=sample_rate,
            window='hann',
            nperseg=nperseg,
            noverlap=0,
            nfft=n_fft,
            scaling='density',
            mode='psd'
        )
        
        df = freqs[1] - freqs[0] if len(freqs) > 1 else 1.0
        
        # Vectorized band power calculation for all TOB bands
        for b_idx, (f_lower, f_upper) in enumerate(band_edges):
            mask = (freqs >= f_lower) & (freqs <= f_upper)
            
            if np.any(mask):
                band_power = np.sum(Sxx[mask, :], axis=0) * df
            else:
                band_power = np.full(n_bins_this_chunk, 1e-20)
            
            band_power = np.maximum(band_power, 1e-20)
            rms = np.sqrt(band_power)
            ltsa[b_idx, bin_start:bin_end] = 20 * np.log10(rms / 1e-6)
    
    times = np.arange(n_time_bins) * time_bin_duration
    
    return tob_centers, times, ltsa