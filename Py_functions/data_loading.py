# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def read_wav_file(filepath):
    """
    Read .wav file using soundfile library.
    
    Parameters
    ----------
    filepath : str
        Path to .wav file
    
    Returns
    -------
    audio_data : numpy array
        Audio samples
    sample_rate : int
        Sample rate in Hz
    """
    if not SOUNDFILE_AVAILABLE:
        print(f"ERROR: soundfile not available. Cannot read {filepath}")
        return None, None
    
    try:
        # Read the .wav file
        audio_data, sample_rate = sf.read(filepath)
        
        # If stereo, convert to mono by averaging channels
        if audio_data.ndim > 1:
            audio_data = np.mean(audio_data, axis=1)
        
        return audio_data, sample_rate
    except Exception as e:
        print(f"ERROR reading {filepath}: {str(e)}")
        return None, None

def get_datetime_from_filename(filename):
    """
    Extract datetime and serial number from SoundTrap filename.
    Format: SERIALNUMBER.YYMMDDHHMMSS.wav
    Example: 9611.251011103045.wav
    
    Returns: datetime object (or None if parsing fails)
    """
    basename = os.path.basename(filename)
    try:
        # Split on dots: [serial_number, datetime_string, 'wav']
        parts = basename.split('.')
        if len(parts) >= 3 and parts[2] == 'wav':
            # Second part should be YYMMDDHHMMSS
            datestr = parts[1]
            # Format: YYMMDDHHMMSS (2-digit year)
            dt = datetime.strptime(datestr, "%y%m%d%H%M%S")
            return dt
        else:
            print(f"Warning: Could not parse datetime from {basename} (unexpected format)")
            return None
    except Exception as e:
        print(f"Warning: Could not parse datetime from {basename}: {str(e)}")
        return None


def get_serial_from_filename(filename):
    """
    Extract serial number (first part before dot) from filenames like:
    9611.251017203830.wav or 9611.251017203830.sud
    Returns serial as string or None if not found.
    """
    basename = os.path.basename(filename)
    parts = basename.split('.')
    if len(parts) >= 2:
        serial = parts[0]
        # Basic validation: serial should be digits (4-digit typically)
        if serial.isdigit():
            return serial
    return None

# Determine if file has already been processed
def is_file_processed(base_name, output_dir):
    """
    Check if a file has already been processed by looking for its output file(s).
    """
    expected_files = [
        f"{base_name}_ltsa_TOB_JOMOPANS.tab",
    ]
    for fname in expected_files:
        if not os.path.exists(os.path.join(output_dir, fname)):
            return False
    return True