"""
Calibration module for SoundTrap audio analysis.

Provides functions to load calibration data and convert audio samples to calibrated pressure in Pa.
"""

import numpy as np
import pandas as pd


def load_calibration(cal_file):
    """
    Load calibration Excel file into a pandas DataFrame indexed by serial number (as string).
    
    Expected columns:
    - 'Serial' (or first column): Serial number of the hydrophone
    - 'Low_Gain': Sensitivity in dB re 1 V/µPa

    Parameters
    ----------
    cal_file : str
        Path to calibration Excel file

    Returns
    -------
    pd.DataFrame
        Dataframe indexed by serial number with 'Low_Gain' column
        Returns None if file cannot be read
    """
    try:
        df = pd.read_excel(cal_file, engine='openpyxl')
    except Exception as e:
        print(f"Warning: Could not read calibration file {cal_file}: {e}")
        return None

    # Ensure serial number column exists - try to infer column name
    # Prefer a column named 'Serial' or index by first column if necessary
    if 'Serial' in df.columns:
        df['Serial'] = df['Serial'].astype(str)
        df = df.set_index('Serial')
    else:
        # Use first column as serial if it looks numeric
        first_col = df.columns[0]
        df = df.rename(columns={first_col: 'Serial'})
        df['Serial'] = df['Serial'].astype(str)
        df = df.set_index('Serial')

    # Ensure High_Gain exists
    if 'High_Gain' not in df.columns:
        print(f"Warning: 'High_Gain' column not found in {cal_file}. Calibration unavailable.")
        return df

    return df


def apply_calibration(audio_data, serial, cal_df, vpp=2.0):
    """
    Convert audio samples (float -1..1) to pressure in Pa using calibration table.

    Assumptions:
    - audio_data is a numpy array in range [-1, 1] as returned by soundfile
    - Vpp (peak-to-peak voltage) for the hydrophone is 2.0 V (given)
    - 'High_Gain' in calibration table is sensitivity in dB re 1 V/µPa

    Conversion steps:
    1. Convert normalized samples to volts: V = sample * (Vpp / 2)
    2. Convert volts to µPa using sensitivity: sensitivity_linear = 10**(sensitivity_dB/20) [V/µPa]
       pressure_µPa = V / sensitivity_linear
    3. Convert µPa to Pa: pressure_Pa = pressure_µPa * 1e-6

    Parameters
    ----------
    audio_data : array
        Audio samples in range [-1, 1]
    serial : str or int
        Serial number of the hydrophone
    cal_df : pd.DataFrame
        Calibration dataframe indexed by serial (from load_calibration)
    vpp : float
        Peak-to-peak voltage of the recording (default 2.0 V)

    Returns
    -------
    array
        Audio data in Pa, or original audio_data if calibration missing
    """
    if cal_df is None:
        print(f"Warning: calibration dataframe not provided; returning uncalibrated audio for serial {serial}")
        return audio_data

    serial_str = str(serial)
    # Some serials may have leading/trailing spaces or be stored as ints
    if serial_str not in cal_df.index:
        # try integer form
        try:
            if serial_str.isdigit():
                alt = str(int(serial_str))
            else:
                alt = serial_str
            if alt in cal_df.index:
                serial_str = alt
            else:
                # try removing leading zeros
                alt2 = str(int(serial_str)) if serial_str.isdigit() else serial_str
                if alt2 in cal_df.index:
                    serial_str = alt2
                else:
                    print(f"Warning: serial {serial} not found in calibration table; returning uncalibrated audio")
                    return audio_data
        except Exception:
            print(f"Warning: serial {serial} not found in calibration table; returning uncalibrated audio")
            return audio_data

    try:
        sens_db = cal_df.loc[serial_str]['High_Gain']
    except Exception as e:
        print(f"Warning: could not read High_Gain for serial {serial_str}: {e}")
        return audio_data

    # If sensitivity is NaN or missing
    if pd.isna(sens_db):
        print(f"Warning: High_Gain for serial {serial_str} is NaN; returning uncalibrated audio")
        return audio_data

    try:
        # Manufacturer note: High_Gain is dB re 1 µPa. To convert wav samples to µPa multiply by 10^(cal/20)
        cal_lin = 10 ** (float(sens_db) / 20.0)

        # audio_data * cal_lin -> units of µPa
        pressure_uPa = audio_data * cal_lin

        # Convert µPa to Pa
        pressure_pa = pressure_uPa * 1e-6

        return pressure_pa
    except Exception as e:
        print(f"Error applying calibration for serial {serial_str}: {e}")
        return audio_data
