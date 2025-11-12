import os
import pandas as pd
import numpy as np
import mne
import traceback
import pywt
from scipy.fft import fft
import torch
import torch.nn.functional as F
from scipy.signal import butter, lfilter, medfilt, iirfilter, savgol_filter



# --------- Dataframes to tensor ---------
def df_to_tensors(df):
    """
    Convert a DataFrame (from Layer 1) into EEG + Label tensors.
    """
    
    # ---- Fix: split EEG channels and labels ----
    label_columns = [col for col in df.columns if col.lower() in ["label", "class", "target"]]
    eeg_channels = [col for col in df.columns if col not in label_columns and col.lower() != "time"]
    
    # ----------------- Clean EEG columns -----------------
    for col in eeg_channels:
        df[col] = pd.to_numeric(df[col], errors='coerce')  # convert strings/non-numeric to NaN
    df[eeg_channels] = df[eeg_channels].fillna(0.0)  # replace NaNs with 0
    
    # ----------------- Clean label columns -----------------
    for col in label_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    if label_columns:
        df[label_columns] = df[label_columns].fillna(0.0)
    
    # ----------------- Convert to torch tensors -----------------
    eeg_tensor = torch.tensor(df[eeg_channels].values, dtype=torch.float32)
    label_tensor = torch.tensor(df[label_columns].values, dtype=torch.float32) if label_columns else None

    return eeg_tensor, label_tensor, eeg_channels, label_columns
 
 
def median_and_bandpass_filter(eeg_tensor, lowcut, highcut, fs, order=5):
    """
    Apply a 3rd-order median filter followed by a Butterworth bandpass filter to an EEG tensor(channels only, no labels).

    Parameters:
        eeg_tensor (torch.Tensor): EEG signal tensor of shape [samples, channels].
        lowcut (float): Low cutoff frequency in Hz for bandpass filter.
        highcut (float): High cutoff frequency in Hz for bandpass filter.
        fs (float): Sampling frequency in Hz.
        order (int): Order of the Butterworth bandpass filter (default: 5).

    Returns:
        torch.Tensor: Filtered EEG tensor with same shape and device as input.

    Notes:
        - Median filter removes spike noise (kernel size = 3).
        - Bandpass filter preserves frequencies between lowcut and highcut Hz.
        - Fully tensor-compatible; no manual conversions required.
    """
    # Step 1: Median filtering
    array = eeg_tensor.detach().cpu().numpy()
    median_filtered = medfilt(array, kernel_size=3)
    median_filtered = np.nan_to_num(median_filtered, nan=0.0)
    median_tensor = torch.tensor(median_filtered, dtype=torch.float32, device=eeg_tensor.device)

    # Step 2: Bandpass filtering
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')

    filtered_np = lfilter(b, a, median_tensor.detach().cpu().numpy(), axis=0)
    bandpassed_tensor = torch.tensor(filtered_np, dtype=torch.float32, device=eeg_tensor.device)

    return bandpassed_tensor

#notch filter apllied at 60hz
def Implement_Notch_Filter(time, band, freq, ripple, order, data, filter_type='butter'):
    """
    Apply a notch (bandstop) filter to remove powerline interference.
    
    Parameters:
        eeg_tensor (torch.Tensor or numpy array): EEG channel data [samples, channels].
        time (float): Sampling period (1 / sampling frequency) in seconds.
        band (float): Bandwidth of the notch filter around the notch frequency.
        freq (float): Notch frequency to remove (e.g., 50 or 60 Hz).
        ripple (float): Filter ripple for Chebyshev (ignored if Butterworth).
        order (int): Filter order.
        data (torch.Tensor or numpy array): Input EEG signal.
        filter_type (str): Type of filter ('butter', 'cheby1', 'cheby2').
    
    Returns:
        torch.Tensor: Notch-filtered EEG tensor.
    
    Notes:
        - Operates only on EEG channel tensor (not labels).
        - Converts tensor → numpy, applies scipy iirfilter and lfilter, converts back.
        - Preserves device (CPU/GPU) of input tensor.
    """
    fs   = 1/time
    nyq  = fs/2.0
    low  = freq - band/2.0
    high = freq + band/2.0
    low  = low/nyq
    high = high/nyq
    b, a = iirfilter(order, [low, high], rp=ripple, btype='bandstop',
                     analog=False, ftype=filter_type)                 
    
    # Convert tensor to numpy if needed
    if isinstance(data, torch.Tensor):
        data_np = data.detach().cpu().numpy()
    else:
        data_np = data
        
    # Apply filter
    filtered_np = lfilter(b, a, data_np, axis=0)   
    
    # Convert back to tensor
    notch_tensor = torch.tensor(filtered_np, dtype=torch.float32, device=data.device if isinstance(data, torch.Tensor) else None)
    return notch_tensor


def wavelet_denoise(eeg_tensor, wavelet='db4', level=4, mode='soft', threshold_method='BayesShrink'):
    """
    Performs wavelet denoising on a multi-channel EEG tensor.

    Args:
        eeg_tensor (torch.Tensor or np.array): EEG tensor of shape [samples, channels].
        wavelet (str): The name of the wavelet to use (e.g., 'db4', 'haar').
        level (int): The number of decomposition levels.
        mode (str): The thresholding mode ('soft' or 'hard').
        threshold_method (str): The method for estimating the threshold ('BayesShrink' or 'VisuShrink').

    Returns:
        torch.Tensor: Denoised EEG tensor of shape [samples, channels].
    """
    # Convert to numpy if it's a torch tensor
    if isinstance(eeg_tensor, torch.Tensor):
        eeg_np = eeg_tensor.detach().cpu().numpy()
    else:
        eeg_np = eeg_tensor

    denoised_np = np.zeros_like(eeg_np, dtype=np.float32)

    # Apply original wavelet denoising channel-wise
    for ch in range(eeg_np.shape[1]):
        signal = eeg_np[:, ch]
        coeffs = pywt.wavedec(signal, wavelet, level=level)
        sigma = np.median(np.abs(coeffs[-1])) / 0.6745  # Estimate noise standard deviation

        thresholds = []
        for i in range(1, len(coeffs)):
            if threshold_method == 'BayesShrink':
                threshold = sigma**2 / np.std(coeffs[i]) if np.std(coeffs[i]) != 0 else 0
            elif threshold_method == 'VisuShrink':
                threshold = sigma * np.sqrt(2 * np.log(len(signal)))
            else:
                raise ValueError("Invalid threshold_method. Choose 'BayesShrink' or 'VisuShrink'.")
            thresholds.append(threshold)

        denoised_coeffs = [coeffs[0]] + [pywt.threshold(c, t, mode=mode) for c, t in zip(coeffs[1:], thresholds)]
        reconstructed = pywt.waverec(denoised_coeffs, wavelet)
        
        # Ensure length matches original signal
        if len(reconstructed) < len(signal):
            reconstructed = np.pad(reconstructed, (0, len(signal)-len(reconstructed)))
        denoised_np[:, ch] = reconstructed[:len(signal)]        
        

    # Convert back to torch tensor
    denoised_tensor = torch.tensor(denoised_np, dtype=torch.float32, device=eeg_tensor.device if isinstance(eeg_tensor, torch.Tensor) else None)
    return denoised_tensor
    
def savgol_smooth_eeg(eeg_tensor, window_length=11, polyorder=3, mode='interp'):
    """
    Apply Savitzky–Golay smoothing to each EEG channel independently.

    Args:
        eeg_tensor (torch.Tensor or np.array): EEG tensor of shape [samples, channels].
        window_length (int): Length of the filter window (must be odd).
        polyorder (int): Polynomial order to fit.
        mode (str): How to handle edges ('mirror', 'constant', 'nearest', 'interp', etc.).

    Returns:
        torch.Tensor: Smoothed EEG tensor of shape [samples, channels].
    """
    # Convert to numpy if it's a torch tensor
    if isinstance(eeg_tensor, torch.Tensor):
        eeg_np = eeg_tensor.detach().cpu().numpy()
        device = eeg_tensor.device
    else:
        eeg_np = eeg_tensor
        device = None

    smoothed_np = np.zeros_like(eeg_np, dtype=np.float32)

    # Apply SG smoothing channel-wise
    for ch in range(eeg_np.shape[1]):
        signal = eeg_np[:, ch]
        try:
            # Adjust window length if signal is shorter
            win_len = min(window_length, len(signal))
            if win_len % 2 == 0:
                win_len -= 1  # must be odd
            if win_len < 3:
                # Too short, skip smoothing
                smoothed_np[:, ch] = signal
            else:
                poly = min(polyorder, win_len - 1)
                smoothed_np[:, ch] = savgol_filter(signal, window_length=win_len, polyorder=poly, mode=mode)
        except Exception as e:
            print(f"Error smoothing channel {ch}: {e}")
            traceback.print_exc()
            smoothed_np[:, ch] = signal  # fallback: keep original signal

    # Convert back to torch tensor
    smoothed_tensor = torch.tensor(smoothed_np, dtype=torch.float32, device=device)
    return smoothed_tensor   

def ica_clean_eeg(eeg_tensor, eeg_channels, sfreq=256.0, n_components=20, method='fastica', random_state=42, highpass=1.0):
    
    """
    Apply ICA preprocessing on a torch EEG tensor and return processed tensor.

    Parameters:
    -----------
    eeg_tensor : torch.Tensor
        EEG tensor of shape [samples, channels].
    eeg_channels : list of str
        List of channel names corresponding to columns in eeg_tensor.
    sfreq : float
        Sampling frequency in Hz.
    n_components : int
        Number of ICA components.
    method : str
        ICA method ('fastica', 'infomax', 'picard').
    random_state : int
        Random seed for reproducibility.
    highpass : float
        Lower frequency bound for high-pass filter (Hz). Use None to skip.    

    Returns:
    --------
    ica_tensor : torch.Tensor
        EEG tensor after ICA artifact removal, shape [samples, channels].
    """
    n_channels = len(eeg_channels)
    if n_channels < 2:
        # No EEG channels, skip ICA
        return eeg_tensor

    # Ensure n_components does not exceed number of channels
    n_components = min(n_components, n_channels - 1)
    
    # Convert tensor to numpy array: [channels, samples]
    eeg_np = eeg_tensor.detach().cpu().numpy().T

    # Create MNE Raw object
    info = mne.create_info(ch_names=eeg_channels, sfreq=sfreq, ch_types='eeg')
    raw = mne.io.RawArray(eeg_np, info)
    
    # High-pass filter
    if highpass is not None:
        raw.filter(l_freq=highpass, h_freq=None)

    try:
        ica = mne.preprocessing.ICA(n_components=n_components,
                                    method=method,
                                    random_state=random_state)
        ica.fit(raw)
        raw_clean = ica.apply(raw.copy())
        ica_tensor = torch.tensor(raw_clean.get_data().T, dtype=torch.float32, device=eeg_tensor.device)
    except Exception as e:
        print("ICA skipped due to error:")
        traceback.print_exc()  # prints full traceback
        ica_tensor = eeg_tensor  # fallback to original tensor

    return ica_tensor
    

def preprocess_eeg(
    eeg_tensor,
    label_tensor=None,
    eeg_channels=None,
    filters="all",  # Accept "all" or a list of filters
    wavelet_params={'wavelet':'db4', 'level':4, 'mode':'soft', 'threshold_method':'BayesShrink'},
    savgol_params={'window_length':11, 'polyorder':3, 'mode':'interp'},
    median_bandpass_params={'lowcut':0.5, 'highcut':40, 'fs':256.0, 'order':5},
    notch_params={'time':1/256.0, 'band':2.0, 'freq':60.0, 'ripple':30, 'order':2, 'filter_type':'butter'},
    ica_params={'sfreq':256.0, 'n_components':20, 'method':'fastica', 'random_state':42}
):
    """
    Main EEG preprocessing function applying selected filters in order.

    Args:
        eeg_tensor (torch.Tensor): Raw EEG tensor [samples, channels].
        label_tensor (torch.Tensor or None): Optional label tensor to reattach.
        filters (str or list): "all" or a list of filters to apply. Options: 
                               "wavelet", "savgol", "median_bandpass", "notch", "ica".
        *_params (dict): Optional parameters for each filter.

    Returns:
        torch.Tensor: Processed EEG tensor with labels reattached if available.
    """
    
    # If user selects "all", expand to all filters
    if filters == "all":
        filters = ["ica", "wavelet", "savgol", "median_bandpass", "notch"]

    processed_eeg = eeg_tensor
    
    # 0. ICA preprocessing
    if "ica" in filters:
        processed_eeg = ica_clean_eeg(processed_eeg, eeg_channels, **ica_params)

    # 1. Wavelet denoising
    if "wavelet" in filters:
        processed_eeg = wavelet_denoise(processed_eeg, **wavelet_params)

    # 2. Savitzky–Golay smoothing
    if "savgol" in filters:
        processed_eeg = savgol_smooth_eeg(processed_eeg, **savgol_params)

    # 3. Median + bandpass filter
    if "median_bandpass" in filters:
        processed_eeg = median_and_bandpass_filter(processed_eeg, **median_bandpass_params)

    # 4. Notch filter
    if "notch" in filters:
        processed_eeg = Implement_Notch_Filter(data=processed_eeg, **notch_params)

    # Reattach labels if they exist
    if label_tensor is not None and label_tensor.shape[1] > 0:
        processed_eeg = torch.cat([processed_eeg, label_tensor], dim=1)

    return processed_eeg


def run_pipeline(df, filters="all", user_id=None):
    """
    Main block for EEG preprocessing.
    User can select filters via the `filters` variable:
        - "all" → apply all filters
        - ["wavelet", "savgol"] → apply only wavelet and Savitzky–Golay
        - ["median_bandpass"] → apply only median + bandpass
        - ["notch"] → apply only notch
    """
    
    # file path 
    #file_path = "/home/tanisha-bhatia/Documents/EEG_Model/dummy_eeg.xlsx"

    # load EEG into tensor 
    # Shape: [samples, channels]
    eeg_tensor, label_tensor, eeg_channels, label_columns = df_to_tensors(df)
    
    
    # Apply selected preprocessing
    processed_tensor = preprocess_eeg(
        eeg_tensor,
        label_tensor=label_tensor,
        eeg_channels=eeg_channels,
        filters=filters
    )
    
    # Directly wrap into DataFrame with existing column names
    all_columns = eeg_channels + label_columns
    processed_df = pd.DataFrame(processed_tensor.detach().cpu().numpy(), columns=all_columns)
    
    # Force label column to be "Label"
    if len(label_columns) == 1:
        processed_df.rename(columns={label_columns[0]: "Label"}, inplace=True)
    
    # Print results
    print("\nProcessed EEG tensor shape:", processed_tensor.shape)
    print("\nFirst few rows of processed EEG with column names:\n")
    print(processed_df.head(10))
    
    return processed_tensor, processed_df
    

    
