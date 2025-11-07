import os
import numpy as np
import pandas as pd
import mne


def process_eeg_to_npy(filepath, output_dir, subject_id, keep_filename=True):
   
    ext = os.path.splitext(filepath)[1].lower()

    if ext == ".csv":
        df = pd.read_csv(filepath)
        data = df.values
    elif ext == ".set":
        raw = mne.io.read_raw_eeglab(filepath, preload=True)
        data = raw.get_data().T
    elif ext == ".edf":
        raw = mne.io.read_raw_edf(filepath, preload=True)
        data = raw.get_data().T
    else:
        raise ValueError(f"Unsupported EEG file extension: {ext}. Allowed: .csv, .set, .edf")

    # Decide output filename
    if keep_filename:
        base = os.path.splitext(os.path.basename(filepath))[0]
        filename = f"{base}.npy"
    else:
        filename = f"{subject_id}_eeg.npy"

    out_path = os.path.join(output_dir, filename)
    np.save(out_path, data)
    return out_path, data.shape[1]  # return file path + channel length


