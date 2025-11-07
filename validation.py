import os

ALLOWED_EEG_EXT = {'.csv', '.set', '.edf'}
ALLOWED_METADATA_EXT = {'.json', '.txt', '.csv', '.xls', '.xlsx', '.yml', '.yaml', '.xml'}


def _ext(filename):
    return os.path.splitext(filename)[1].lower()


def validate_eeg_file(filename: str):
    ext = _ext(filename)
    if ext not in ALLOWED_EEG_EXT:
        raise ValueError(f"Unsupported EEG file extension: {ext}. Allowed: {ALLOWED_EEG_EXT}")
    return True


def validate_metadata_file(filename: str):
    ext = _ext(filename)
    if ext not in ALLOWED_METADATA_EXT:
        raise ValueError(f"Unsupported metadata file extension: {ext}. Allowed: {ALLOWED_METADATA_EXT}")
    return True
