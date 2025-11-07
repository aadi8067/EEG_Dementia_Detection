import os
import uuid

# Base directory where all user data will be stored
USER_BASE_DIR = os.path.join("users")


def ensure_user_dirs(user_id: str = None):
    """
    Ensures that the base 'users' folder and (optionally) user-specific subfolders exist.
    Returns relative paths instead of absolute.
    """
    # create base dir if not exists
    os.makedirs(USER_BASE_DIR, exist_ok=True)

    if user_id:
        user_dir = os.path.join(USER_BASE_DIR, user_id)
        raw_dir = os.path.join(user_dir, "raw_input_files")
        eeg_dir = os.path.join(user_dir, "eeg_files")

        os.makedirs(user_dir, exist_ok=True)
        os.makedirs(raw_dir, exist_ok=True)
        os.makedirs(eeg_dir, exist_ok=True)

        return {
            "user_dir": user_dir,
            "raw_input_files": raw_dir,
            "eeg_files": eeg_dir,
        }

    return {"base_dir": USER_BASE_DIR}



def save_uploaded_file(file_storage, upload_folder):
    # file_storage is a Werkzeug FileStorage
    filename = file_storage.filename
    safe_name = f"{uuid.uuid4().hex}_{filename}"
    dest = os.path.join(upload_folder, safe_name)
    file_storage.save(dest)
    return dest
