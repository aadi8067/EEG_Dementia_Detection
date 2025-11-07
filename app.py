from flask import Flask, request, jsonify
import os
import uuid
import json
from werkzeug.utils import secure_filename
from datetime import datetime
from validation import validate_eeg_file, validate_metadata_file
from storage import ensure_user_dirs, USER_BASE_DIR
from metadata import extract_metadata, preview_metadata
from processing import process_eeg_to_npy

ensure_user_dirs()
app = Flask(__name__)
BASE_DIR = os.path.abspath(os.path.dirname(__file__))

def make_relative(path):
    return os.path.relpath(path, BASE_DIR)


# --- Metadata schema validation ---
def validate_metadata_schema(metadata: dict):
    """Ensure metadata has subject or subject_id."""
    if isinstance(metadata, list):
        for row in metadata:
            if not ("subject_id" in row or "subject" in row):
                raise ValueError("Each metadata row must contain 'subject_id' or 'subject'")
    elif isinstance(metadata, dict):
        if not ("subject_id" in metadata or "subject" in metadata):
            raise ValueError("Metadata must contain 'subject_id' or 'subject' field")
    else:
        raise ValueError("Metadata must be a dict or list of dicts")
    return True


def get_subject_id(metadata: dict, fallback: str = None) -> str:
    """Pick subject identifier, normalize as subject_id."""
    if "subject_id" in metadata:
        return str(metadata["subject_id"])
    elif "subject" in metadata:
        return str(metadata["subject"])
    elif fallback:
        return fallback
    else:
        raise ValueError("No subject_id or subject found in metadata")

# --- Health check endpoints ---
@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")


@app.route("/health", methods=["GET"])
def health():
    return {"status": "ok", "message": "Flask EEG Backend is healthy"}

@app.route('/eeg_types/<user_id>', methods=['POST'])
def set_eeg_type(user_id):
    user_dir = os.path.join(USER_BASE_DIR, user_id)
    metadata_path = os.path.join(user_dir, 'metadata.json')

    if not os.path.exists(metadata_path):
        return jsonify({"error": "metadata.json not found for this user"}), 404

    data = request.get_json() or {}
    eeg_type = data.get("eeg_type")

    # Allowed EEG file types
    valid_types = ["edf", "csv", "set"]
    if eeg_type not in valid_types:
        return jsonify({
            "error": f"Invalid EEG type. Must be one of {valid_types}"
        }), 400

    # Load existing metadata
    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    # Update metadata with chosen EEG type
    if isinstance(metadata, list):
        for row in metadata:
            row["eeg_type"] = eeg_type
    else:
        metadata["eeg_type"] = eeg_type

    # Save updated metadata
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    return jsonify({
        "message": f"EEG type set to '{eeg_type}' for user {user_id}",
        "metadata": metadata
    }), 200

# --- Upload metadata and create user folder ---
@app.route('/metadata', methods=['POST'])
def upload_metadata():
    # if no file uploaded
    if 'metadata' not in request.files:
        return jsonify({"error": "metadata file is required in form field 'metadata'"}), 400

    # check multiple files
    metadata_files = request.files.getlist("metadata")
    if len(metadata_files) > 1:
        return jsonify({"error": "Only one metadata file can be uploaded at a time"}), 400

    metadata_file = metadata_files[0]

    # 1. validate extension
    try:
        validate_metadata_file(metadata_file.filename)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    # 2. generate or reuse user_id
    user_id = request.form.get('user_id')
    if user_id and os.path.exists(os.path.join(USER_BASE_DIR, user_id)):
        user_dir = os.path.join(USER_BASE_DIR, user_id)
    else:
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        rand = uuid.uuid4().hex[:6]
        user_id = f"user_{ts}_{rand}"
        user_dir = os.path.join(USER_BASE_DIR, user_id)
        ensure_user_dirs(user_id)

    # save uploaded metadata temporarily
    tmp_path = os.path.join(user_dir, metadata_file.filename)
    metadata_file.save(tmp_path)

    # 3. parse into dict or list
    try:
        metadata_dict = extract_metadata(tmp_path, subject_id=user_id)
        metadata = metadata_dict["metadata"]
        preview = metadata_dict["preview"]
    except Exception as e:
        return jsonify({"error": f"Metadata file not readable: {e}"}), 400

    # 4. validate schema
    try:
        validate_metadata_schema(metadata)
    except ValueError as e:
        return jsonify({"error": f"Metadata schema error: {e}"}), 400

    metadata_json_path = os.path.join(user_dir, 'metadata.json')

    # --- Handle single-row or multi-row metadata ---
    if isinstance(metadata, list):
        for row in metadata:
            row["subject_id"] = row.get("subject_id") or user_id
            row.pop("subject", None)
            if 'eeg_files' not in row:
                row['eeg_files'] = []
            if 'eeg_type' not in row:
                row['eeg_type'] = None
    else:
        subject_id = get_subject_id(metadata, fallback=user_id)
        metadata["subject_id"] = subject_id
        metadata.pop("subject", None)
        if 'eeg_files' not in metadata:
            metadata['eeg_files'] = []
        if 'eeg_type' not in metadata:
            metadata['eeg_type'] = None

    with open(metadata_json_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    if os.path.exists(tmp_path) and tmp_path != metadata_json_path:
        os.remove(tmp_path)

    return jsonify({
        'message': 'Metadata uploaded and validated',
        'user_id': user_id,
        'user_dir': make_relative(user_dir),
        'metadata_path': make_relative(metadata_json_path),
        'metadata': metadata   # show full metadata
    }), 200


def _unique_path_for(dirpath: str, filename: str) -> str:
    """Return a non-colliding path in dirpath using filename (adds _1, _2 ... if needed)."""
    base, ext = os.path.splitext(filename)
    candidate = f"{base}{ext}"
    counter = 1
    while os.path.exists(os.path.join(dirpath, candidate)):
        candidate = f"{base}_{counter}{ext}"
        counter += 1
    return os.path.join(dirpath, candidate)


@app.route('/eegs/<user_id>', methods=['POST'])
def upload_eeg(user_id):
    user_dir = os.path.join(USER_BASE_DIR, user_id)
    if not os.path.isdir(user_dir):
        return jsonify({"error": "user_id not found"}), 404

    if 'eeg' not in request.files:
        return jsonify({"error": "eeg file(s) are required in form field 'eeg'"}), 400

    eeg_files = request.files.getlist('eeg')
    if not eeg_files:
        return jsonify({"error": "no eeg files uploaded"}), 400

    # Load metadata.json
    metadata_path = os.path.join(user_dir, 'metadata.json')
    if not os.path.exists(metadata_path):
        return jsonify({"error": "metadata.json not found for this user"}), 404

    with open(metadata_path, 'r') as f:
        user_metadata = json.load(f)

    # Normalize chosen eeg_type (if set)
    chosen_type = None
    if isinstance(user_metadata, dict):
        chosen_type = user_metadata.get('eeg_type')
    elif isinstance(user_metadata, list) and len(user_metadata) > 0:
        chosen_type = user_metadata[0].get('eeg_type')

    if chosen_type:
        if not chosen_type.startswith('.'):
            chosen_type = '.' + chosen_type.lower()
        else:
            chosen_type = chosen_type.lower()

    # Validate all files first
    errors = []
    for filestorage in eeg_files:
        orig_name = filestorage.filename or ""
        safe_name = secure_filename(orig_name)
        if not safe_name:
            errors.append(f"Invalid filename: {orig_name}")
            continue

        ext = os.path.splitext(safe_name)[1].lower()

        # enforce chosen type if set
        if chosen_type and ext != chosen_type:
            errors.append(
                f"File '{orig_name}' has extension '{ext}' but expected '{chosen_type}'"
            )
            continue

        # validate against allowed types
        try:
            validate_eeg_file(safe_name)
        except ValueError as e:
            errors.append(f"{orig_name}: {e}")

    if errors:
        return jsonify({"error": "Validation failed", "details": errors}), 400

    # Passed validation — save and process
    raw_dir = os.path.join(user_dir, 'raw_input_files')
    os.makedirs(raw_dir, exist_ok=True)
    eeg_files_dir = os.path.join(user_dir, 'eeg_files')
    os.makedirs(eeg_files_dir, exist_ok=True)

    results = []
    if isinstance(user_metadata, dict):
        subject_id = get_subject_id(user_metadata, fallback=user_id)
    else:
        subject_id = user_id  # fallback when multiple subjects exist

    for filestorage in eeg_files:
        orig_name = filestorage.filename
        safe_name = secure_filename(orig_name)

        # ensure unique path in raw_input_files
        raw_path = _unique_path_for(raw_dir, safe_name)
        filestorage.save(raw_path)

        try:
            # process → npy
            npy_path, channel_length = process_eeg_to_npy(
                raw_path, eeg_files_dir,
                subject_id,
                keep_filename=True
            )
        except Exception as e:
            return jsonify({"error": f"Processing failed for '{orig_name}': {e}"}), 400

        rel_npy = make_relative(npy_path)

        # update metadata.json
        if isinstance(user_metadata, dict):
            if 'eeg_files' not in user_metadata:
                user_metadata['eeg_files'] = []
            user_metadata['eeg_files'].append(rel_npy)
        else:
            # if multiple rows, attach to first subject (or extend later if needed)
            if 'eeg_files' not in user_metadata[0]:
                user_metadata[0]['eeg_files'] = []
            user_metadata[0]['eeg_files'].append(rel_npy)

        results.append({
            "original_filename": orig_name,
            "raw_path": make_relative(raw_path),
            "npy_path": rel_npy,
            "channel_length": channel_length
        })

    # --- 🔧 normalize & save metadata.json but preserve labels ---
    if isinstance(user_metadata, list):
        for row in user_metadata:
            row["subject_id"] = row.get("subject_id") or user_id
            row.pop("subject", None)
            if 'eeg_files' not in row:
                row['eeg_files'] = []
            if 'eeg_type' not in row:
                row['eeg_type'] = None
    else:
        user_metadata["subject_id"] = subject_id
        user_metadata.pop("subject", None)
        if 'eeg_files' not in user_metadata:
            user_metadata['eeg_files'] = []
        if 'eeg_type' not in user_metadata:
            user_metadata['eeg_type'] = None

    with open(metadata_path, 'w') as f:
        json.dump(user_metadata, f, indent=2)

    return jsonify({
        "message": f"{len(results)} EEG file(s) uploaded and processed",
        "user_id": user_id,
        "metadata_path": make_relative(metadata_path),
        "metadata": user_metadata,   # return full updated metadata
        "results": results
    }), 200

# --- Hardware endpoint ---
@app.route("/hardwares", methods=["POST"])
def hardware_register():
    data = request.get_json() or {}
    hardware_id = data.get("hardware_id") or str(uuid.uuid4())
    hardware_type = data.get("type", "unknown")

    return {
        "status": "success",
        "hardware_id": hardware_id,
        "message": f"Hardware {hardware_id} ({hardware_type}) registered successfully."
    }

from flask import render_template

@app.route('/ui')
def ui_home():
    """Frontend UI for uploading metadata, EEG type, and EEG files"""
    return render_template('index.html')

if __name__ == '__main__':
    print("Starting Flask EEG server...")
    app.run(debug=True, host='127.0.0.1', port=5000)
