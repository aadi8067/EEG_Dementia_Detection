from flask import Flask, request, jsonify
import os
import uuid
import pandas as pd
import mne   
import io

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

USER_DATA = {}

# process EEG files
def process_eeg_file(filepath):
    ext = os.path.splitext(filepath)[1].lower()

    if ext == ".set":
        raw = mne.io.read_raw_eeglab(filepath, preload=True)
        data = raw.get_data().T 
        df = pd.DataFrame(data, columns=raw.ch_names)
        return df

    elif ext == ".edf":
        raw = mne.io.read_raw_edf(filepath, preload=True)
        data = raw.get_data().T
        df = pd.DataFrame(data, columns=raw.ch_names)
        return df

    elif ext == ".csv":
        df = pd.read_csv(filepath)
        return df

    else:
        raise ValueError("Unsupported file format. Allowed: .csv, .set, .edf")

# Health check endpoints
@app.route("/", methods=["GET"])
def home():
    return {"status": "ok", "message": "Flask EEG Backend is running"}

@app.route("/health", methods=["GET"])
def health():
    return {"status": "ok", "message": "Flask EEG Backend is healthy"}

# Upload multiple EEG files + metadata 
@app.route("/uploads", methods=["POST"])
def upload_files():

    # Assign unique user ID
    user_id = str(uuid.uuid4())

    if "files" not in request.files:
        return jsonify({"error": "No files uploaded"}), 400

    files = request.files.getlist("files")
    if not files:
        return jsonify({"error": "Empty file list"}), 400

    eeg_files = []
    metadata_df = None

    for file in files:
        filename = file.filename.lower()
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)   

        try:
            df = process_eeg_file(filepath)  
        except Exception as e:
            return jsonify({"error": str(e)}), 400

        cols = [c.lower() for c in df.columns]
        if any(c in cols for c in ["subject", "gender", "age", "group", "mmse", "trial_type", "task"]):
            metadata_df = df
        else:
            eeg_files.append(df)

    if metadata_df is None or len(eeg_files) == 0:
        return jsonify({"error": "Need at least one EEG CSV and one metadata CSV"}), 400

    final_records = []

    for idx in range(len(metadata_df)):
        meta_row = metadata_df.iloc[idx].to_dict()

        eeg_arrays = []
        for eeg_df in eeg_files:
            if idx < len(eeg_df):  
                row_data = eeg_df.iloc[idx].values.tolist()
                eeg_arrays.extend(row_data)

        record = {
            "Subject id": meta_row.get("Subject id", idx),
            "eeg_array": eeg_arrays,
            "label": meta_row.get("label", None),
            "msme": meta_row.get("msme", None),
            "age": meta_row.get("age", None),
            "gender": meta_row.get("gender", None),
            "channel_n": len(eeg_files),     
            "channel_length": len(eeg_arrays)
        }
        final_records.append(record)

    final_df = pd.DataFrame(final_records)

    USER_DATA[user_id] = final_df

    return jsonify({
        "message": "Files processed successfully",
        "user_id": user_id,
        "rows": len(final_df),
        "columns": list(final_df.columns)
    }), 200


@app.route("/dataframes/<user_id>", methods=["GET"])
def get_dataframe(user_id):
    if user_id not in USER_DATA:
        return jsonify({"error": "User not found"}), 404

    df = USER_DATA[user_id]
    
    return df.to_html(), 200

# Hardware registration
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

# Run
if __name__ == "__main__":
    print("Starting Flask EEG server...")
    app.run(debug=True, host="127.0.0.1", port=5001)

