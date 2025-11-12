import os
import io
import time
import pandas as pd
import torch
from flask import Flask, request, jsonify, send_file
from eeg_utils import preprocess_eeg, run_pipeline

app = Flask(__name__)

# -----------------------------------------------------------
# Configuration
# -----------------------------------------------------------
# Available filters (these are the options user can choose from)
AVAILABLE_FILTERS = ["wavelet", "savgol", "median_bandpass", "notch"]

# Folders for saving input/output
UPLOAD_FOLDER = "uploads"      # Incoming files will be placed here (if needed)
PROCESSED_FOLDER = "processed" # All processed EEG outputs will be stored here
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)


# -----------------------------------------------------------
#  API → List all available filters
# -----------------------------------------------------------
@app.route("/filters", methods=["GET"])
def list_filters():
    """
    List all available EEG filters.
    Useful for front-end / Layer-1 to know valid filter names.
    Example:
      GET /filters
    Response:
      { "available_filters": ["wavelet", "savgol", "median_bandpass", "notch"] }
    """
    return {"available_filters": AVAILABLE_FILTERS}


# -----------------------------------------------------------
# 3. Apply Filter API 
# -----------------------------------------------------------
@app.route("/apply_filter", methods=["POST"])
def apply_filter():
    """
    EEG Preprocessing API - DataFrame based
    ------------------------------------------------------
    - Accepts EEG data as CSV, XLSX (Excel), etc.
    - Reads into Pandas DataFrame
    - Converts EEG → Torch tensor → applies preprocessing filters
    - Saves processed DataFrame locally in /processed folder
    - Returns processed DataFrame as downloadable CSV

    Example (Postman):
      POST /apply_filter
      form-data:
        key = "data" → EEG.csv / EEG.xlsx
        key = "filters" → "wavelet,savgol"  OR  "all"
    """
    print("DEBUG: Content-Type:", request.content_type)
    print("DEBUG: request.files keys:", list(request.files.keys()))
    print("DEBUG: request.form keys:", list(request.form.keys()))
    print("DEBUG: request.data length:", len(request.data))
    
    if "data" not in request.files:
      return {"error": "No DataFrame received"}, 400


    file = request.files["data"]
    filters = request.form.get("filters", "all")

    try:
        # ---------------------------------
        # Step 1: Read DataFrame (detect format)
        # ---------------------------------
        filename = file.filename.lower()

        if filename.endswith(".csv"):
            df = pd.read_csv(file)
        elif filename.endswith(".xlsx"):
            df = pd.read_excel(file, engine='openpyxl')
        elif filename.endswith(".xls"):
            df = pd.read_excel(file, engine='xlrd')
        else:
            return {"error": f"Unsupported file format: {filename}"}, 400

        # Split: EEG (numeric only) vs Labels (non-numeric)
        eeg_columns = df.select_dtypes(include=["number"]).columns.tolist()
        label_columns = df.select_dtypes(exclude=["number"]).columns.tolist()

        # EEG → Torch tensor
        eeg_tensor = torch.tensor(df[eeg_columns].to_numpy(), dtype=torch.float32)

        # Labels → keep as DataFrame
        label_df = df[label_columns] if label_columns else pd.DataFrame()

        # ---------------------------------
        # Step 2: Determine filters
        # ---------------------------------
        if filters.lower() == "all":
            filters_to_apply = AVAILABLE_FILTERS
            filter_tag = "all"
        else:
            filters_to_apply = [f.strip() for f in filters.split(",")]
            filter_tag = "_".join(filters_to_apply)

        # ---------------------------------
        # Step 3: Apply preprocessing
        # ---------------------------------
        processed_tensor = preprocess_eeg(
            eeg_tensor,
            label_tensor=None,
            filters=filters_to_apply
        )

        # Back to DataFrame
        processed_df = pd.DataFrame(
            processed_tensor.detach().cpu().numpy(),
            columns=eeg_columns
        )
        if not label_df.empty:
            processed_df = pd.concat([processed_df, label_df.reset_index(drop=True)], axis=1)

        # ---------------------------------
        # Step 4: Save locally
        # ---------------------------------
        timestamp = int(time.time())
        processed_filename = f"processed_{filter_tag}_{timestamp}.csv"
        output_path = os.path.join(PROCESSED_FOLDER, processed_filename)
        processed_df.to_csv(output_path, index=False)

        # ---------------------------------
        # Step 5: Return file
        # ---------------------------------
        buffer = io.BytesIO()
        processed_df.to_csv(buffer, index=False)
        buffer.seek(0)

        return send_file(
            buffer,
            mimetype="text/csv",
            as_attachment=True,
            download_name=processed_filename
        )

    except Exception as e:
        return {"error": str(e)}, 500


# -----------------------------------------------------------
# Run the Flask App
# -----------------------------------------------------------
if __name__ == "__main__":
    print("Starting Layer2 Filter API on port 5001...")
    app.run(debug=True, host='127.0.0.1', port=5001)
