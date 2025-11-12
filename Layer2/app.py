from flask import Flask, request, jsonify
from eeg_utils import run_pipeline

app = Flask(__name__)

@app.route("/preprocess", methods=["GET"])
def preprocess_api():
    """
    API just selects filters and calls preprocess_eeg().
    File path, saving, and other details remain in eeg_utils.py.
    Example:
      GET /preprocess?filters=all
      GET /preprocess?filters=wavelet,savgol
    """
    try:
        # read filters from query param
        filters = request.args.get("filters", "all")

        if filters.lower() == "all":
            filters = "all"
        else:
            filters = [f.strip() for f in filters.split(",")]
            
        # run the pipeline defined in eeg_utils.py
        processed_tensor, processed_df = run_pipeline(filters)
        
        # simply return the filters chosen, actual processing stays in eeg_utils.py
        return jsonify({
            "message": "EEG preprocessing successful",
            #"filters": filters,
            #"preview": processed_df.head(10).to_dict(orient="records"),
            "saved_file": "Processed data saved to processed_output.csv"
            
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)

