import os
import json
import pandas as pd
import yaml
import xmltodict

ALLOWED_METADATA_EXT = {'.json', '.txt', '.csv', '.xls', '.xlsx', '.yml', '.yaml', '.xml'}

def extract_metadata(filepath: str, subject_id: str = None) -> dict:
    """
    Extract metadata from multiple supported formats and return as dictionary or list of dicts.
    Supports multiple rows (CSV/Excel).
    """
    ext = os.path.splitext(filepath)[1].lower()
    preview = None

    if ext == '.json':
        with open(filepath, 'r') as f:
            data = json.load(f)
        preview = {k: data[k] for k in list(data.keys())[:5]}

    elif ext == '.txt':
        data = _parse_txt(filepath)
        preview = {k: data[k] for k in list(data.keys())[:5]}

    elif ext == '.csv':
        df = pd.read_csv(filepath)
        data = df.to_dict(orient="records")   #  ALL rows
        preview = df.head(5).to_dict(orient="records")

    elif ext in ['.xls', '.xlsx']:
        df = pd.read_excel(filepath)
        data = df.to_dict(orient="records")   #  ALL rows
        preview = df.head(5).to_dict(orient="records")

    elif ext in ['.yml', '.yaml']:
        with open(filepath, 'r') as f:
            data = yaml.safe_load(f)
        preview = {k: data[k] for k in list(data.keys())[:5]}

    elif ext == '.xml':
        data = _parse_xml(filepath)
        preview = {k: data[k] for k in list(data.keys())[:5]}

    else:
        raise ValueError(f'Unsupported metadata format: {ext}')

    # Ensure subject_id
    if isinstance(data, list):  # multiple rows
        for row in data:
            sid = row.get("subject_id") or row.get("subject") or subject_id
            if not sid:
                raise ValueError("Each row must contain a 'subject_id' or 'subject' field")
            row["subject_id"] = sid
    else:  # single row
        sid = data.get("subject_id") or data.get("subject") or subject_id
        if not sid:
            raise ValueError("Metadata must contain a 'subject_id' or 'subject' field")
        data["subject_id"] = sid

    return {
        "metadata": data,
        "preview": preview
    }

def validate_metadata_schema(metadata):
    if isinstance(metadata, list):
        for row in metadata:
            if 'subject_id' not in row:
                raise ValueError("Metadata row missing required field: subject_id")
    elif isinstance(metadata, dict):
        if 'subject_id' not in metadata:
            raise ValueError("Metadata missing required field: subject_id")
    else:
        raise ValueError("Metadata must be a dict or list of dicts after parsing")
    return True

def preview_metadata(metadata, max_items=5):
    if isinstance(metadata, dict):
        keys = list(metadata.keys())[:max_items]
        return {k: metadata[k] for k in keys}
    elif isinstance(metadata, list):
        return metadata[:max_items]
    return str(metadata)[:200]
