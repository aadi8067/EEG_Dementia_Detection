import os
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    r2_score, mean_squared_error, mean_absolute_error
)

# Function to get user-specific checkpoint directory
def get_user_checkpoint_dir(user_id):
    user_dir = os.path.join(str(user_id), "checkpoints")
    if not os.path.exists(user_dir):
        os.makedirs(user_dir)
    return user_dir

# ----------------- UTILITIES -----------------
def compute_test_size(train_percent, test_percent):
    if train_percent is None and test_percent is None:
        return 0.20
    if train_percent is not None and test_percent is not None:
        if abs((float(train_percent) + float(test_percent)) - 100) > 1e-6:
            raise ValueError("train_percent + test_percent must equal 100.")
        return float(test_percent) / 100.0
    if test_percent is not None:
        return float(test_percent) / 100.0
    return 1.0 - float(train_percent) / 100.0

def prep_features(df, target_col):
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found.")
    X = df.drop(columns=[target_col])
    y = df[target_col]
    X = pd.get_dummies(X, drop_first=True)
    valid = X.notna().all(axis=1) & y.notna()
    return X.loc[valid], y.loc[valid]

# ----------------- METRICS -----------------
def classification_metrics(y_test, y_pred):
    return {
        "accuracy_pct": round(accuracy_score(y_test, y_pred) * 100, 2),
        "precision_pct": round(precision_score(y_test, y_pred, average="weighted", zero_division=0) * 100, 2),
        "recall_pct": round(recall_score(y_test, y_pred, average="weighted", zero_division=0) * 100, 2),
        "f1_pct": round(f1_score(y_test, y_pred, average="weighted", zero_division=0) * 100, 2),
    }

def regression_metrics(y_test, y_pred):
    return {
        "r2": round(r2_score(y_test, y_pred), 4),
        "rmse": round(np.sqrt(mean_squared_error(y_test, y_pred)), 4),
        "mae": round(mean_absolute_error(y_test, y_pred), 4),
    }

# ----------------- MODEL FUNCTIONS -----------------
def run_logistic(X, y, test_size, random_state, user_id):
    le = LabelEncoder()
    y_enc = le.fit_transform(y.astype(str))
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=test_size, random_state=random_state, stratify=y_enc
    )
    model = LogisticRegression(max_iter=2000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    checkpoint_dir = get_user_checkpoint_dir(user_id)
    checkpoint_path = os.path.join(checkpoint_dir, "logistic_model.pkl")
    joblib.dump({"model": model, "classes": le.classes_}, checkpoint_path)

    total = len(X_train) + len(X_test)
    train_pct = round(len(X_train) / total * 100, 2) if total else 0
    test_pct = round(len(X_test) / total * 100, 2) if total else 0
    return {
        "task": "classification",
        "model": "LogisticRegression",
        "metrics": classification_metrics(y_test, y_pred),
        "target_classes": list(le.classes_),
        "split": {
            "train_count": len(X_train),
            "test_count": len(X_test),
            "train_pct": train_pct,
            "test_pct": test_pct
        },
        "checkpoint": checkpoint_path
    }

def run_linear(X, y, test_size, random_state, user_id):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    checkpoint_dir = get_user_checkpoint_dir(user_id)
    checkpoint_path = os.path.join(checkpoint_dir, "linear_model.pkl")
    joblib.dump({"model": model}, checkpoint_path)

    total = len(X_train) + len(X_test)
    train_pct = round(len(X_train) / total * 100, 2) if total else 0
    test_pct = round(len(X_test) / total * 100, 2) if total else 0
    return {
        "task": "regression",
        "model": "LinearRegression",
        "metrics": regression_metrics(y_test, y_pred),
        "split": {
            "train_count": len(X_train),
            "test_count": len(X_test),
            "train_pct": train_pct,
            "test_pct": test_pct
        },
        "checkpoint": checkpoint_path
    }

def run_rf_classifier(X, y, test_size, random_state, user_id):
    le = LabelEncoder()
    y_enc = le.fit_transform(y.astype(str))
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=test_size, random_state=random_state, stratify=y_enc
    )
    model = RandomForestClassifier(random_state=random_state)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    checkpoint_dir = get_user_checkpoint_dir(user_id)
    checkpoint_path = os.path.join(checkpoint_dir, "rf_classifier_model.pkl")
    joblib.dump({"model": model, "classes": le.classes_}, checkpoint_path)

    total = len(X_train) + len(X_test)
    train_pct = round(len(X_train) / total * 100, 2) if total else 0
    test_pct = round(len(X_test) / total * 100, 2) if total else 0
    return {
        "task": "classification",
        "model": "RandomForestClassifier",
        "metrics": classification_metrics(y_test, y_pred),
        "target_classes": list(le.classes_),
        "split": {
            "train_count": len(X_train),
            "test_count": len(X_test),
            "train_pct": train_pct,
            "test_pct": test_pct
        },
        "checkpoint": checkpoint_path
    }

def run_rf_regressor(X, y, test_size, random_state, user_id):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    model = RandomForestRegressor(random_state=random_state)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    checkpoint_dir = get_user_checkpoint_dir(user_id)
    checkpoint_path = os.path.join(checkpoint_dir, "rf_regressor_model.pkl")
    joblib.dump({"model": model}, checkpoint_path)

    total = len(X_train) + len(X_test)
    train_pct = round(len(X_train) / total * 100, 2) if total else 0
    test_pct = round(len(X_test) / total * 100, 2) if total else 0
    return {
        "task": "regression",
        "model": "RandomForestRegressor",
        "metrics": regression_metrics(y_test, y_pred),
        "split": {
            "train_count": len(X_train),
            "test_count": len(X_test),
            "train_pct": train_pct,
            "test_pct": test_pct
        },
        "checkpoint": checkpoint_path
    }

def run_dt_regressor(X, y, test_size, random_state, user_id):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    model = DecisionTreeRegressor(random_state=random_state)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    checkpoint_dir = get_user_checkpoint_dir(user_id)
    checkpoint_path = os.path.join(checkpoint_dir, "dt_regressor_model.pkl")
    joblib.dump({"model": model}, checkpoint_path)

    total = len(X_train) + len(X_test)
    train_pct = round(len(X_train) / total * 100, 2) if total else 0
    test_pct = round(len(X_test) / total * 100, 2) if total else 0
    return {
        "task": "regression",
        "model": "DecisionTreeRegressor",
        "metrics": regression_metrics(y_test, y_pred),
        "split": {
            "train_count": len(X_train),
            "test_count": len(X_test),
            "train_pct": train_pct,
            "test_pct": test_pct
        },
        "checkpoint": checkpoint_path
    }

def run_knn(X, y, test_size, random_state, user_id):
    le = LabelEncoder()
    y_enc = le.fit_transform(y.astype(str))
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=test_size, random_state=random_state, stratify=y_enc
    )
    model = KNeighborsClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    checkpoint_dir = get_user_checkpoint_dir(user_id)
    checkpoint_path = os.path.join(checkpoint_dir, "knn_model.pkl")
    joblib.dump({"model": model, "classes": le.classes_}, checkpoint_path)

    total = len(X_train) + len(X_test)
    train_pct = round(len(X_train) / total * 100, 2) if total else 0
    test_pct = round(len(X_test) / total * 100, 2) if total else 0
    return {
        "task": "classification",
        "model": "KNeighborsClassifier",
        "metrics": classification_metrics(y_test, y_pred),
        "target_classes": list(le.classes_),
        "split": {
            "train_count": len(X_train),
            "test_count": len(X_test),
            "train_pct": train_pct,
            "test_pct": test_pct
        },
        "checkpoint": checkpoint_path
    }

def run_svm(X, y, test_size, random_state, user_id):
    le = LabelEncoder()
    y_enc = le.fit_transform(y.astype(str))
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=test_size, random_state=random_state, stratify=y_enc
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    model = SVC()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    checkpoint_dir = get_user_checkpoint_dir(user_id)
    checkpoint_path = os.path.join(checkpoint_dir, "svm_model.pkl")
    joblib.dump({"model": model, "classes": le.classes_}, checkpoint_path)

    total = len(X_train) + len(X_test)
    train_pct = round(len(X_train) / total * 100, 2) if total else 0
    test_pct = round(len(X_test) / total * 100, 2) if total else 0
    return {
        "task": "classification",
        "model": "SVM",
        "metrics": classification_metrics(y_test, y_pred),
        "target_classes": list(le.classes_),
        "split": {
            "train_count": len(X_train),
            "test_count": len(X_test),
            "train_pct": train_pct,
            "test_pct": test_pct
        },
        "checkpoint": checkpoint_path
    }
