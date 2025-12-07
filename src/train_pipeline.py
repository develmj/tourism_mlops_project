"""
tourism_mlops_project/src/train_pipeline.py

End-to-end training script for the
"Advanced Machine Learning and MLOps: Tourism Package Prediction" project.

This script is designed to be:

- Runnable locally (python src/train_pipeline.py)
- Runnable inside GitHub Actions (pipeline.yml)
- Reusable from notebooks if needed (main() is guarded)

High-level steps
----------------
1. Load train/test splits from Hugging Face Datasets Hub
2. Separate features and target
3. Build a preprocessing + GradientBoostingClassifier pipeline
4. Run hyperparameter tuning with GridSearchCV
5. Evaluate on test data (ROC-AUC + classification report)
6. Persist:
   - Trained best model  -> models/gb_best_model.joblib
   - Experiment metadata -> models/experiment_log.json
7. Attempt to upload artifacts to Hugging Face Model Hub
   (fails gracefully if permissions/token are missing)
"""

import json
import os
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from datasets import load_dataset
from huggingface_hub import upload_file
from huggingface_hub.utils import HfHubHTTPError
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Repository on Hugging Face containing the split CSV files (train/test)
HF_SPLIT_DATASET_REPO = "mjiyer/tourism-wellness-split"

# Repository on Hugging Face Model Hub where the trained model is registered
HF_MODEL_REPO = "mjiyer/tourism-wellness-gb-model"

# Target column in the dataset
TARGET_COL = "ProdTaken"

# Local directories (relative to the Git repo root)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def load_splits_from_hub() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load the pre-split train and test CSVs from Hugging Face Datasets Hub
    and convert them to pandas DataFrames.

    Returns
    -------
    (train_df, test_df) : tuple of pandas.DataFrame
    """
    print("🔹 Loading train/test splits from Hugging Face dataset:", HF_SPLIT_DATASET_REPO)

    dataset_dict = load_dataset(
        HF_SPLIT_DATASET_REPO,
        data_files={"train": "train.csv", "test": "test.csv"},
    )

    train_df = dataset_dict["train"].to_pandas()
    test_df = dataset_dict["test"].to_pandas()

    print(f"   Train shape: {train_df.shape}")
    print(f"   Test  shape: {test_df.shape}")
    return train_df, test_df


def build_pipeline(categorical_cols: list[str], numeric_cols: list[str]) -> GridSearchCV:
    """
    Build the end-to-end scikit-learn Pipeline and wrap it with GridSearchCV.

    The pipeline:
      - One-hot encodes categorical features
      - Passes numerical features through unscaled
      - Trains a GradientBoostingClassifier

    Parameters
    ----------
    categorical_cols : list of str
        Column names that should be treated as categorical.
    numeric_cols : list of str
        Column names that should be treated as numeric.

    Returns
    -------
    grid_search : sklearn.model_selection.GridSearchCV
        Configured grid-search object ready to be fitted.
    """
    # Preprocessor: different transformations for categorical vs numeric
    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
            ("numeric", "passthrough", numeric_cols),
        ]
    )

    # Base Gradient Boosting model
    gb_model = GradientBoostingClassifier(random_state=42)

    # Hyperparameter search space
    param_grid = {
        "model__n_estimators": [100, 200],
        "model__learning_rate": [0.05, 0.1],
        "model__max_depth": [3, 4],
    }

    # Full pipeline: preprocessing + model
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", gb_model),
        ]
    )

    # Grid search with 5-fold CV using ROC-AUC
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=5,
        scoring="roc_auc",
        n_jobs=-1,
        verbose=2,
    )

    return grid_search, param_grid


def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series) -> tuple[float, dict, str]:
    """
    Evaluate the model on the test data.

    Computes ROC-AUC and a full classification report.

    Returns
    -------
    test_roc_auc : float
    report_dict : dict
    report_text : str
    """
    # Predicted class probabilities (for positive class = 1)
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)

    test_roc_auc = roc_auc_score(y_test, y_proba)
    report_text = classification_report(y_test, y_pred)
    report_dict = classification_report(y_test, y_pred, output_dict=True)

    print("🔹 Test ROC-AUC:", test_roc_auc)
    print("🔹 Classification report:\n", report_text)

    return test_roc_auc, report_dict, report_text


def save_artifacts(
    best_model,
    param_grid: dict,
    best_params: dict,
    best_cv_score: float,
    test_roc_auc: float,
    report_dict: dict,
    categorical_cols: list[str],
    numeric_cols: list[str],
) -> tuple[Path, Path]:
    """
    Persist the trained model and experiment metadata to disk.

    Returns
    -------
    (model_path, log_path) : tuple of Path
        Paths to the saved model and JSON log.
    """
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

    # 1. Save model
    model_path = MODELS_DIR / "gb_best_model.joblib"
    joblib.dump(best_model, model_path)
    print(f"✅ Saved best model to: {model_path}")

    # 2. Save experiment log
    experiment_log = {
        "timestamp": timestamp,
        "algorithm": "GradientBoostingClassifier",
        "param_grid": param_grid,
        "best_params": best_params,
        "best_cv_roc_auc": best_cv_score,
        "test_roc_auc": test_roc_auc,
        "classification_report": report_dict,
        "feature_columns": {
            "categorical": categorical_cols,
            "numeric": numeric_cols,
        },
        "target_column": TARGET_COL,
        "hf_datasets": {
            "split_dataset_repo": HF_SPLIT_DATASET_REPO,
        },
        "hf_model_repo": HF_MODEL_REPO,
    }

    log_path = MODELS_DIR / "experiment_log.json"
    with log_path.open("w") as f:
        json.dump(experiment_log, f, indent=2)

    print(f"✅ Saved experiment log to: {log_path}")

    return model_path, log_path


def upload_to_hf_model_hub(model_path: Path, log_path: Path) -> None:
    """
    Attempt to upload the trained model and experiment log to the
    Hugging Face Model Hub.

    This function is intentionally defensive:
    - If the token or permissions are missing, it will print a warning
      but WILL NOT raise an exception. This ensures CI/CD or local runs
      do not fail solely due to registry permissions.
    """
    print(f"🔹 Attempting to upload artifacts to HF Model Hub: {HF_MODEL_REPO}")

    try:
        # Upload model file
        upload_file(
            path_or_fileobj=str(model_path),
            path_in_repo="gb_best_model.joblib",
            repo_id=HF_MODEL_REPO,
            repo_type="model",
        )

        # Upload experiment log
        upload_file(
            path_or_fileobj=str(log_path),
            path_in_repo="experiment_log.json",
            repo_id=HF_MODEL_REPO,
            repo_type="model",
        )

        print("✅ Successfully uploaded model and log to Hugging Face Model Hub.")

    except HfHubHTTPError as e:
        # Non-fatal: training succeeded, only registry step failed
        print("⚠️ Could not upload artifacts to Hugging Face Model Hub.")
        print("   Error message:")
        print(f"   {e}")
        print("   Training artifacts are still available locally in the 'models/' folder.")


# ---------------------------------------------------------------------------
# Main training entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """
    Run the full training, evaluation and (best-effort) registration pipeline.
    This is the function invoked by GitHub Actions and local runs.
    """
    print("====================================================")
    print("  Tourism Wellness Package Prediction - Train Script")
    print("====================================================")

    # 1. Load pre-split data from Hugging Face
    train_df, test_df = load_splits_from_hub()

    # 2. Separate features and target
    y_train = train_df[TARGET_COL]
    X_train = train_df.drop(columns=[TARGET_COL])

    y_test = test_df[TARGET_COL]
    X_test = test_df.drop(columns=[TARGET_COL])

    # 3. Infer feature types
    categorical_cols = X_train.select_dtypes(include=["object"]).columns.tolist()
    numeric_cols = [c for c in X_train.columns if c not in categorical_cols]

    print("🔹 Categorical columns:", categorical_cols)
    print("🔹 Numeric columns:", numeric_cols)

    # 4. Build pipeline + grid search
    grid_search, param_grid = build_pipeline(categorical_cols, numeric_cols)

    # 5. Fit grid search on training data
    print("🔹 Starting grid search over hyperparameters...")
    grid_search.fit(X_train, y_train)
    print("✅ Grid search completed.")

    # Best estimator and metrics
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    best_cv_score = grid_search.best_score_

    print("🔹 Best hyperparameters:", best_params)
    print("🔹 Best CV ROC-AUC:", best_cv_score)

    # 6. Evaluate on test set
    test_roc_auc, report_dict, report_text = evaluate_model(
        best_model, X_test, y_test
    )

    # 7. Save artifacts (model + log)
    model_path, log_path = save_artifacts(
        best_model=best_model,
        param_grid=param_grid,
        best_params=best_params,
        best_cv_score=best_cv_score,
        test_roc_auc=test_roc_auc,
        report_dict=report_dict,
        categorical_cols=categorical_cols,
        numeric_cols=numeric_cols,
    )

    # 8. Best-effort upload to HF Model Hub
    upload_to_hf_model_hub(model_path, log_path)

    print("✅ Training pipeline completed successfully.")
    print(f"   Local model path: {model_path}")
    print(f"   Local log path:   {log_path}")


if __name__ == "__main__":
    main()

