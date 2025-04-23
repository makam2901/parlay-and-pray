import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, ParameterSampler
from sklearn.metrics import classification_report
import mlflow
import mlflow.sklearn
from preprocess import load_and_preprocess
import logging
# Suppress all non-warning logs globally
logging.getLogger().setLevel(logging.WARNING)
# Suppress MLflow-specific loggers
logging.getLogger("mlflow").setLevel(logging.WARNING)
logging.getLogger("mlflow.store.model_registry").setLevel(logging.ERROR)
logging.getLogger("mlflow.tracking.fluent").setLevel(logging.ERROR)


def train():
    # MLflow setup
    mlflow.set_tracking_uri("http://localhost:5001")
    mlflow.set_experiment("parlay-and-pray-heart-disease")

    # Load and preprocess data
    df = load_and_preprocess("data/heart-disease.csv")
    X = df.drop("target", axis=1)
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Define hyperparameter search space
    n_features = X_train.shape[1]
    param_dist = {
        'n_estimators': [50, 100, 150],
        'max_depth': [5, 10, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': list(range(1, n_features + 1))
    }

    param_list = list(ParameterSampler(param_distributions=param_dist, n_iter=20, random_state=42))

    best_score = -1
    best_model = None
    best_params = None
    best_run_id = None

    # Loop through all parameter combinations
    for i, params in enumerate(param_list):
        with mlflow.start_run(run_name=f"rf_run_{i}") as run:
            model = RandomForestClassifier(random_state=42, **params)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            report = classification_report(y_test, y_pred, output_dict=True)
            f1 = report['1']['f1-score']

            # Log parameters and metrics
            mlflow.log_params(params)
            mlflow.log_metrics({
                "f1-score": f1,
                "precision": report["1"]["precision"],
                "recall": report["1"]["recall"]
            })

            print(f"Run {i} | F1-score: {f1:.4f}")

            # Track the best model
            if f1 > best_score:
                best_score = f1
                best_model = model
                best_params = params
                best_run_id = run.info.run_id

    # Register the best model
    if best_model is not None:
        with mlflow.start_run(run_id=best_run_id):
            mlflow.sklearn.log_model(
                sk_model=best_model,
                artifact_path="model",
                registered_model_name="parlay_model"
            )
            print(f"✅ Best model registered with F1-score: {best_score:.4f}")
    else:
        print("⚠️ No model was good enough to register.")

    print("🎯 Training completed.")


if __name__ == "__main__":
    train()