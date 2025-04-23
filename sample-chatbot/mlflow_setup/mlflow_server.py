import os
import subprocess

def start_mlflow_server():
    backend_store_uri = os.path.abspath("mlflow_setup/mlflow.db")
    artifact_root = os.path.abspath("mlflow_setup/mlruns")

    os.makedirs(artifact_root, exist_ok=True)

    subprocess.run([
        "mlflow", "server",
        "--backend-store-uri", f"sqlite:///{backend_store_uri}",
        "--default-artifact-root", f"file://{artifact_root}",
        "--host", "0.0.0.0",
        "--port", "5001"
    ])

if __name__ == "__main__":
    start_mlflow_server()