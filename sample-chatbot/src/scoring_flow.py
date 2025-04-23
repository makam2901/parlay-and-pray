from metaflow import FlowSpec, step, Parameter, JSONType
import pandas as pd
import mlflow.sklearn
import mlflow
from mlflow.tracking import MlflowClient

class ScoringFlow(FlowSpec):

    # Accepts a list of feature values passed via CLI
    vector = Parameter(
        "vector",
        type=JSONType,
        help="JSON list of features to score (must match training feature count)",
        default=None
    )

    @step
    def start(self):
        if self.vector is None:
            raise ValueError("No input vector provided. Use --vector '[45,1,3,120,...]'")
        self.next(self.load_model)

    @step
    def load_model(self):
        
        mlflow.set_tracking_uri("http://localhost:5001")  # or the appropriate tracking URI
        mlflow.set_experiment("parlay-and-pray-heart-disease")

        client = MlflowClient()
        model_name = "parlay_model"  # replace with your model's name
        versions = client.get_latest_versions(model_name, stages=[])
        latest_version = max(versions, key=lambda v: int(v.version))

        # Load the model
        self.model = mlflow.sklearn.load_model(f"models:/{model_name}/{latest_version.version}")
        print(f"Loaded model '{model_name}' version {latest_version.version}")
        self.next(self.predict)

    @step
    def predict(self):
        input_df = pd.DataFrame([self.vector])
        preds = self.model.predict(input_df)
        print("Prediction:", preds[0])
        self.prediction = preds[0]
        self.next(self.end)

    @step
    def end(self):
        print("Scoring completed.")