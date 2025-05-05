import os
import json
from metaflow import FlowSpec, step, Parameter, current, Flow
from preprocessing import DataPreprocessor
from training import ModelTrainer
from scoring import FantasyScorer
import mlflow

class ScoringFlow(FlowSpec):
    
    home_team = Parameter("home_team", help="Home team", default="India")
    away_team = Parameter("away_team", help="Away team", default="Australia")
    venue = Parameter("venue", help="Match venue", default="Wankhede Stadium")

    @step
    def start(self):
        print("🚀 Starting ScoringFlow...")
        self.next(self.load_model)
    
    @step
    def load_model(self):
        print("📥 Loading models and features from MLflow + Metaflow...")
    
        # Set the MLflow tracking URI
        mlflow.set_tracking_uri("https://mlflow-run-337769530755.us-west2.run.app")
    
        # Initialize trainer
        self.model_trainer = ModelTrainer()
    
        # Load latest model versions from MLflow model registry
        self.model_trainer.bat_model = mlflow.sklearn.load_model("models:/Dream11BattingModel/latest")
        self.model_trainer.bowl_model = mlflow.sklearn.load_model("models:/Dream11BowlingModel/latest")
    
        # Use Metaflow Flow object to fetch latest successful training run
        latest_run = Flow("Dream11TrainingFlow").latest_successful_run

        # Load feature lists from that run
        self.model_trainer.bat_features = latest_run.data.bat_features
        self.model_trainer.bowl_features = latest_run.data.bowl_features
    
        self.next(self.predict)

    @step
    def predict(self):
        print("🔮 Running prediction step...")
        preprocessor = DataPreprocessor()
        preprocessor.load_data()

        self.batting_data = preprocessor.batting
        self.bowling_data = preprocessor.bowling
        self.fielding_data = preprocessor.fielding
        self.players_data = preprocessor.players

        scorer = FantasyScorer(preprocessor, self.model_trainer)
        self.predictions = scorer.predict_fantasy_points(
            self.batting_data, self.bowling_data, self.fielding_data
        )

        print(f"✅ Generated predictions for {len(self.predictions)} players.")
        self.next(self.end)

    @step
    def end(self):
        print("🏁 ScoringFlow complete.")
        print(self.predictions.head())

if __name__ == '__main__':
    ScoringFlow()