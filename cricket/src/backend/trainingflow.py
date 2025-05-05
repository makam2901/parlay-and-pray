import os
print("Using METAFLOW_PROFILE:", os.environ.get("METAFLOW_PROFILE"))
print("Checking config file exists:", os.path.exists(os.path.expanduser("~/.metaflowconfig/config.json")))

from metaflow import FlowSpec, step, Parameter, IncludeFile, current
from training import ModelTrainer
from preprocessing import DataPreprocessor
import pickle

class Dream11TrainingFlow(FlowSpec):
    """
    A Metaflow pipeline for training Dream11 prediction models
    """
    
    hyperparameter_tuning = Parameter('hyperparameter_tuning', default=True, type=bool)
    
    @step
    def start(self):
        print("🚀 Starting Dream11TrainingFlow...")
        # Initialize ModelTrainer
        self.trainer = ModelTrainer()
        self.next(self.load_data)
        
    @step
    def load_data(self):
        print("📦 Loading and preprocessing data...")
        
        self.preprocessing = DataPreprocessor()
        self.preprocessing.load_data()
        self.next(self.train_models)
        
    @step
    def train_models(self):
        print("⚙️ Training models...")
        # Train batting and bowling models
        bat_model, bowl_model = self.trainer.train_models(
            self.preprocessing, 
            hyperparameter_tuning=self.hyperparameter_tuning
        )
        self.bat_model = pickle.dumps(bat_model)
        self.bowl_model = pickle.dumps(bowl_model)
        self.next(self.register_results)
        
    @step
    def register_results(self):
        print("📝 Registering results...")
    
        # Save model and feature information as Metaflow artifacts
        self.bat_model = self.trainer.bat_model
        self.bowl_model = self.trainer.bowl_model
        self.bat_features = self.trainer.bat_features  # ✅ Save as artifact
        self.bowl_features = self.trainer.bowl_features  # ✅ Save as artifact
        self.models_trained = self.trainer.models_trained
    
        print("Training flow complete!")
        self.next(self.end)
        
    @step
    def end(self):
        print("🏁 Dream11TrainingFlow finished.")
        # print(f"Artifacts available: {list(self.artifacts.keys())}")

if __name__ == '__main__':
    Dream11TrainingFlow()