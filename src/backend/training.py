import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from datetime import datetime
import os
import mlflow
import mlflow.sklearn

class ModelTrainer:
    """Handles model training and evaluation for Dream11 predictions"""
    
    def __init__(self, model_dir="models"):
        """Initialize the model trainer"""
        self.model_dir = model_dir
        self.bat_model = None
        self.bowl_model = None
        self.bat_features = None
        self.bowl_features = None
        self.models_trained = False
        
        # Create model directory if it doesn't exist
        os.makedirs(model_dir, exist_ok=True)
        
    def train_model(self, X, y, model_type='gradient_boosting', hyperparameter_tuning=False):
        """Train a model with optional hyperparameter tuning"""
        # Double-check for any remaining inf values
        X_clean = X.copy()
        X_clean.replace([np.inf, -np.inf], 0, inplace=True)
        
        if model_type == 'gradient_boosting':
            if hyperparameter_tuning:
                model = GradientBoostingRegressor(
                    n_estimators=200,
                    learning_rate=0.1,
                    max_depth=5,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    random_state=42
                )
            else:
                model = GradientBoostingRegressor()
        else:
            model = GradientBoostingRegressor()
        
        # Try fitting with appropriate dtype to avoid overflow
        try:
            model.fit(X_clean, y)
        except Exception as e:
            print(f"Error during model fitting: {e}")
            print("Attempting to fit with float64 dtype...")
            X_clean = X_clean.astype(np.float64)
            model.fit(X_clean, y)
        
        return model
        
    def evaluate_model(self, model, X_test, y_test):
        """Evaluate model performance"""
        # Clean test data to ensure no inf values
        X_test_clean = X_test.copy()
        X_test_clean.replace([np.inf, -np.inf], 0, inplace=True)
        
        # Predict on test set
        y_pred = model.predict(X_test_clean)
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        print(f"RMSE: {rmse:.4f}")
        print(f"R² Score: {r2:.4f}")
        
        return rmse, r2, y_pred
        
    def get_feature_importance(self, model, feature_names):
        """Extract feature importance from trained model"""
        if hasattr(model, 'feature_importances_'):
            importance = pd.DataFrame({
                'feature': feature_names,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            return importance
        return None
        
    def train_models(self, preprocessor, hyperparameter_tuning=True):
        """Train prediction models with enhanced features and proper NaN handling"""
        if not preprocessor.data_loaded:
            preprocessor.load_data()
        
        print("Preprocessing batting data...")
        X_bat_train, y_bat_train, X_bat_test, y_bat_test = preprocessor.preprocess_batting(preprocessor.batting)
        
        print("Preprocessing bowling data...")
        X_bowl_train, y_bowl_train, X_bowl_test, y_bowl_test = preprocessor.preprocess_bowling(preprocessor.bowling)
        
        # Verify no NaN values before training
        print(f"Checking for NaN values in batting features: {X_bat_train.isna().sum().sum()}")
        print(f"Checking for NaN values in bowling features: {X_bowl_train.isna().sum().sum()}")
        
        # Check for infinity values
        print(f"Checking for infinity values in batting features: {np.isinf(X_bat_train).sum().sum()}")
        print(f"Checking for infinity values in bowling features: {np.isinf(X_bowl_train).sum().sum()}")
        
        # Save feature lists for prediction
        self.bat_features = X_bat_train.columns.tolist()
        self.bowl_features = X_bowl_train.columns.tolist()
        
        # Set MLFlow experiment
        mlflow.set_tracking_uri("https://mlflow-run-337769530755.us-west2.run.app")
        mlflow.set_experiment("Dream11")
        best_bat_rmse, best_bowl_rmse = float("inf"), float("inf")
        best_bat_model, best_bowl_model = None, None
        
        param_grid = {
        "n_estimators": [100, 120, 140, 160, 180, 200],
        "learning_rate": [0.03, 0.05, 0.07, 0.09],
        "max_depth": [3, 4, 5, 6],
        }
        
        for i in range(5):
            timestamp = datetime.now().strftime("%Y_%m_%d_%H%M")
            with mlflow.start_run(run_name=f"iteration_{i+1}_{timestamp}", nested=True):
                print(f"\n🔁 Hyperparameter tuning iteration {i+1}...")
                
                bat_search = RandomizedSearchCV(
                    estimator=GradientBoostingRegressor(random_state=42),
                    param_distributions=param_grid,
                    n_iter=1,
                    scoring='neg_root_mean_squared_error',
                    cv=3,
                    random_state=i,
                    n_jobs=-1
                )
                
                bowl_search = RandomizedSearchCV(
                    estimator=GradientBoostingRegressor(random_state=42),
                    param_distributions=param_grid,
                    n_iter=1,
                    scoring='neg_root_mean_squared_error',
                    cv=3,
                    random_state=i + 10,
                    n_jobs=-1
                )
                
                print("Fitting batting model...")
                bat_search.fit(X_bat_train, y_bat_train)
                bat_model = bat_search.best_estimator_

                print("Fitting bowling model...")
                bowl_search.fit(X_bowl_train, y_bowl_train)
                bowl_model = bowl_search.best_estimator_
                
                bat_rmse, _, _ = self.evaluate_model(bat_model, X_bat_test, y_bat_test)                
                mlflow.log_params({f"bat_{k}": v for k, v in bowl_model.get_params().items()})
                mlflow.log_metric("batting_rmse", bat_rmse)
                mlflow.sklearn.log_model(bat_model, artifact_path="bat_model")
                
                bowl_rmse, _, _ = self.evaluate_model(bowl_model, X_bowl_test, y_bowl_test)
                mlflow.log_params({f"bowl_{k}": v for k, v in bowl_model.get_params().items()})
                mlflow.log_metric("bowling_rmse", bowl_rmse)
                mlflow.sklearn.log_model(bowl_model, artifact_path="bowl_model")

                if bat_rmse < best_bat_rmse:
                    best_bat_rmse = bat_rmse
                    best_bat_model = bat_model
                    best_bat_run_id = mlflow.active_run().info.run_id

                if bowl_rmse < best_bowl_rmse:
                    best_bowl_rmse = bowl_rmse
                    best_bowl_model = bowl_model
                    best_bowl_run_id = mlflow.active_run().info.run_id
                
        print(f"\n Registering best batting model with RMSE: {best_bat_rmse:.4f}")
        mlflow.register_model(f"runs:/{best_bat_run_id}/bat_model", "Dream11BattingModel")

        print(f" Registering best bowling model with RMSE: {best_bowl_rmse:.4f}")
        mlflow.register_model(f"runs:/{best_bowl_run_id}/bowl_model", "Dream11BowlingModel")
        
        self.bat_model = best_bat_model
        self.bowl_model = best_bowl_model
        self.models_trained = True

        return self.bat_model, self.bowl_model
