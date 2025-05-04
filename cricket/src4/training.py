import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
from datetime import datetime
import os

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
        
        # Train batting model
        print("Training batting model...")
        bat_model = self.train_model(X_bat_train, y_bat_train, model_type='gradient_boosting', 
                            hyperparameter_tuning=hyperparameter_tuning)
        
        # Train bowling model
        print("Training bowling model...")
        bowl_model = self.train_model(X_bowl_train, y_bowl_train, model_type='gradient_boosting',
                            hyperparameter_tuning=hyperparameter_tuning)
        
        # Evaluate models
        print("\nEvaluating batting model:")
        bat_rmse, bat_r2, bat_pred = self.evaluate_model(bat_model, X_bat_test, y_bat_test)
        
        print("\nEvaluating bowling model:")
        bowl_rmse, bowl_r2, bowl_pred = self.evaluate_model(bowl_model, X_bowl_test, y_bowl_test)
        
        # Get feature importance
        bat_importance = self.get_feature_importance(bat_model, self.bat_features)
        bowl_importance = self.get_feature_importance(bowl_model, self.bowl_features)
        
        if bat_importance is not None:
            print("\nBatting feature importance:")
            print(bat_importance.head(5))
            
        if bowl_importance is not None:
            print("\nBowling feature importance:")
            print(bowl_importance.head(5))
        
        # Save models
        self.bat_model = bat_model
        self.bowl_model = bowl_model
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        joblib.dump(bat_model, os.path.join(self.model_dir, f"bat_model_{timestamp}.pkl"))
        joblib.dump(bowl_model, os.path.join(self.model_dir, f"bowl_model_{timestamp}.pkl"))
        
        print(f"Models saved to {self.model_dir}")
        self.models_trained = True
        
        return bat_model, bowl_model 