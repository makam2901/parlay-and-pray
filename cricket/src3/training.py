import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score

def train_model(X, y, model_type='random_forest', hyperparameter_tuning=False):
    """
    Train a model with optional hyperparameter tuning
    
    Args:
        X: Features DataFrame
        y: Target variable
        model_type: Type of model to train ('random_forest' or 'gradient_boosting')
        hyperparameter_tuning: Whether to perform hyperparameter tuning
        
    Returns:
        Trained model
    """
    # Sanitize data
    X = X.select_dtypes(include=[np.number])
    X = X.replace([np.inf, -np.inf], 0).fillna(0)
    
    # Create pipeline with preprocessing
    steps = [
        ('scaler', StandardScaler())
    ]
    
    if model_type == 'random_forest':
        steps.append(('model', RandomForestRegressor(random_state=42)))
        
        if hyperparameter_tuning:
            param_grid = {
                'model__n_estimators': [50, 100, 200],
                'model__max_depth': [None, 10, 20, 30],
                'model__min_samples_split': [2, 5, 10],
                'model__min_samples_leaf': [1, 2, 4]
            }
    else:  # gradient_boosting
        steps.append(('model', GradientBoostingRegressor(random_state=42)))
        
        if hyperparameter_tuning:
            param_grid = {
                'model__n_estimators': [50, 100, 200],
                'model__max_depth': [3, 5, 7],
                'model__learning_rate': [0.01, 0.1, 0.2],
                'model__subsample': [0.8, 0.9, 1.0]
            }
    
    # Create pipeline
    pipeline = Pipeline(steps)
    
    if hyperparameter_tuning:
        print(f"Performing hyperparameter tuning for {model_type}...")
        grid_search = GridSearchCV(
            pipeline, param_grid, cv=5, n_jobs=-1, verbose=1, scoring='neg_mean_squared_error'
        )
        grid_search.fit(X, y)
        best_model = grid_search.best_estimator_
        print(f"Best parameters: {grid_search.best_params_}")
        return best_model
    else:
        pipeline.fit(X, y)
        return pipeline

def evaluate_model(model, X_test, y_test):
    """Evaluate model performance on test data"""
    # Fill NaN values with 0 for testing
    X_test = X_test.fillna(0)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Model Evaluation:")
    print(f"RMSE: {rmse:.2f}")
    print(f"R²: {r2:.2f}")
    
    return rmse, r2, y_pred

def get_feature_importance(model, feature_names):
    """Get feature importance from trained model"""
    # Extract the model from the pipeline
    if hasattr(model, 'named_steps') and 'model' in model.named_steps:
        model_step = model.named_steps['model']
    else:
        model_step = model
    
    # Check if model has feature_importances_
    if hasattr(model_step, 'feature_importances_'):
        importances = model_step.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        importance_df = pd.DataFrame({
            'Feature': [feature_names[i] for i in indices],
            'Importance': [importances[i] for i in indices]
        })
        
        return importance_df
    else:
        return None