# Parlay-and-Pray: AI-Powered Dream11 Fantasy Team Predictor

## Overview

**parlay-and-pray** is an AI-powered assistant designed to help users create optimal fantasy cricket teams for the Dream11 platform, focusing initially on the Indian Premier League (IPL). It leverages machine learning models trained on historical match data to provide predictive insights and data-driven recommendations. The application features a user-friendly web interface where users can input match details and receive team suggestions, along with a chat interface powered by a Large Language Model (LLM) for answering general cricket-related queries.

### What is Fantasy Cricket (Dream11)?

Fantasy cricket platforms like Dream11 allow users to create virtual teams composed of real-life players scheduled to play in an upcoming match. Users earn points based on the actual performance of their selected players in that match (e.g., runs scored, wickets taken, catches).

Key aspects of creating a Dream11 team include:
* **Selecting 11 Players:** Choosing a combination of player roles (Wicketkeepers (WK), Batsmen (BAT), All-Rounders (AR), Bowlers (BWL)) within specific count constraints.
* **Credit Limit:** Each player has a credit value, and the total value of the 11 selected players cannot exceed a predefined credit limit (e.g., 100 credits).
* **Team Constraints:** Limits on the maximum number of players that can be selected from a single real-life team (e.g., max 7 players per team).
* **Captain (C) & Vice-Captain (VC):** Users designate one player as Captain (earns 2x points) and one as Vice-Captain (earns 1.5x points).

This project aims to simplify this selection process by predicting player performance and suggesting a team that maximizes potential points while adhering to all constraints.

## Team Members

* Manikesh Makam - [makam2901](https://github.com/makam2901)
* Vani Singh - [vanisingh123](https://github.com/vanisingh123)
* Dhavni Patel - [dhvani32](https://github.com/dhvani32)
* Esha Yamani - [eshayamani](https://github.com/eshayamani)
* Kavin Indrajit - [kavinindirajith](https://github.com/kavinindirajith)

## Flow of the Application

The application consists of a frontend user interface and a backend prediction/chat engine:

1.  **User Interaction (Frontend - Streamlit):**
    * The user accesses the web application built with Streamlit.
    * **Prediction:**
        * Selects Home Team, Away Team, Venue.
        * Specifies desired player role counts (WK, BAT, AR, BWL).
        * Sets optional constraints (max credits, max players per team).
        * Clicks "Predict Team".
    * **Chat:**
        * Types cricket-related questions into the chat input.

2.  **Initialization & Training (Backend - FastAPI Startup):**
    * Upon starting, the backend initializes the `DataPreprocessor`, `ModelTrainer`, and `FantasyScorer`.
    * It loads historical data via the `DataPreprocessor`.
    * **Crucially, it triggers the model training process** (`model_trainer.train_models()`) as described in the "Model Training & MLflow Integration" section below. This ensures the prediction models are ready when the first request arrives.
    * It retrieves the Google Gemini API Key from the environment variable `GEMINI_API_KEY` required for the chat functionality.

3.  **Processing (Backend - FastAPI Runtime):**
    * **Prediction Request (`/predict_team/`):**
        * Receives match details and constraints.
        * Uses the pre-loaded data and **already trained models** from the startup phase.
        * The `FantasyScorer` predicts fantasy points using the trained models.
        * Applies contextual factors (form, venue, opponent).
        * Determines player roles.
        * Constructs the optimal team based on constraints using a greedy approach.
        * Selects Captain and Vice-Captain.
        * Returns the final team and summary to the frontend.
    * **Chat Request (`/chat/`):**
        * Receives user message and history.
        * Passes the request to the LLM handler (`llm.py`).
        * The handler checks for prediction keywords (redirects if found).
        * Otherwise, it calls the Google Gemini API (details in "LLM Integration" section).
        * Returns the chat response and updated history.

4.  **Display Results (Frontend - Streamlit):**
    * **Prediction:**
        * Displays the predicted team, player cards, C/VC, statistics, and visualizations.
    * **Chat:**
        * Displays the assistant's reply.

## Model Training & MLflow Integration

The core predictive capability relies on machine learning models trained to estimate player fantasy points. The training process is integrated into the backend application's startup sequence and utilizes MLflow for experiment tracking and model management:

1.  **Data Preparation:** The `DataPreprocessor` loads and cleans historical batting, bowling, and fielding data. It performs feature engineering (e.g., strike rate, economy rate, recent form rolling averages) and splits the data into training and testing sets based on IPL seasons. Separate datasets are prepared for batting and bowling predictions.
2.  **Model Choice:** `GradientBoostingRegressor` from Scikit-learn is used for both batting and bowling fantasy point prediction.
3.  **Hyperparameter Tuning:** `RandomizedSearchCV` is employed over multiple iterations to find effective hyperparameters for the Gradient Boosting models (e.g., `n_estimators`, `learning_rate`, `max_depth`).
4.  **MLflow Tracking:**
    * The application connects to a configured MLflow Tracking Server (`mlflow.set_tracking_uri(...)`).
    * An MLflow Experiment named "Dream11" is used (`mlflow.set_experiment(...)`).
    * Each hyperparameter tuning iteration runs within a nested MLflow run (`mlflow.start_run(...)`).
    * Inside each run, the following are logged:
        * **Parameters:** The hyperparameters used for the batting and bowling models (`mlflow.log_params(...)`).
        * **Metrics:** The Root Mean Squared Error (RMSE) evaluated on the test set for both models (`mlflow.log_metric(...)`).
        * **Artifacts:** The trained Scikit-learn model objects themselves are logged (`mlflow.sklearn.log_model(...)`).
5.  **Model Selection & Registration:**
    * After all tuning iterations, the run corresponding to the lowest RMSE for the batting model and the bowling model is identified.
    * The best performing batting and bowling models are registered in the MLflow Model Registry under the names "Dream11BattingModel" and "Dream11BowlingModel" respectively (`mlflow.register_model(...)`). This allows for versioning and potential future deployment pipelines.
6.  **Usage:** The backend service loads and uses these best-trained models (retrieved via MLflow or potentially loaded directly after training in the current setup) for making predictions during runtime via the `FantasyScorer`.

This integration ensures that the training process is repeatable, experiments are tracked, and the best models are systematically identified and registered for use in the application.

## LLM Integration (Chat Bot)

The application includes a chat interface allowing users to ask general cricket-related questions. This functionality is powered by Google's Generative AI models:

1.  **Model Used:** The specific Large Language Model (LLM) employed is **`gemini-1.5-flash-latest`**. This model is chosen for its balance of performance, speed, and cost-effectiveness.
2.  **API Key:** Interaction with the Gemini API requires an API key, which is securely accessed from the `GEMINI_API_KEY` environment variable set during deployment.
3.  **Interaction Flow:**
    * When a user sends a message via the chat interface, the frontend sends it along with the recent conversation history to the backend's `/chat/` endpoint.
    * The backend handler (`llm.py`) first checks if the user's message contains keywords associated with team prediction (e.g., "predict team", "suggest", "generate"). If such keywords are found, it bypasses the LLM and returns a standard message guiding the user to use the prediction features in the sidebar. This prevents unnecessary API calls for tasks handled elsewhere.
    * If no prediction keywords are detected, the handler formats the conversation history into the structure required by the Gemini API.
    * It then initializes the `gemini-1.5-flash-latest` model using the `google-generativeai` Python library.
    * The user's message and the formatted history are sent to the Gemini API.
    * The text response generated by Gemini is received by the backend.
    * The backend updates the conversation history (in its own format) and sends the LLM's reply and the updated history back to the frontend for display.

## Tech Stack

* **Frontend:** Streamlit, Requests, Pandas, Matplotlib, Seaborn, Pillow
* **Backend:** FastAPI, Uvicorn, Pandas, Numpy, Scikit-learn, MLflow, Google Generative AI (Gemini 1.5 Flash), Google Cloud Secret Manager, Google Cloud Storage, GCSFS
* **MLOps/Infra:** Docker, Kubernetes (kubectl), Google Cloud Platform (GCP), Google Kubernetes Engine (GKE Autopilot), Google Artifact Registry
* **Data:** CSV files stored in Google Cloud Storage
* **Language:** Python

## Permissions Note

**Important:** 
- Deploying this application to Google Cloud Platform (GCP) and pushing updated container images requires specific IAM permissions within the `parlay-and-pray-app` GCP project. 
- These permissions must be granted by the project owner (Manikesh Makam).
- If you are a team member and need to perform deployment or development actions as outlined in `GCP GUIDE.md`, please ensure you have been granted the necessary roles (e.g., Editor, Artifact Registry Writer, Container Developer) as specified in the "Permissions" section of the guide.

## How to Start the Application

- To deploy and run the application on Google Cloud Platform, please follow the instructions in the **Deployment** section of the `GCP GUIDE.md` file. 
- This guide covers configuring `gcloud`, applying Kubernetes manifests (`*.yaml` files), checking deployment status, finding the live application URL, and terminating the deployment.

*(Note: Development instructions for building and pushing updated Docker images are also available in `GCP GUIDE.md` under the "Development" section. Remember the permissions note above.)*