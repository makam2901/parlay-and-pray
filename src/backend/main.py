# src/backend/main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, List, Optional
import pandas as pd
import sys
import os
from pathlib import Path
import logging # Added logging

# --- Add current directory to sys.path ---
current_dir = Path(__file__).parent.absolute()
sys.path.append(str(current_dir))

# --- Import your modules ---
from preprocessing import DataPreprocessor
from training import ModelTrainer
from scoring import FantasyScorer
from utils import get_player_role
from llm import get_chat_response # Import the new LLM function

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = FastAPI(title="Fantasy Cricket Prediction API")

# --- Global Variables / State ---
preprocessor = None
model_trainer = None
scorer = None
gemini_api_key = None # To store the API key

# --- Load data and train models on startup ---
@app.on_event("startup")
async def startup_event():
    global preprocessor, model_trainer, scorer, gemini_api_key # Make them global
    logging.info("Initializing Dream11 Prediction System...")
    try:
        preprocessor = DataPreprocessor()
        model_trainer = ModelTrainer(model_dir=str(current_dir.parent / "models")) # Ensure model dir is correct
        scorer = FantasyScorer(preprocessor, model_trainer)

        preprocessor.load_data()
        model_trainer.train_models(preprocessor)

        # --- Get Gemini API Key from environment variable ---
        gemini_api_key = os.getenv("GEMINI_API_KEY")
        if not gemini_api_key:
            logging.warning("GEMINI_API_KEY environment variable not set. Chat functionality will not work.")
        else:
             logging.info("GEMINI_API_KEY loaded successfully.")

        logging.info("System initialized and ready!!")

    except Exception as e:
        logging.error(f"Error during startup: {e}", exc_info=True)
        # Depending on severity, you might want to exit or handle differently
        raise HTTPException(status_code=500, detail=f"System initialization failed: {e}")


# --- Pydantic Models ---
class TeamPredictionRequest(BaseModel):
    home_team: str
    away_team: str
    venue: Optional[str] = None
    # train_new: Optional[bool] = False # Removed as train_models is now on startup
    wk_count: int
    bat_count: int
    ar_count: int
    bwl_count: int
    # Added max_credits and max_per_team as they seem to be used in frontend/scoring
    max_credits: Optional[int] = 100
    max_per_team: Optional[int] = 7


class PlayerAnalysisRequest(BaseModel):
    player_name: str

class PlayerAnalysisResponse(BaseModel):
    role: str
    matches_played: int
    avg_runs: Optional[float] = None
    avg_sr: Optional[float] = None
    avg_fp: Optional[float] = None
    performance_at_venue: Optional[str] = None
    performance_against_opponent: Optional[str] = None

# --- NEW: Chat Models ---
class ChatMessage(BaseModel):
    role: str # 'user' or 'assistant'/'model'
    content: str

class ChatRequest(BaseModel):
    message: str
    history: List[ChatMessage] = []

class ChatResponse(BaseModel):
    reply: str
    history: List[ChatMessage]


# --- API Endpoints ---
@app.get("/")
async def root():
    return {"message": "Fantasy Cricket Prediction API is running"}

# --- NEW: Chat Endpoint ---
@app.post("/chat/", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    global gemini_api_key
    if not gemini_api_key:
         logging.error("Chat endpoint called but GEMINI_API_KEY is not configured.")
         raise HTTPException(status_code=503, detail="Chat service is not configured (API Key missing).")

    logging.info(f"Received chat message: {request.message}")
    logging.debug(f"Received chat history length: {len(request.history)}")

    try:
        # Convert Pydantic models to simple dicts for the llm function
        history_dicts = [msg.dict() for msg in request.history]

        reply, updated_history_dicts = get_chat_response(
            api_key=gemini_api_key,
            user_message=request.message,
            history=history_dicts
        )

        # Convert the updated history back to Pydantic models
        updated_history_models = [ChatMessage(**msg) for msg in updated_history_dicts]

        logging.info(f"Sending reply: {reply[:100]}...")
        return ChatResponse(reply=reply, history=updated_history_models)

    except Exception as e:
        logging.error(f"Error processing chat request: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing chat request: {e}")

# --- Existing Endpoints (Modified slightly for clarity/consistency) ---
@app.post("/predict_team/")
async def predict_team(request: TeamPredictionRequest):
    global preprocessor, scorer # Ensure access to initialized components
    if not preprocessor or not scorer or not preprocessor.data_loaded or not model_trainer.models_trained:
        logging.error("Predict endpoint called before system initialization completed.")
        raise HTTPException(status_code=503, detail="Prediction service is not ready yet.")

    try:
        logging.info(f"\nReceived prediction request for {request.home_team} vs {request.away_team}...")

        # Prepare match data
        logging.info("Preparing match data...")
        all_players, team_batting, team_bowling, team_fielding, venue = preprocessor.prepare_match_data(
            request.home_team, request.away_team
        )

        # Use request venue if provided, otherwise use the one derived from data
        match_venue = request.venue if request.venue else venue
        if not match_venue:
             raise ValueError("Match venue could not be determined.")
        logging.info(f"Using venue: {match_venue}")


        # Predict fantasy points
        logging.info("Predicting fantasy points...")
        df = scorer.predict_fantasy_points(team_batting, team_bowling, team_fielding)

        # Apply contextual factors
        logging.info("Applying contextual factors...")
        df = scorer.apply_contextual_factors(
            df, team_batting, team_bowling, match_venue, request.home_team, request.away_team
        )

        # Assign roles to players
        logging.info("Assigning roles to players...")
        df['role'] = df['fullName'].apply(
            lambda x: get_player_role(team_batting, team_bowling, team_fielding, x, preprocessor.KNOWN_WICKETKEEPERS)
        )

        # Add credit information
        logging.info("Adding credit information...")
        player_info = preprocessor.players.set_index('players')
        df['credit'] = df['fullName'].map(player_info['credit'])
        # Ensure credits are numeric and handle potential NaNsrobustly
        df['credit'] = pd.to_numeric(df['credit'], errors='coerce')
        avg_credit = preprocessor.players['credit'].mean() # Calculate average credit
        df['credit'] = df['credit'].fillna(avg_credit) # Fill NaNs with average
        logging.info(f"Filled {df['credit'].isna().sum()} missing credits with average {avg_credit:.2f}")


        # Build team with user-specified role counts
        user_roles = {
            "WK": request.wk_count,
            "BAT": request.bat_count,
            "AR": request.ar_count,
            "BWL": request.bwl_count
        }
        logging.info(f"User specified roles: {user_roles}")

        logging.info("Building team...")
        team_df = scorer.build_role_constrained_team(
            df, preprocessor.players, match_venue, request.home_team, request.away_team,
            max_credits=request.max_credits, # Pass constraints from request
            max_per_team=request.max_per_team,
            user_roles=user_roles
        )

        if team_df.empty:
            logging.warning("Team building resulted in an empty DataFrame.")
            raise HTTPException(status_code=400, detail="Could not generate a team with the given constraints. Try adjusting role counts or constraints.")


        # Select captain and vice-captain
        captain, vice_captain = scorer.select_captain_vice_captain(team_df)
        logging.info(f"Selected Captain: {captain}, Vice-Captain: {vice_captain}")

        # Calculate team stats
        team_stats = scorer.calculate_team_stats(team_df)
        logging.info(f"Team stats: {team_stats}")

        # Format response
        team_dict_list = team_df.to_dict(orient="records")

        # --- Add context factors to the response for player details tab ---
        context_factors_map = df.set_index('fullName')[['form_factor', 'venue_factor', 'opponent_factor', 'Adjusted_Total_FP']].to_dict('index')
        for player_dict in team_dict_list:
             player_name = player_dict.get('name')
             if player_name in context_factors_map:
                 factors = context_factors_map[player_name]
                 player_dict['form_factor'] = factors.get('form_factor')
                 player_dict['venue_factor'] = factors.get('venue_factor')
                 player_dict['opponent_factor'] = factors.get('opponent_factor')
                 player_dict['adjusted_fp'] = factors.get('Adjusted_Total_FP') # Add adjusted points

        team_summary = {
            # "players": team_dict_list, # Keep raw list separate
            "captain": captain,
            "vice_captain": vice_captain,
            "total_credits": team_stats['total_credits'],
            "total_points": team_stats['total_points'], # This is the sum of Adjusted_Total_FP if available
            "role_distribution": team_stats['role_distribution']
        }

        return {
            "team_summary": team_summary,
            "team_raw": team_dict_list # Send the detailed list here
        }

    except ValueError as ve: # Catch specific expected errors like venue issues
         logging.error(f"Value error during prediction: {ve}", exc_info=True)
         raise HTTPException(status_code=400, detail=str(ve))
    except HTTPException as he: # Re-raise HTTP exceptions
         raise he
    except Exception as e:
        logging.error(f"Unexpected error in predict_team: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {e}")


@app.post("/analyze_player/", response_model=PlayerAnalysisResponse)
async def analyze_player(request: PlayerAnalysisRequest):
    global preprocessor
    if not preprocessor or not preprocessor.data_loaded:
         logging.error("Analyze player endpoint called before system initialization completed.")
         raise HTTPException(status_code=503, detail="Analysis service is not ready yet.")
    try:
        # Get player's historical data - Use the data stored in the preprocessor
        player_batting = preprocessor.batting[preprocessor.batting['fullName'] == request.player_name]
        player_bowling = preprocessor.bowling[preprocessor.bowling['fullName'] == request.player_name]
        player_fielding = preprocessor.fielding[preprocessor.fielding['fullName'] == request.player_name]


        if player_batting.empty and player_bowling.empty and player_fielding.empty:
            raise HTTPException(status_code=404, detail=f"Player '{request.player_name}' not found in historical data.")

        # Calculate basic stats
        # Consider unique matches if a player batted and bowled in the same match
        bat_matches = set(player_batting['match_id']) if 'match_id' in player_batting else set()
        bowl_matches = set(player_bowling['match_id']) if 'match_id' in player_bowling else set()
        matches_played = len(bat_matches.union(bowl_matches))

        avg_runs = player_batting['runs'].mean() if not player_batting.empty else 0.0
        avg_sr = player_batting['strike_rate'].mean() if not player_batting.empty else 0.0

        total_bat_fp = player_batting['Batting_FP'].sum() if 'Batting_FP' in player_batting else 0
        total_bowl_fp = player_bowling['Bowling_FP'].sum() if 'Bowling_FP' in player_bowling else 0
        total_field_fp = player_fielding['Fielding_FP'].sum() if 'Fielding_FP' in player_fielding else 0 # Add fielding
        total_fp = total_bat_fp + total_bowl_fp + total_field_fp
        avg_fp = total_fp / matches_played if matches_played > 0 else 0.0

        # Get role
        role = get_player_role(
            preprocessor.batting, # Pass full dfs for context if needed by role func
            preprocessor.bowling,
            preprocessor.fielding, # Pass fielding df
            request.player_name,
            preprocessor.KNOWN_WICKETKEEPERS
        )

        # Convert numpy types to standard Python types for JSON serialization if necessary
        avg_runs = float(avg_runs) if avg_runs is not None else None
        avg_sr = float(avg_sr) if avg_sr is not None else None
        avg_fp = float(avg_fp) if avg_fp is not None else None


        return PlayerAnalysisResponse(
            role=role,
            matches_played=matches_played,
            avg_runs=avg_runs,
            avg_sr=avg_sr,
            avg_fp=avg_fp,
            performance_at_venue="Not implemented",
            performance_against_opponent="Not implemented"
        )
    except HTTPException as he: # Re-raise HTTP exceptions
        raise he
    except Exception as e:
        logging.error(f"Error analyzing player {request.player_name}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error analyzing player: {e}")

# --- Main execution block (for local testing) ---
if __name__ == "__main__":
    import uvicorn
    # Load env vars from .env file for local dev if it exists
    # from dotenv import load_dotenv
    # load_dotenv()
    port = int(os.getenv("PORT", 8000)) # Use PORT env var if set (common in cloud run)
    uvicorn.run(app, host="0.0.0.0", port=port)