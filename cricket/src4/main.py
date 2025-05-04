from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, List, Optional
import pandas as pd
import sys
import os
from pathlib import Path

# Add src2 to Python path
current_dir = Path(__file__).parent.absolute()
sys.path.append(str(current_dir))

from preprocessing import DataPreprocessor
from training import ModelTrainer
from scoring import FantasyScorer
from utils import get_player_role

app = FastAPI(title="Fantasy Cricket Prediction API")

# Initialize components with correct paths
preprocessor = DataPreprocessor()
model_trainer = ModelTrainer(model_dir=str(current_dir.parent / "models"))
scorer = FantasyScorer(preprocessor, model_trainer)

# Load data and train models on startup
@app.on_event("startup")
async def startup_event():
    print("Initializing Dream11 Prediction System...")
    preprocessor.load_data()
    model_trainer.train_models(preprocessor)
    print("System initialized and ready!")

class TeamPredictionRequest(BaseModel):
    home_team: str
    away_team: str
    venue: Optional[str] = None
    train_new: Optional[bool] = False
    wk_count: int
    bat_count: int
    ar_count: int
    bwl_count: int

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

@app.get("/")
async def root():
    return {"message": "Fantasy Cricket Prediction API is running"}

@app.post("/predict_team/")
async def predict_team(request: TeamPredictionRequest):
    try:
        # Prepare match data
        print(f"\nPreparing match data for {request.home_team} vs {request.away_team}...")
        all_players, team_batting, team_bowling, team_fielding, venue = preprocessor.prepare_match_data(
            request.home_team, request.away_team
        )
        
        # Predict fantasy points
        print("\nPredicting fantasy points...")
        df = scorer.predict_fantasy_points(team_batting, team_bowling, team_fielding)
        
        # Apply contextual factors
        print("\nApplying contextual factors...")
        df = scorer.apply_contextual_factors(
            df, team_batting, team_bowling, venue, request.home_team, request.away_team
        )
        
        # Assign roles to players
        print("\nAssigning roles to players...")
        df['role'] = df['fullName'].apply(
            lambda x: get_player_role(team_batting, team_bowling, team_fielding, x, preprocessor.KNOWN_WICKETKEEPERS)
        )
        
        # Add credit information
        print("\nAdding credit information...")
        player_info = preprocessor.players.set_index('players')
        df['credit'] = df['fullName'].map(player_info['credit'])
        df['credit'] = df['credit'].fillna(preprocessor.players['credit'].mean())
        
        # Build team with user-specified role counts
        user_roles = {
            "WK": request.wk_count,
            "BAT": request.bat_count,
            "AR": request.ar_count,
            "BWL": request.bwl_count
        }
        
        print("\nBuilding team...")
        team_df = scorer.build_role_constrained_team(
            df, preprocessor.players, venue, request.home_team, request.away_team,
            user_roles=user_roles
        )
        
        # Rename columns for consistency
        team_df = team_df.rename(columns={
            'fullName': 'name',
            'player_team': 'team'
        })
        
        # Select captain and vice-captain
        captain, vice_captain = scorer.select_captain_vice_captain(team_df)
        
        # Calculate team stats
        team_stats = scorer.calculate_team_stats(team_df)
        
        # Format response to match existing frontend expectations
        team_summary = {
            "players": team_df.to_dict(orient="records"),
            "captain": captain,
            "vice_captain": vice_captain,
            "total_credits": team_stats['total_credits'],
            "total_points": team_stats['total_points'],
            "role_distribution": team_stats['role_distribution']
        }
        
        return {
            "team_summary": team_summary,
            "team_raw": team_df.to_dict(orient="records")
        }
        
    except Exception as e:
        print(f"Error in predict_team: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze_player/")
async def analyze_player(request: PlayerAnalysisRequest):
    try:
        # Get player's historical data
        player_batting = preprocessor.batting_data[preprocessor.batting_data['fullName'] == request.player_name]
        player_bowling = preprocessor.bowling_data[preprocessor.bowling_data['fullName'] == request.player_name]
        
        if player_batting.empty and player_bowling.empty:
            raise HTTPException(status_code=404, detail="Player not found")
        
        # Calculate basic stats
        matches_played = max(len(player_batting), len(player_bowling))
        avg_runs = player_batting['runs'].mean() if not player_batting.empty else None
        avg_sr = player_batting['strike_rate'].mean() if not player_batting.empty else None
        avg_fp = (player_batting['Batting_FP'].sum() + player_bowling['Bowling_FP'].sum()) / matches_played
        
        # Get role
        role = get_player_role(
            player_batting, player_bowling, None, request.player_name, preprocessor.KNOWN_WICKETKEEPERS
        )
        
        return PlayerAnalysisResponse(
            role=role,
            matches_played=matches_played,
            avg_runs=avg_runs,
            avg_sr=avg_sr,
            avg_fp=avg_fp,
            performance_at_venue="Not implemented",  # Can be implemented if needed
            performance_against_opponent="Not implemented"  # Can be implemented if needed
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)