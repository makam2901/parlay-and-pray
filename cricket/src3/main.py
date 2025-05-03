from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from src3.execute import Dream11Predictor
from src3.scoring import get_team_summary


app = FastAPI()

# Initialize the Dream11 predictor
predictor = Dream11Predictor(model_dir="models")

# Define a Pydantic model to capture user input for team prediction
class TeamPredictionRequest(BaseModel):
    home_team: str
    away_team: str
    venue: Optional[str] = None
    train_new: Optional[bool] = False
    # Role-specific input from user
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


@app.post("/predict_team/")
async def predict_team(request: TeamPredictionRequest):
    try:
        # Validate constraints...
        
        final_team = predictor.predict_team(
            home_team=request.home_team,
            away_team=request.away_team,
            min_roles={"WK": request.wk_count, "BAT": request.bat_count,
                       "AR": request.ar_count, "BWL": request.bwl_count},
            venue=request.venue,
            load_saved_models=not request.train_new
        )

        # Get structured team summary and pretty string
        team_summary, pretty_team_log = get_team_summary(final_team)

        # Optional: log for debug
        print(pretty_team_log)

        return {
            "team_summary": team_summary,
            "team_raw": final_team.to_dict(orient="records")  # just in case you want raw too
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analyze_player/")
async def analyze_player(request: PlayerAnalysisRequest):
    try:
        result = predictor.analyze_player(request.player_name)

        return {
            "player_name": result["player_name"],
            "role": result["role"],
            "matches_played": result["matches_played"],
            "batting_stats": result["batting_stats"],
            "bowling_stats": result["bowling_stats"],
            "venue_performance": result["venue_performance"],
            "opponent_performance": result["opponent_performance"]
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run the app (use `uvicorn src2.main_api:app --reload` in terminal)