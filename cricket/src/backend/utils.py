import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union

# Constants for role assignments
ROLE_PRIORITY = ['WK', 'AR', 'BWL', 'BAT']
MIN_ROLES = {"WK": 1, "BAT": 2, "AR": 1, "BWL": 2}
MAX_ROLES = {"WK": 2, "BAT": 5, "AR": 2, "BWL": 5}

# Constants for fantasy points calculation
CAPTAIN_MULTIPLIER = 1.5
VICE_CAPTAIN_MULTIPLIER = 1.25

# Constants for contextual factors
MIN_FACTOR = 0.8
MAX_FACTOR = 1.2
DEFAULT_FACTOR = 1.0

def get_player_role(batting_df: pd.DataFrame, bowling_df: pd.DataFrame, fielding_df: pd.DataFrame, 
                   player_name: str, known_wicketkeepers: Dict[str, List[str]]) -> str:
    """
    Determine player role based on fielding, batting, and bowling performance.
    Priority order: WK > AR > BWL > BAT
    """
    # Get player's data
    player_batting = batting_df[batting_df['fullName'] == player_name]
    player_bowling = bowling_df[bowling_df['fullName'] == player_name]
    player_fielding = fielding_df[fielding_df['fullName'] == player_name] if fielding_df is not None else pd.DataFrame()

    # 1. Check for Wicketkeeper (WK)
    # First check if player is in known wicketkeepers list
    for team, keepers in known_wicketkeepers.items():
        if player_name in keepers:
            return 'WK'
            
    # Then check fielding performance
    if not player_fielding.empty:
        # Must have stumping points to be considered a wicketkeeper
        if 'stumping_FP' in player_fielding.columns:
            stumping_fp = player_fielding['stumping_FP'].sum()
            if stumping_fp > 0:  # If player has any stumping points
                return 'WK'

    # 2. Check for All-Rounder (AR)
    if not player_batting.empty and not player_bowling.empty:
        avg_batting_fp = player_batting['Batting_FP'].mean()
        avg_bowling_fp = player_bowling['Bowling_FP'].mean()
        
        # Player must have significant contributions in both batting and bowling
        if avg_batting_fp >= 20 and avg_bowling_fp >= 20:
            return 'AR'
        # If one is very strong and other is decent
        elif (avg_batting_fp >= 30 and avg_bowling_fp >= 10) or (avg_bowling_fp >= 30 and avg_batting_fp >= 10):
            return 'AR'

    # 3. Check for Bowler (BWL)
    if not player_bowling.empty:
        avg_bowling_fp = player_bowling['Bowling_FP'].mean()
        if avg_bowling_fp >= 25:  # Strong bowling performance
            return 'BWL'

    # 4. Default to Batsman (BAT)
    return 'BAT'

def validate_role_counts(wk_count: int, bat_count: int, bwl_count: int, ar_count: int) -> tuple:
    """
    Validate and adjust role counts to ensure they sum to 11 and meet minimum requirements.
    Returns adjusted counts in the order: wk_count, bat_count, bwl_count, ar_count
    """
    total_roles = wk_count + bat_count + bwl_count + ar_count
    if total_roles != 11:
        print(f"⚠️ Warning: Total roles ({total_roles}) doesn't equal 11. Adjusting to meet requirements...")
        # Adjust counts proportionally to reach 11
        scale = 11 / total_roles
        wk_count = max(1, round(wk_count * scale))
        bat_count = max(2, round(bat_count * scale))
        bwl_count = max(2, round(bwl_count * scale))
        ar_count = max(1, round(ar_count * scale))
        # Final adjustment to ensure total is 11
        while wk_count + bat_count + bwl_count + ar_count > 11:
            if ar_count > 1:
                ar_count -= 1
            elif bat_count > 2:
                bat_count -= 1
            elif bwl_count > 2:
                bwl_count -= 1
        while wk_count + bat_count + bwl_count + ar_count < 11:
            if ar_count < 2:
                ar_count += 1
            elif bat_count < 3:
                bat_count += 1
            elif bwl_count < 3:
                bwl_count += 1
    
    return wk_count, bat_count, bwl_count, ar_count

def format_team_output(team_df: pd.DataFrame, captain: str, vice_captain: str) -> pd.DataFrame:
    """
    Format team output with captain and vice-captain information and effective points.
    """
    team_df = team_df.copy()
    team_df['captain'] = team_df['name'] == captain
    team_df['vice_captain'] = team_df['name'] == vice_captain
    
    # Calculate effective points
    team_df['effective_points'] = team_df.apply(
        lambda x: x['total_fp'] * CAPTAIN_MULTIPLIER if x['captain'] 
        else (x['total_fp'] * VICE_CAPTAIN_MULTIPLIER if x['vice_captain'] 
              else x['total_fp']),
        axis=1
    )
    
    return team_df

def calculate_team_stats(team_df: pd.DataFrame) -> Dict[str, float]:
    """
    Calculate team statistics including total credits, total points, and role distribution.
    """
    stats = {
        'total_credits': team_df['credit'].sum(),
        'total_points': team_df['total_fp'].sum(),
        'effective_points': team_df['effective_points'].sum(),
        'role_distribution': team_df['role'].value_counts().to_dict()
    }
    return stats

def format_role_distribution(role_counts: Dict[str, int]) -> str:
    """
    Format role distribution for display.
    """
    return ", ".join([f"{role}: {count}" for role, count in role_counts.items()]) 