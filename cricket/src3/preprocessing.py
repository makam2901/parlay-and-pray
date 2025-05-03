import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

def preprocess_batting(batting_df, year_split=2023):
    """
    Preprocess batting data with enhanced feature engineering
    Returns training data (pre-2023) and test data (2023)
    """
    batting_df = batting_df.copy()
    
    # Basic cleaning
    batting_df.dropna(subset=['Batting_FP'], inplace=True)
    batting_df = batting_df[batting_df['balls'] > 0]
    
    # Feature engineering
    batting_df['boundary_percentage'] = (batting_df['fours'] + batting_df['sixes']) / batting_df['balls']
    batting_df['boundary_percentage'] = batting_df['boundary_percentage'].fillna(0)
    batting_df['runs_per_ball'] = batting_df['runs'] / batting_df['balls']
    batting_df['runs_per_ball'] = batting_df['runs_per_ball'].fillna(0)
    batting_df['boundary_runs'] = batting_df['fours'] * 4 + batting_df['sixes'] * 6
    batting_df['non_boundary_runs'] = batting_df['runs'] - batting_df['boundary_runs']
    batting_df['non_boundary_balls'] = batting_df['balls'] - batting_df['fours'] - batting_df['sixes']
    
    # Random credit assignment for simulation
    batting_df['credit'] = np.random.uniform(7.0, 10.0, size=len(batting_df)).round(1)
    
    # Handle categorical data
    categorical_cols = ['season', 'home_team', 'away_team', 'batting_team', 'bowling_team', 'venue']
    batting_df[categorical_cols] = batting_df[categorical_cols].astype('category')
    
    # Convert season to numeric for chronological analysis
    batting_df['season'] = pd.to_numeric(batting_df['season'])
    
    # Create recent form feature (last 5 matches average)
    batting_df = batting_df.sort_values(['fullName', 'match_id'])
    batting_df['recent_form'] = batting_df.groupby('fullName')['Batting_FP'].transform(
        lambda x: x.shift(1).rolling(window=5, min_periods=1).mean()
    )
    
    # Split data by year
    train_batting = batting_df[batting_df['season'] < year_split]
    test_batting = batting_df[batting_df['season'] >= year_split]
    
    # Define features for training
    features = [
        'runs', 'balls', 'fours', 'sixes', 'strike_rate', 'batting_position', 
        'boundary_percentage', 'runs_per_ball', 'boundary_runs', 'non_boundary_runs',
        'recent_form', 'batting_innings'
    ]
    
    X_train = train_batting[features]
    y_train = train_batting['Batting_FP']
    
    X_test = test_batting[features]
    y_test = test_batting['Batting_FP']
    
    return X_train, y_train, X_test, y_test, train_batting, test_batting

def preprocess_bowling(bowling_df, year_split=2023):
    """
    Preprocess bowling data with enhanced feature engineering
    Returns training data (pre-2023) and test data (2023)
    """
    bowling_df = bowling_df.copy()
    
    # Basic cleaning
    bowling_df.dropna(subset=['Bowling_FP'], inplace=True)
    
    # Feature engineering
    bowling_df['overs'] = pd.to_numeric(bowling_df['overs'], errors='coerce')
    bowling_df['balls_bowled'] = bowling_df['overs'].apply(
        lambda x: int(x) * 6 + (x % 1) * 10 if pd.notnull(x) else 0
    )
    bowling_df['dot_balls'] = bowling_df['balls_bowled'] - bowling_df['conceded']
    bowling_df['dot_ball_percentage'] = bowling_df['dot_balls'] / bowling_df['balls_bowled']
    bowling_df['dot_ball_percentage'] = bowling_df['dot_ball_percentage'].fillna(0)
    bowling_df['wicket_rate'] = bowling_df['wickets'] / bowling_df['balls_bowled']
    bowling_df['wicket_rate'] = bowling_df['wicket_rate'].fillna(0)
    
    # Handle categorical data
    categorical_cols = ['season', 'home_team', 'away_team', 'batting_team', 'bowling_team', 'venue']
    bowling_df[categorical_cols] = bowling_df[categorical_cols].astype('category')
    
    # Convert season to numeric for chronological analysis
    bowling_df['season'] = pd.to_numeric(bowling_df['season'])
    
    # Create recent form feature (last 5 matches average)
    bowling_df = bowling_df.sort_values(['fullName', 'match_id'])
    bowling_df['recent_form'] = bowling_df.groupby('fullName')['Bowling_FP'].transform(
        lambda x: x.shift(1).rolling(window=5, min_periods=1).mean()
    )
    
    # Split data by year
    train_bowling = bowling_df[bowling_df['season'] < year_split]
    test_bowling = bowling_df[bowling_df['season'] >= year_split]
    
    # Define features for training
    features = [
        'overs', 'maidens', 'conceded', 'wickets', 'economyRate', 'wides', 'noballs',
        'dot_ball_percentage', 'wicket_rate', 'recent_form', 'bowling_innings'
    ]
    
    X_train = train_bowling[features]
    y_train = train_bowling['Bowling_FP']
    
    X_test = test_bowling[features]
    y_test = test_bowling['Bowling_FP']
    
    return X_train, y_train, X_test, y_test, train_bowling, test_bowling

def preprocess_fielding(fielding_df, year_split=2023):
    """
    Preprocess fielding data
    Returns training data (pre-2023) and test data (2023)
    """
    fielding_df = fielding_df.copy()
    
    # Basic cleaning
    fielding_df.dropna(subset=['Fielding_FP'], inplace=True)
    
    # Handle categorical data
    categorical_cols = ['season', 'home_team', 'away_team', 'batting_team', 'bowling_team', 'venue']
    for col in categorical_cols:
        if col in fielding_df.columns:
            fielding_df[col] = fielding_df[col].astype('category')
    
    # Convert season to numeric for chronological analysis
    fielding_df['season'] = pd.to_numeric(fielding_df['season'])
    
    # Split data by year
    train_fielding = fielding_df[fielding_df['season'] < year_split]
    test_fielding = fielding_df[fielding_df['season'] >= year_split]
    
    # Define features for training
    features = ['catching_FP', 'stumping_FP', 'direct_runout_FP', 'indirect_runout_FP']
    
    X_train = train_fielding[features]
    y_train = train_fielding['Fielding_FP']
    
    X_test = test_fielding[features]
    y_test = test_fielding['Fielding_FP']
    
    return X_train, y_train, X_test, y_test, train_fielding, test_fielding

def get_player_role(batting_df, bowling_df, fielding_df, player_name):
    """Determine player role based on historical data"""
    # Get player data
    player_batting = batting_df[batting_df['fullName'] == player_name]
    player_bowling = bowling_df[bowling_df['fullName'] == player_name]
    player_fielding = fielding_df[fielding_df['fullName'] == player_name]
    
    # Check if player is a wicketkeeper
    is_wk = False
    if not player_fielding.empty and 'stumping_FP' in player_fielding.columns:
        is_wk = player_fielding['stumping_FP'].sum() > 0
    
    # Calculate average batting and bowling stats
    avg_batting_fp = player_batting['Batting_FP'].mean() if not player_batting.empty else 0
    avg_bowling_fp = player_bowling['Bowling_FP'].mean() if not player_bowling.empty else 0
    avg_bowling_wickets = player_bowling['wickets'].mean() if not player_bowling.empty else 0
    
    # Assign role based on stats
    if is_wk:
        return 'WK'
    elif avg_bowling_fp >= 20 and avg_batting_fp >= 20:
        return 'AR'  # All-rounder
    elif avg_bowling_fp >= 20 or avg_bowling_wickets >= 1:
        return 'BWL'  # Bowler
    else:
        return 'BAT'  # Batsman