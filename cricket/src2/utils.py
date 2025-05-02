import pandas as pd
import numpy as np
import os

def load_data():
    """Load all required datasets"""
    bucket_path = "gs://dream11-mlflow-bucket"
    
    batting = pd.read_csv(f"{bucket_path}/Batting_data.csv")
    bowling = pd.read_csv(f"{bucket_path}/Bowling_data.csv")
    fielding = pd.read_csv(f"{bucket_path}/Fielding_data.csv")
    fantasy = pd.read_csv(f"{bucket_path}/Final_Fantasy_data.csv")
    match = pd.read_csv(f"{bucket_path}/Match_details.csv")
    players = pd.read_csv(f"{bucket_path}/players.csv")
    return batting, bowling, fielding, fantasy, match, players

def filter_match_data(batting, bowling, fielding, home_team, away_team):
    """Filter data to include only the matches involving the specified teams"""
    team_batting = batting[
        ((batting['home_team'] == home_team) & (batting['away_team'] == away_team)) |
        ((batting['home_team'] == away_team) & (batting['away_team'] == home_team))
    ]
    
    team_bowling = bowling[
        ((bowling['home_team'] == home_team) & (bowling['away_team'] == away_team)) |
        ((bowling['home_team'] == away_team) & (bowling['away_team'] == home_team))
    ]
    
    team_fielding = fielding[
        ((fielding['home_team'] == home_team) & (fielding['away_team'] == away_team)) |
        ((fielding['home_team'] == away_team) & (fielding['away_team'] == home_team))
    ]
    
    return team_batting, team_bowling, team_fielding

def get_player_stats(df, player_name, column):
    """Get statistics for a player from a dataframe"""
    player_data = df[df['fullName'] == player_name]
    if player_data.empty or column not in player_data.columns:
        return None
    return player_data[column].mean()

def get_player_historical_data(batting, bowling, fielding, player_name):
    """Get historical performance data for a player"""
    player_batting = batting[batting['fullName'] == player_name]
    player_bowling = bowling[bowling['fullName'] == player_name]
    player_fielding = fielding[fielding['fullName'] == player_name]
    
    return player_batting, player_bowling, player_fielding

def get_venue_performance(batting, bowling, player_name, venue):
    """Get player performance at specific venue"""
    venue_batting = batting[(batting['fullName'] == player_name) & (batting['venue'] == venue)]
    venue_bowling = bowling[(bowling['fullName'] == player_name) & (bowling['venue'] == venue)]
    
    batting_stats = {
        'matches': len(venue_batting),
        'avg_runs': venue_batting['runs'].mean() if not venue_batting.empty else 0,
        'avg_sr': venue_batting['strike_rate'].mean() if not venue_batting.empty else 0,
        'avg_fp': venue_batting['Batting_FP'].mean() if not venue_batting.empty else 0
    }
    
    bowling_stats = {
        'matches': len(venue_bowling),
        'avg_wickets': venue_bowling['wickets'].mean() if not venue_bowling.empty else 0,
        'avg_economy': venue_bowling['economyRate'].mean() if not venue_bowling.empty else 0,
        'avg_fp': venue_bowling['Bowling_FP'].mean() if not venue_bowling.empty else 0
    }
    
    return batting_stats, bowling_stats

def get_opponent_performance(batting, bowling, player_name, opponent):
    """Get player performance against specific opponent"""
    opponent_batting = batting[(batting['fullName'] == player_name) & 
                              ((batting['bowling_team'] == opponent) |
                               (batting['batting_team'] == opponent))]
    
    opponent_bowling = bowling[(bowling['fullName'] == player_name) & 
                              ((bowling['batting_team'] == opponent) |
                               (bowling['bowling_team'] == opponent))]
    
    batting_stats = {
        'matches': len(opponent_batting),
        'avg_runs': opponent_batting['runs'].mean() if not opponent_batting.empty else 0,
        'avg_sr': opponent_batting['strike_rate'].mean() if not opponent_batting.empty else 0,
        'avg_fp': opponent_batting['Batting_FP'].mean() if not opponent_batting.empty else 0
    }
    
    bowling_stats = {
        'matches': len(opponent_bowling),
        'avg_wickets': opponent_bowling['wickets'].mean() if not opponent_bowling.empty else 0,
        'avg_economy': opponent_bowling['economyRate'].mean() if not opponent_bowling.empty else 0,
        'avg_fp': opponent_bowling['Bowling_FP'].mean() if not opponent_bowling.empty else 0
    }
    
    return batting_stats, bowling_stats
