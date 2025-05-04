import pandas as pd
import numpy as np
import os

class DataPreprocessor:
    """Handles data loading and preprocessing for Dream11 predictions"""
    
    # Known wicketkeepers by team
    KNOWN_WICKETKEEPERS = {
        'CSK': ['MS Dhoni', 'Devon Conway'],
        'DC': ['Abishek Porel', 'Sarfaraz Khan'],
        'GT': ['Wriddhiman Saha'],
        'KKR': ['Rahmanullah Gurbaz', 'Litton Das'],
        'LSG': ['KL Rahul', 'Quinton de Kock'],
        'MI': ['Ishan Kishan', 'Vishnu Vinod'],
        'PBKS': ['Jitesh Sharma', 'Prabhsimran Singh'],
        'RCB': ['Dinesh Karthik', 'Anuj Rawat'],
        'RR': ['Sanju Samson', 'Jos Buttler', 'Dhruv Jurel'],
        'SRH': ['Heinrich Klaasen', 'Glenn Phillips']
    }
    
    def __init__(self):
        """Initialize the data preprocessor"""
        self.data_loaded = False
        self.batting = None
        self.bowling = None
        self.fielding = None
        self.match = None
        self.players = None
        
    def load_data(self):
        """Load the datasets"""
        print("Loading datasets...")
        self.batting = pd.read_csv("data/Batting_data.csv")
        self.bowling = pd.read_csv("data/Bowling_data.csv")
        self.fielding = pd.read_csv("data/Fielding_data.csv")
        self.match = pd.read_csv("data/Match_details.csv")
        self.players = pd.read_csv("data/players.csv")
        self.data_loaded = True
        print(f"Data loaded successfully: {len(self.batting)} batting records, {len(self.bowling)} bowling records")
        
    def get_team_players(self, team_name):
        """Get current players for a team"""
        team_players = self.players[self.players['team'] == team_name]['players'].tolist()
        if not team_players:
            raise ValueError(f"No players found for team: {team_name}")
        return team_players
        
    def prepare_match_data(self, home_team, away_team):
        """Prepare data for model training for specific teams"""
        if not self.data_loaded:
            self.load_data()
            
        print(f"Preparing match data for {home_team} vs {away_team}...")
        
        # Get current players for both teams
        home_players = self.get_team_players(home_team)
        away_players = self.get_team_players(away_team)
        all_players = home_players + away_players
        
        # Filter data for these players only
        team_batting = self.batting[self.batting['fullName'].isin(all_players)].copy()
        team_bowling = self.bowling[self.bowling['fullName'].isin(all_players)].copy()
        team_fielding = self.fielding[self.fielding['fullName'].isin(all_players)].copy()
        
        # Map team and credit information
        player_info = self.players.set_index('players')
        
        # Add team information
        team_batting['player_team'] = team_batting['fullName'].map(player_info['team'])
        team_bowling['player_team'] = team_bowling['fullName'].map(player_info['team'])
        team_fielding['player_team'] = team_fielding['fullName'].map(player_info['team'])
        
        # Add credit information
        team_batting['credit'] = team_batting['fullName'].map(player_info['credit'])
        team_bowling['credit'] = team_bowling['fullName'].map(player_info['credit'])
        team_fielding['credit'] = team_fielding['fullName'].map(player_info['credit'])
        
        # Fill missing credits with average
        avg_credit = self.players['credit'].mean()
        team_batting['credit'] = team_batting['credit'].fillna(avg_credit)
        team_bowling['credit'] = team_bowling['credit'].fillna(avg_credit)
        team_fielding['credit'] = team_fielding['credit'].fillna(avg_credit)
        
        # Get venue information
        venues = self.match[
            ((self.match['home_team'] == home_team) & (self.match['away_team'] == away_team)) |
            ((self.match['home_team'] == away_team) & (self.match['away_team'] == home_team))
        ]['venue'].value_counts()
        
        venue = venues.index[0] if not venues.empty else None
        
        print(f"Prepared data for {len(all_players)} players")
        print(f"Selected venue: {venue}")
        
        return all_players, team_batting, team_bowling, team_fielding, venue
        
    def preprocess_batting(self, batting_df, year_split=2023):
        """Preprocess batting data with enhanced feature engineering"""
        batting_df = batting_df.copy()
        
        # Basic cleaning
        batting_df.dropna(subset=['Batting_FP'], inplace=True)
        batting_df = batting_df[batting_df['balls'] > 0]
        
        # Feature engineering
        batting_df['boundary_percentage'] = batting_df.apply(
            lambda x: (x['fours'] + x['sixes']) / x['balls'] if x['balls'] > 0 else 0, 
            axis=1
        )
        
        batting_df['runs_per_ball'] = batting_df.apply(
            lambda x: x['runs'] / x['balls'] if x['balls'] > 0 else 0,
            axis=1
        )
        
        batting_df['boundary_runs'] = batting_df['fours'] * 4 + batting_df['sixes'] * 6
        batting_df['non_boundary_runs'] = batting_df['runs'] - batting_df['boundary_runs']
        
        batting_df['non_boundary_balls'] = batting_df.apply(
            lambda x: max(0, x['balls'] - x['fours'] - x['sixes']),
            axis=1
        )
        
        # Handle categorical data
        categorical_cols = ['season', 'home_team', 'away_team', 'batting_team', 'bowling_team', 'venue']
        for col in categorical_cols:
            if col in batting_df.columns:
                batting_df[col] = batting_df[col].astype('category')
        
        # Convert season to numeric
        batting_df['season'] = pd.to_numeric(batting_df['season'], errors='coerce')
        
        # Create recent form feature
        batting_df = batting_df.sort_values(['fullName', 'match_id'])
        batting_df['recent_form'] = batting_df.groupby('fullName')['Batting_FP'].transform(
            lambda x: x.shift(1).rolling(window=5, min_periods=1).mean()
        )
        batting_df['recent_form'] = batting_df['recent_form'].fillna(0)
        
        # Split data by year
        train_batting = batting_df[batting_df['season'] < year_split]
        test_batting = batting_df[batting_df['season'] >= year_split]
        
        # Define features
        features = [
            'runs', 'balls', 'fours', 'sixes', 'strike_rate', 'batting_position', 
            'boundary_percentage', 'runs_per_ball', 'boundary_runs', 'non_boundary_runs',
            'recent_form', 'batting_innings'
        ]
        
        # Ensure all features exist
        for feature in features:
            if feature not in train_batting.columns:
                print(f"Warning: Feature '{feature}' not found in training data. Adding with zeros.")
                train_batting[feature] = 0
                test_batting[feature] = 0
        
        return train_batting[features], train_batting['Batting_FP'], test_batting[features], test_batting['Batting_FP']
        
    def preprocess_bowling(self, bowling_df, year_split=2023):
        """Preprocess bowling data with enhanced feature engineering"""
        bowling_df = bowling_df.copy()
        
        # Basic cleaning
        bowling_df.dropna(subset=['Bowling_FP'], inplace=True)
        
        # Feature engineering
        bowling_df['overs'] = pd.to_numeric(bowling_df['overs'], errors='coerce')
        
        bowling_df['balls_bowled'] = bowling_df['overs'].apply(
            lambda x: int(x) * 6 + (x % 1) * 10 if pd.notnull(x) else 0
        )
        
        bowling_df['dot_balls'] = bowling_df.apply(
            lambda x: max(0, x['balls_bowled'] - x['conceded']) if x['balls_bowled'] > 0 else 0,
            axis=1
        )
        
        bowling_df['dot_ball_percentage'] = bowling_df.apply(
            lambda x: x['dot_balls'] / x['balls_bowled'] if x['balls_bowled'] > 0 else 0,
            axis=1
        )
        
        bowling_df['wicket_rate'] = bowling_df.apply(
            lambda x: x['wickets'] / x['balls_bowled'] if x['balls_bowled'] > 0 else 0,
            axis=1
        )
        
        # Handle categorical data
        categorical_cols = ['season', 'home_team', 'away_team', 'batting_team', 'bowling_team', 'venue']
        for col in categorical_cols:
            if col in bowling_df.columns:
                bowling_df[col] = bowling_df[col].astype('category')
        
        # Convert season to numeric
        bowling_df['season'] = pd.to_numeric(bowling_df['season'], errors='coerce')
        
        # Create recent form feature
        bowling_df = bowling_df.sort_values(['fullName', 'match_id'])
        bowling_df['recent_form'] = bowling_df.groupby('fullName')['Bowling_FP'].transform(
            lambda x: x.shift(1).rolling(window=5, min_periods=1).mean()
        )
        bowling_df['recent_form'] = bowling_df['recent_form'].fillna(0)
        
        # Split data by year
        train_bowling = bowling_df[bowling_df['season'] < year_split]
        test_bowling = bowling_df[bowling_df['season'] >= year_split]
        
        # Define features
        features = [
            'overs', 'maidens', 'conceded', 'wickets', 'economyRate', 'wides', 'noballs',
            'dot_ball_percentage', 'wicket_rate', 'recent_form', 'bowling_innings'
        ]
        
        # Ensure all features exist
        for feature in features:
            if feature not in train_bowling.columns:
                print(f"Warning: Feature '{feature}' not found in training data. Adding with zeros.")
                train_bowling[feature] = 0
                test_bowling[feature] = 0
        
        return train_bowling[features], train_bowling['Bowling_FP'], test_bowling[features], test_bowling['Bowling_FP']
        
    def _clean_features(self, df):
        """Clean DataFrame by removing NaNs, infs, and extreme values"""
        df = df.copy()
        
        # Replace inf with NaN
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        # Fill remaining NaNs with 0
        df.fillna(0, inplace=True)
        
        # Handle extreme values
        for col in df.columns:
            if df[col].dtype.name == 'category':
                continue
                
            upper_limit = df[col].quantile(0.999)
            if upper_limit > 1e6:
                df.loc[df[col] > upper_limit, col] = upper_limit
                
            lower_limit = df[col].quantile(0.001)
            if lower_limit < -1e6:
                df.loc[df[col] < lower_limit, col] = lower_limit
        
        return df 