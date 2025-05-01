import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import joblib
from datetime import datetime

# Import local modules
from utils import (
    load_data, filter_match_data, get_player_stats, get_player_historical_data,
    get_venue_performance, get_opponent_performance
)
from preprocessing import (
    preprocess_batting, preprocess_bowling, preprocess_fielding, get_player_role
)
from training import (
    train_model, evaluate_model, get_feature_importance
)
from scoring import (
    predict_fantasy_points, assign_roles, apply_contextual_factors,
    build_role_constrained_team, select_captain_vice_captain, print_team_details,
    visualize_team_composition
)

class Dream11Predictor:
    """Dream11 Team Prediction System"""
    
    def __init__(self, model_dir="models"):
        """Initialize the Dream11 predictor"""
        self.model_dir = model_dir
        self.bat_model = None
        self.bowl_model = None
        self.bat_features = None
        self.bowl_features = None
        self.data_loaded = False
        self.models_trained = False
        
        # Create model directory if it doesn't exist
        os.makedirs(model_dir, exist_ok=True)

    def load_data(self):
        """Load the datasets"""
        print("Loading datasets...")
        self.batting, self.bowling, self.fielding, self.fantasy, self.match, self.players = load_data()
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
        
        # Get all current players for both teams
        home_players = self.get_team_players(home_team)
        away_players = self.get_team_players(away_team)
        all_players = home_players + away_players
        
        # Filter data for matches involving these teams
        team_batting, team_bowling, team_fielding = filter_match_data(
            self.batting, self.bowling, self.fielding, home_team, away_team
        )
        
        # Set match-specific data
        self.match_batting = team_batting
        self.match_bowling = team_bowling
        self.match_fielding = team_fielding
        self.all_players = all_players
        self.home_team = home_team
        self.away_team = away_team
        
        # Venue information
        # Get most common venue for these teams
        venues = self.match[
            ((self.match['home_team'] == home_team) & (self.match['away_team'] == away_team)) |
            ((self.match['home_team'] == away_team) & (self.match['away_team'] == home_team))
        ]['venue'].value_counts()
        
        self.venue = venues.index[0] if not venues.empty else None
        
        print(f"Prepared data for {len(all_players)} players")
        print(f"Selected venue: {self.venue}")
        
        return all_players, team_batting, team_bowling, team_fielding

    def train_models(self, hyperparameter_tuning=True):
        """Train prediction models with enhanced features"""
        if not self.data_loaded:
            self.load_data()
        
        print("Preprocessing batting data...")
        X_bat_train, y_bat_train, X_bat_test, y_bat_test, train_batting, test_batting = preprocess_batting(self.batting)
        
        print("Preprocessing bowling data...")
        X_bowl_train, y_bowl_train, X_bowl_test, y_bowl_test, train_bowling, test_bowling = preprocess_bowling(self.bowling)
        
        # Save feature lists for prediction
        self.bat_features = X_bat_train.columns.tolist()
        self.bowl_features = X_bowl_train.columns.tolist()
        
        # Train batting model
        print("Training batting model...")
        bat_model = train_model(X_bat_train, y_bat_train, model_type='gradient_boosting', 
                                hyperparameter_tuning=False)
        
        # Train bowling model
        print("Training bowling model...")
        bowl_model = train_model(X_bowl_train, y_bowl_train, model_type='gradient_boosting',
                                hyperparameter_tuning=False)
        
        # Evaluate models
        print("\nEvaluating batting model:")
        bat_rmse, bat_r2, bat_pred = evaluate_model(bat_model, X_bat_test, y_bat_test)
        
        print("\nEvaluating bowling model:")
        bowl_rmse, bowl_r2, bowl_pred = evaluate_model(bowl_model, X_bowl_test, y_bowl_test)
        
        # Get feature importance
        bat_importance = get_feature_importance(bat_model, self.bat_features)
        bowl_importance = get_feature_importance(bowl_model, self.bowl_features)
        
        if bat_importance is not None:
            print("\nBatting feature importance:")
            print(bat_importance.head(10))
            
        if bowl_importance is not None:
            print("\nBowling feature importance:")
            print(bowl_importance.head(10))
        
        # Save models
        self.bat_model = bat_model
        self.bowl_model = bowl_model
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        joblib.dump(bat_model, os.path.join(self.model_dir, f"bat_model_{timestamp}.pkl"))
        joblib.dump(bowl_model, os.path.join(self.model_dir, f"bowl_model_{timestamp}.pkl"))
        
        print(f"Models saved to {self.model_dir}")
        self.models_trained = True
        
        # Save training results
        self.train_results = {
            'bat_rmse': bat_rmse,
            'bat_r2': bat_r2,
            'bowl_rmse': bowl_rmse,
            'bowl_r2': bowl_r2,
            'bat_importance': bat_importance,
            'bowl_importance': bowl_importance,
            'train_batting': train_batting,
            'test_batting': test_batting,
            'train_bowling': train_bowling,
            'test_bowling': test_bowling
        }
        
        return bat_model, bowl_model

    def load_models(self, bat_model_path=None, bowl_model_path=None):
        """Load pre-trained models"""
        if bat_model_path is None or bowl_model_path is None:
            # Find latest models
            bat_models = [f for f in os.listdir(self.model_dir) if f.startswith('bat_model_')]
            bowl_models = [f for f in os.listdir(self.model_dir) if f.startswith('bowl_model_')]
            
            if not bat_models or not bowl_models:
                raise FileNotFoundError("No pre-trained models found. Please train models first.")
            
            bat_model_path = os.path.join(self.model_dir, sorted(bat_models)[-1])
            bowl_model_path = os.path.join(self.model_dir, sorted(bowl_models)[-1])
        
        print(f"Loading models from {bat_model_path} and {bowl_model_path}")
        self.bat_model = joblib.load(bat_model_path)
        self.bowl_model = joblib.load(bowl_model_path)
        self.models_trained = True
        
        # Get feature names for prediction
        if hasattr(self.bat_model, 'feature_names_in_'):
            self.bat_features = self.bat_model.feature_names_in_.tolist()
        if hasattr(self.bowl_model, 'feature_names_in_'):
            self.bowl_features = self.bowl_model.feature_names_in_.tolist()
            
        return self.bat_model, self.bowl_model

    def predict_team(self, home_team, away_team, venue=None, load_saved_models=False):
        """Predict optimal team for a match"""
        if not self.data_loaded:
            self.load_data()
            
        # Prepare match data
        all_players, team_batting, team_bowling, team_fielding = self.prepare_match_data(home_team, away_team)
        
        # Set venue if provided
        if venue:
            self.venue = venue
        
        # Train or load models
        if not self.models_trained:
            if load_saved_models:
                try:
                    self.load_models()
                except FileNotFoundError:
                    print("No saved models found. Training new models...")
                    self.train_models()
            else:
                self.train_models()
        
        print(f"\nPredicting team for {home_team} vs {away_team} at {self.venue}...")
        
        # Re-preprocess data to match features used in training
        X_bat, _, _, _, _, preprocessed_batting = preprocess_batting(self.match_batting)
        X_bowl, _, _, _, _, preprocessed_bowling = preprocess_bowling(self.match_bowling)

        predicted_points = predict_fantasy_points(
            self.bat_model, self.bowl_model,
            X_bat, X_bowl, self.match_fielding,
            self.bat_features, self.bowl_features,
            all_players,
            batting_df=preprocessed_batting,
            bowling_df=preprocessed_bowling
        )
        
        # Assign roles to players
        predicted_points = assign_roles(predicted_points, self.batting, self.bowling, self.fielding)
        
        # Apply contextual factors (form, venue, opponent)
        adjusted_points = apply_contextual_factors(
            predicted_points, self.batting, self.bowling, self.venue, home_team, away_team
        )
        
        # Build optimal team
        optimal_team = build_role_constrained_team(
            adjusted_points, self.venue, home_team, away_team,
            max_players=11, max_credits=100, max_per_team=7,
            min_roles={"WK": 1, "BAT": 3, "AR": 1, "BWL": 3},
            max_roles={"WK": 4, "BAT": 6, "AR": 4, "BWL": 6}
        )
        
        # Select captain and vice-captain
        final_team = select_captain_vice_captain(optimal_team, self.batting, self.bowling)
        
        # Visualize team composition
        visualize_team_composition(final_team)
        
        # Print team details
        print_team_details(final_team)
        
        return final_team
    
    def analyze_player(self, player_name):
        """Analyze performance of a specific player"""
        if not self.data_loaded:
            self.load_data()
            
        print(f"\nAnalyzing player: {player_name}")
        
        # Get player data
        player_batting, player_bowling, player_fielding = get_player_historical_data(
            self.batting, self.bowling, self.fielding, player_name
        )
        
        # Player role
        role = get_player_role(self.batting, self.bowling, self.fielding, player_name)
        print(f"Role: {role}")
        
        # Overall stats
        batting_matches = len(player_batting) if not player_batting.empty else 0
        bowling_matches = len(player_bowling) if not player_bowling.empty else 0
        matches_played = max(batting_matches, bowling_matches)
        
        print(f"Matches played: {matches_played}")
        
        if not player_batting.empty:
            avg_runs = player_batting['runs'].mean()
            avg_sr = player_batting['strike_rate'].mean()
            avg_bat_fp = player_batting['Batting_FP'].mean()
            print(f"Batting: Avg Runs: {avg_runs:.2f}, Avg SR: {avg_sr:.2f}, Avg FP: {avg_bat_fp:.2f}")
            
        if not player_bowling.empty:
            avg_wickets = player_bowling['wickets'].mean()
            avg_economy = player_bowling['economyRate'].mean()
            avg_bowl_fp = player_bowling['Bowling_FP'].mean()
            print(f"Bowling: Avg Wickets: {avg_wickets:.2f}, Avg Economy: {avg_economy:.2f}, Avg FP: {avg_bowl_fp:.2f}")
        
        # Venue performance analysis if venue is set
        if hasattr(self, 'venue') and self.venue:
            bat_venue, bowl_venue = get_venue_performance(
                self.batting, self.bowling, player_name, self.venue
            )
            
            print(f"\nPerformance at {self.venue}:")
            if bat_venue['matches'] > 0:
                print(f"Batting: {bat_venue['matches']} matches, Avg Runs: {bat_venue['avg_runs']:.2f}, Avg SR: {bat_venue['avg_sr']:.2f}")
            if bowl_venue['matches'] > 0:
                print(f"Bowling: {bowl_venue['matches']} matches, Avg Wickets: {bowl_venue['avg_wickets']:.2f}, Avg Economy: {bowl_venue['avg_economy']:.2f}")
        
        # Opponent analysis if teams are set
        if hasattr(self, 'home_team') and hasattr(self, 'away_team'):
            for opponent in [self.home_team, self.away_team]:
                bat_opp, bowl_opp = get_opponent_performance(
                    self.batting, self.bowling, player_name, opponent
                )
                
                print(f"\nPerformance against {opponent}:")
                if bat_opp['matches'] > 0:
                    print(f"Batting: {bat_opp['matches']} matches, Avg Runs: {bat_opp['avg_runs']:.2f}, Avg SR: {bat_opp['avg_sr']:.2f}")
                if bowl_opp['matches'] > 0:
                    print(f"Bowling: {bowl_opp['matches']} matches, Avg Wickets: {bowl_opp['avg_wickets']:.2f}, Avg Economy: {bowl_opp['avg_economy']:.2f}")
        
        return player_batting, player_bowling, player_fielding

def main():
    """Main function to run the Dream11 prediction system"""
    print("=" * 80)
    print(" " * 25 + "DREAM11 TEAM PREDICTOR")
    print("=" * 80)
    
    predictor = Dream11Predictor()
    predictor.load_data()
    
    # Get user input for teams
    print("\nAvailable teams:")
    teams = sorted(predictor.players['team'].unique())
    for i, team in enumerate(teams):
        print(f"{i+1}. {team}")
    
    try:
        home_team_idx = int(input("\nSelect home team (number): ")) - 1
        away_team_idx = int(input("Select away team (number): ")) - 1
        
        home_team = teams[home_team_idx]
        away_team = teams[away_team_idx]
        
        print(f"\nSelected match: {home_team} vs {away_team}")
        
        # Ask for venue
        venues = predictor.match['venue'].unique()
        print("\nCommon venues:")
        for i, venue in enumerate(venues[:10]):  # Show top 10 venues
            print(f"{i+1}. {venue}")
        
        use_venue = input("\nSpecify venue? (y/n): ").lower() == 'y'
        venue = None
        
        if use_venue:
            venue_idx = int(input("Select venue (number): ")) - 1
            venue = venues[venue_idx]
            print(f"Selected venue: {venue}")
        
        # Ask for model training option
        train_new = input("\nTrain new models? (y/n, default: y): ").lower() != 'n'
        
        # Predict team
        final_team = predictor.predict_team(home_team, away_team, venue, not train_new)
        
        # Option to analyze specific players
        analyze_player = input("\nAnalyze specific player? (y/n): ").lower() == 'y'
        while analyze_player:
            player_name = input("Enter player name: ")
            predictor.analyze_player(player_name)
            analyze_player = input("\nAnalyze another player? (y/n): ").lower() == 'y'
        
        print("\nThank you for using Dream11 Team Predictor!")
        
    except (IndexError, ValueError) as e:
        print(f"Error: {e}")
        print("Please try again with valid inputs.")

if __name__ == "__main__":
    main()