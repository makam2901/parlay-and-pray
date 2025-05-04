import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

class FantasyScorer:
    """Handles fantasy points prediction and team building"""
    
    def __init__(self, preprocessor, model_trainer):
        """Initialize the fantasy scorer"""
        self.preprocessor = preprocessor
        self.model_trainer = model_trainer
        
    def predict_fantasy_points(self, batting_data, bowling_data, fielding_data, players_list=None):
        """Predict fantasy points for players with proper NaN handling"""
        # Get unique players for the current match
        current_players = pd.concat([
            batting_data[['fullName', 'player_team']],
            bowling_data[['fullName', 'player_team']]
        ]).drop_duplicates('fullName')
        
        print(f"\nPredicting fantasy points for {len(current_players)} current match players")
        
        # Create a base DataFrame with current players
        df_combined = pd.DataFrame({
            'fullName': current_players['fullName'],
            'player_team': current_players['player_team']
        })
        
        # Get historical data for each player
        for _, player in current_players.iterrows():
            name = player['fullName']
            team = player['player_team']
            
            # Get player's historical batting data
            player_batting = batting_data[batting_data['fullName'] == name].copy()
            if not player_batting.empty:
                # Sort by date (most recent first)
                player_batting = player_batting.sort_values('match_id', ascending=False)
                # Take most recent 5 matches
                recent_batting = player_batting.head(5).copy()
                
                # Ensure all necessary features exist
                for feature in self.model_trainer.bat_features:
                    if feature not in recent_batting.columns:
                        recent_batting.loc[:, feature] = 0
                
                # Clean features
                batting_features = self.preprocessor._clean_features(recent_batting[self.model_trainer.bat_features])
                # Predict batting points
                batting_fp = self.model_trainer.bat_model.predict(batting_features)
                # Use average of recent predictions
                df_combined.loc[df_combined['fullName'] == name, 'Predicted_FP'] = batting_fp.mean()
            
            # Get player's historical bowling data
            player_bowling = bowling_data[bowling_data['fullName'] == name].copy()
            if not player_bowling.empty:
                # Sort by date (most recent first)
                player_bowling = player_bowling.sort_values('match_id', ascending=False)
                # Take most recent 5 matches
                recent_bowling = player_bowling.head(5).copy()
                
                # Ensure all necessary features exist
                for feature in self.model_trainer.bowl_features:
                    if feature not in recent_bowling.columns:
                        recent_bowling.loc[:, feature] = 0
                
                # Clean features
                bowling_features = self.preprocessor._clean_features(recent_bowling[self.model_trainer.bowl_features])
                # Predict bowling points
                bowling_fp = self.model_trainer.bowl_model.predict(bowling_features)
                # Use average of recent predictions
                df_combined.loc[df_combined['fullName'] == name, 'Predicted_Bowling_FP'] = bowling_fp.mean()
        
        # Fill missing values
        df_combined.loc[:, 'Predicted_FP'] = df_combined['Predicted_FP'].fillna(0)
        df_combined.loc[:, 'Predicted_Bowling_FP'] = df_combined['Predicted_Bowling_FP'].fillna(0)
        df_combined.loc[:, 'Total_FP'] = df_combined['Predicted_FP'] + df_combined['Predicted_Bowling_FP']
        
        return df_combined
        
    def calculate_player_form(self, df, player_name, recent_matches=5):
        """Calculate player form based on recent performance"""
        # If we don't have match_id, just use the latest performance
        if 'match_id' not in df.columns:
            player_data = df[df['fullName'] == player_name]
            if len(player_data) < 1:
                return 1.0  # Default multiplier if no data
            
            # Use the latest performance
            avg_fp = player_data['Total_FP'].mean()
            all_player_avg = df['Total_FP'].mean()
            
            if all_player_avg == 0:  # Avoid division by zero
                form_multiplier = 1.0
            else:
                form_multiplier = 1.0 + (avg_fp - all_player_avg) / (all_player_avg * 2)
            
            # Clamp to reasonable range
            form_multiplier = max(0.8, min(1.2, form_multiplier))
            return form_multiplier
        
        # If we have match_id, use chronological data
        player_data = df[df['fullName'] == player_name].sort_values('match_id', ascending=False)
        
        if len(player_data) < 2:
            return 1.0  # Default multiplier if insufficient data
        
        # Use only recent matches
        recent_data = player_data.head(recent_matches)
        
        # Calculate average fantasy points
        avg_fp = recent_data['Total_FP'].mean()
        
        # Calculate form multiplier (normalized to range 0.8-1.2)
        all_player_avg = df.groupby('fullName')['Total_FP'].mean().mean()
        if all_player_avg == 0:  # Avoid division by zero
            form_multiplier = 1.0
        else:
            form_multiplier = 1.0 + (avg_fp - all_player_avg) / (all_player_avg * 2)
        
        # Clamp to reasonable range
        form_multiplier = max(0.8, min(1.2, form_multiplier))
        
        return form_multiplier
        
    def calculate_venue_factor(self, batting_df, bowling_df, player_name, venue):
        """Calculate venue-specific performance factor"""
        # Get player's performance at the venue
        venue_batting = batting_df[(batting_df['fullName'] == player_name) & (batting_df['venue'] == venue)]
        venue_bowling = bowling_df[(bowling_df['fullName'] == player_name) & (bowling_df['venue'] == venue)]
        
        # Get player's overall performance
        all_batting = batting_df[batting_df['fullName'] == player_name]
        all_bowling = bowling_df[bowling_df['fullName'] == player_name]
        
        # Calculate venue factor for batting
        bat_venue_factor = 1.0
        if not venue_batting.empty and not all_batting.empty:
            venue_avg = venue_batting['Batting_FP'].mean()
            overall_avg = all_batting['Batting_FP'].mean()
            if overall_avg > 0:
                bat_venue_factor = venue_avg / overall_avg
        
        # Calculate venue factor for bowling
        bowl_venue_factor = 1.0
        if not venue_bowling.empty and not all_bowling.empty:
            venue_avg = venue_bowling['Bowling_FP'].mean()
            overall_avg = all_bowling['Bowling_FP'].mean()
            if overall_avg > 0:
                bowl_venue_factor = venue_avg / overall_avg
        
        # Combine factors (with bounds)
        combined_factor = (bat_venue_factor + bowl_venue_factor) / 2
        combined_factor = max(0.8, min(1.2, combined_factor))
        
        return combined_factor
        
    def calculate_opponent_factor(self, batting_df, bowling_df, player_name, opponent):
        """Calculate opponent-specific performance factor"""
        # Get player's performance against this opponent
        opp_batting = batting_df[(batting_df['fullName'] == player_name) & 
                               ((batting_df['bowling_team'] == opponent) | 
                                (batting_df['batting_team'] == opponent))]
        
        opp_bowling = bowling_df[(bowling_df['fullName'] == player_name) & 
                               ((bowling_df['bowling_team'] == opponent) | 
                                (bowling_df['batting_team'] == opponent))]
        
        # Get player's overall performance
        all_batting = batting_df[batting_df['fullName'] == player_name]
        all_bowling = bowling_df[bowling_df['fullName'] == player_name]
        
        # Calculate opponent factor for batting
        bat_opp_factor = 1.0
        if not opp_batting.empty and not all_batting.empty:
            opp_avg = opp_batting['Batting_FP'].mean()
            overall_avg = all_batting['Batting_FP'].mean()
            if overall_avg > 0:
                bat_opp_factor = opp_avg / overall_avg
        
        # Calculate opponent factor for bowling
        bowl_opp_factor = 1.0
        if not opp_bowling.empty and not all_bowling.empty:
            opp_avg = opp_bowling['Bowling_FP'].mean()
            overall_avg = all_bowling['Bowling_FP'].mean()
            if overall_avg > 0:
                bowl_opp_factor = opp_avg / overall_avg
        
        # Combine factors (with bounds)
        combined_factor = (bat_opp_factor + bowl_opp_factor) / 2
        combined_factor = max(0.8, min(1.2, combined_factor))
        
        return combined_factor
        
    def apply_contextual_factors(self, df, batting_df, bowling_df, venue, home_team, away_team):
        """Apply contextual factors to predicted points"""
        df = df.copy()
        
        # Calculate and apply form, venue and opponent factors
        for i, row in df.iterrows():
            player = row['fullName']
            team = row['player_team']
            opponent = home_team if team == away_team else away_team
            
            # Calculate factors
            form_factor = self.calculate_player_form(df, player)
            venue_factor = self.calculate_venue_factor(batting_df, bowling_df, player, venue)
            opponent_factor = self.calculate_opponent_factor(batting_df, bowling_df, player, opponent)
            
            # Apply factors
            df.loc[i, 'form_factor'] = form_factor
            df.loc[i, 'venue_factor'] = venue_factor
            df.loc[i, 'opponent_factor'] = opponent_factor
            
            # Adjust predicted points
            combined_factor = form_factor * venue_factor * opponent_factor
            df.loc[i, 'Adjusted_Total_FP'] = row['Total_FP'] * combined_factor
        
        return df
        
    def build_role_constrained_team(self, df, players_df, match_venue, home_team, away_team,
                                  max_players=11, max_credits=100, max_per_team=7,
                                  min_roles={"WK": 1, "BAT": 2, "AR": 1, "BWL": 2},
                                  max_roles={"WK": 3, "BAT": 5, "AR": 3, "BWL": 5},
                                  user_roles=None):
        """Build optimal fantasy cricket team with role and credit constraints"""
        df = df.copy()
        
        # If user roles are specified, use them as both min and max
        if user_roles is not None:
            min_roles = user_roles.copy()
            max_roles = user_roles.copy()
        
        # Fix 'Unknown' team values using players_df
        if 'player_team' in df.columns:
            df.loc[df['player_team'] == 'Unknown', 'player_team'] = (
                df[df['player_team'] == 'Unknown']['fullName']
                .map(players_df.set_index('players')['team'])
            )
            df = df[~df['player_team'].isin(['Unknown'])]

        # Filter to relevant match teams
        df = df[df['player_team'].isin([home_team, away_team])]
        
        # Print available players by role
        print("\nAvailable players by role:")
        for role in min_roles.keys():
            role_players = df[df['role'] == role]
            print(f"{role}: {len(role_players)} players")
            print(f"  {home_team}: {len(role_players[role_players['player_team'] == home_team])}")
            print(f"  {away_team}: {len(role_players[role_players['player_team'] == away_team])}")
        
        # Initialize team state
        team = []
        team_credits = 0
        team_counts = {home_team: 0, away_team: 0}
        role_counts = {role: 0 for role in min_roles}
        selected_players = set()

        # Sort players by adjusted fantasy points
        df_sorted = df.sort_values(by='Adjusted_Total_FP', ascending=False)

        # 1. First, satisfy minimum role requirements
        for role, min_required in min_roles.items():
            role_df = df_sorted[df_sorted['role'] == role]

            for _, player in role_df.iterrows():
                if role_counts[role] >= min_required:
                    break

                name = player['fullName']
                credit = player['credit']
                team_name = player['player_team']
                fp = player['Adjusted_Total_FP']

                if name in selected_players:
                    continue
                if team_counts[team_name] >= max_per_team:
                    print(f"⚠️ Skipping {name} - team {team_name} already has {team_counts[team_name]} players")
                    continue
                if team_credits + credit > max_credits:
                    print(f"⚠️ Skipping {name} - would exceed credit limit ({team_credits + credit:.1f} > {max_credits})")
                    continue

                team.append({
                    'name': name,
                    'team': team_name,
                    'role': role,
                    'credit': credit,
                    'total_fp': fp
                })
                selected_players.add(name)
                team_counts[team_name] += 1
                role_counts[role] += 1
                team_credits += credit
                print(f"Added {name} ({role}) from {team_name} - Credits: {credit:.1f}, Points: {fp:.1f}")

        # 2. Fill remaining slots within all constraints
        df_remaining = df_sorted[~df_sorted['fullName'].isin(selected_players)]

        while len(team) < max_players:
            added = False
            for _, player in df_remaining.iterrows():
                name = player['fullName']
                credit = player['credit']
                role = player['role']
                team_name = player['player_team']
                fp = player['Adjusted_Total_FP']

                if name in selected_players:
                    continue
                if team_counts[team_name] >= max_per_team:
                    print(f"⚠️ Skipping {name} - team {team_name} already has {team_counts[team_name]} players")
                    continue
                if role_counts[role] >= max_roles[role]:
                    print(f"⚠️ Skipping {name} - already have {role_counts[role]} {role}s")
                    continue
                if team_credits + credit > max_credits:
                    print(f"⚠️ Skipping {name} - would exceed credit limit ({team_credits + credit:.1f} > {max_credits})")
                    continue

                team.append({
                    'name': name,
                    'team': team_name,
                    'role': role,
                    'credit': credit,
                    'total_fp': fp
                })
                selected_players.add(name)
                team_counts[team_name] += 1
                role_counts[role] += 1
                team_credits += credit
                added = True
                print(f"Added {name} ({role}) from {team_name} - Credits: {credit:.1f}, Points: {fp:.1f}")
                break

            if not added:
                print("\n⚠️ Couldn't fill all 11 slots under constraints. Current state:")
                print(f"Team size: {len(team)}")
                print(f"Total credits: {team_credits:.1f}")
                print(f"Players per team: {team_counts}")
                print(f"Role counts: {role_counts}")
                break

        team_df = pd.DataFrame(team)
        
        # Verify minimum requirements are met
        for role, min_required in min_roles.items():
            actual_count = len(team_df[team_df['role'] == role])
            if actual_count < min_required:
                print(f"⚠️ Warning: Could not meet minimum requirement of {min_required} {role}s. Only {actual_count} selected.")

        if len(team_df) < max_players:
            print(f"⚠️ Warning: Could only select {len(team_df)} players out of {max_players} due to constraints.")
            
        return team_df
        
    def calculate_team_stats(self, team_df):
        """Calculate team statistics including total credits, total points, and role distribution."""
        stats = {
            'total_credits': team_df['credit'].sum(),
            'total_points': team_df['total_fp'].sum(),
            'role_distribution': team_df['role'].value_counts().to_dict()
        }
        return stats

    def select_captain_vice_captain(self, team_df):
        """Select captain and vice-captain based on predicted fantasy points."""
        # Sort by total fantasy points
        sorted_team = team_df.sort_values('total_fp', ascending=False)
        
        # Select top two players as captain and vice-captain
        captain = sorted_team.iloc[0]['name']
        vice_captain = sorted_team.iloc[1]['name']
        
        return captain, vice_captain
        
    def visualize_team(self, team_df, captain, vice_captain, home_team, away_team):
        """Visualize the selected Dream11 team"""
        # Create a figure
        plt.figure(figsize=(12, 8))
        
        # Title
        plt.suptitle(f"Dream11 Team: {home_team} vs {away_team}", fontsize=16, fontweight='bold')
        
        # Group players by role
        role_order = ['WK', 'BAT', 'AR', 'BWL']
        roles = {}
        for role in role_order:
            roles[role] = team_df[team_df['role'] == role]
        
        # Set up grid for visualizing
        rows = len(role_order)
        
        for i, role in enumerate(role_order):
            role_players = roles[role]
            n_players = len(role_players)
            
            # Create subplot for this role
            plt.subplot(rows, 1, i+1)
            plt.title(f"{role} ({n_players})", fontweight='bold')
            
            # Remove axes
            plt.axis('off')
            
            # Position players evenly
            positions = np.linspace(0.1, 0.9, n_players)
            
            for j, (_, player) in enumerate(role_players.iterrows()):
                name = player['name']
                team_name = player['team']
                credit = player['credit']
                points = player['total_fp']
                
                # Add captain/vc markers
                if name == captain:
                    name = f"{name} (C)"
                elif name == vice_captain:
                    name = f"{name} (VC)"
                
                # Color code by team
                color = 'blue' if team_name == home_team else 'red'
                
                # Draw player
                plt.text(positions[j], 0.5, name, 
                        ha='center', va='center', 
                        bbox=dict(facecolor=color, alpha=0.3),
                        fontweight='bold' if name in [captain, vice_captain] else 'normal')
                
                # Add credit and points below
                plt.text(positions[j], 0.3, f"₹{credit:.1f}", ha='center', va='center', fontsize=8)
                plt.text(positions[j], 0.2, f"Pts: {points:.1f}", ha='center', va='center', fontsize=8)
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(f"dream11_team_{home_team}_vs_{away_team}.png", dpi=300)
        plt.show()
        
    def export_team(self, team_df, captain, vice_captain, home_team, away_team, output_format='csv'):
        """Export the Dream11 team to a file"""
        # Add captain and vice-captain info
        team_df = team_df.copy()
        team_df['captain'] = team_df['name'] == captain
        team_df['vice_captain'] = team_df['name'] == vice_captain
        
        # Calculate effective points (1.5x for captain, 1.25x for vice_captain)
        team_df['effective_points'] = team_df.apply(
            lambda x: x['total_fp'] * 1.5 if x['captain'] else (x['total_fp'] * 1.25 if x['vice_captain'] else x['total_fp']),
            axis=1
        )
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"dream11_team_{home_team}_vs_{away_team}_{timestamp}"
        
        if output_format == 'csv':
            team_df.to_csv(f"{filename}.csv", index=False)
            print(f"Team exported to {filename}.csv")
        elif output_format == 'json':
            team_df.to_json(f"{filename}.json", orient='records')
            print(f"Team exported to {filename}.json")
        elif output_format == 'excel':
            team_df.to_excel(f"{filename}.xlsx", index=False)
            print(f"Team exported to {filename}.xlsx")
        
        return filename 