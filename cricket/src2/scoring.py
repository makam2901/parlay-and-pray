import numpy as np
import pandas as pd
import os
from collections import defaultdict
import matplotlib.pyplot as plt
"""
    Predict fantasy points for players
    
    Args:
        model_bat: Trained batting model
        model_bowl: Trained bowling model
        batting_data: Preprocessed batting data
        bowling_data: Preprocessed bowling data
        fielding_data: Preprocessed fielding data
        bat_features: Batting features list
        bowl_features: Bowling features list
        players_list: List of active players (optional)
        
    Returns:
        DataFrame with predicted fantasy points
    """
def predict_fantasy_points(model_bat, model_bowl, batting_data, bowling_data, fielding_data, 
                           bat_features, bowl_features, players_list=None, batting_df=None, bowling_df=None):

    # Filter original DataFrames (which contain fullName) by players
    if batting_df is not None and players_list is not None:
        batting = batting_df[batting_df['fullName'].isin(players_list)].copy()
    else:
        batting = batting_data.copy()

    if bowling_df is not None and players_list is not None:
        bowling = bowling_df[bowling_df['fullName'].isin(players_list)].copy()
    else:
        bowling = bowling_data.copy()

    fielding = fielding_data.copy() if fielding_data is not None else None

    if players_list is not None and fielding is not None and 'fullName' in fielding.columns:
        fielding = fielding[fielding['fullName'].isin(players_list)]
    
    # Fill missing values with zeros
    batting_features = batting[bat_features].fillna(0)
    bowling_features = bowling[bowl_features].fillna(0)
    
    # Predict fantasy points
    batting['Predicted_FP'] = model_bat.predict(batting_features)
    bowling['Predicted_Bowling_FP'] = model_bowl.predict(bowling_features)
    
    # Merge batting and bowling predictions
    df_combined = pd.merge(
        batting[['match_id', 'fullName', 'batting_team', 'bowling_team', 'Predicted_FP', 'credit']],
        bowling[['match_id', 'fullName', 'Predicted_Bowling_FP']],
        on=['match_id', 'fullName'], how='outer'
    )
    
    # Fill missing values
    df_combined['Predicted_FP'] = df_combined['Predicted_FP'].fillna(0)
    df_combined['Predicted_Bowling_FP'] = df_combined['Predicted_Bowling_FP'].fillna(0)
    df_combined['Total_FP'] = df_combined['Predicted_FP'] + df_combined['Predicted_Bowling_FP']
    
    # Generate random credits for players missing credit info
    missing_credit = df_combined['credit'].isna()
    df_combined.loc[missing_credit, 'credit'] = np.random.uniform(7.0, 10.0, size=missing_credit.sum()).round(1)
    
    # Handle missing team info
    for col in ['batting_team', 'bowling_team']:
        if df_combined[col].isna().any():
            # Add 'Unknown' category if needed
            if pd.api.types.is_categorical_dtype(df_combined[col]):
                df_combined[col] = df_combined[col].cat.add_categories(['Unknown'])
    
    df_combined['batting_team'] = df_combined['batting_team'].fillna(df_combined['bowling_team'])
    df_combined['bowling_team'] = df_combined['bowling_team'].fillna(df_combined['batting_team'])
    df_combined['batting_team'] = df_combined['batting_team'].fillna('Unknown')
    df_combined['bowling_team'] = df_combined['bowling_team'].fillna('Unknown')
    
    # Remove rows with unknown teams
    df_combined = df_combined[df_combined['batting_team'] != 'Unknown']
    
    return df_combined

def assign_roles(df, batting_df, bowling_df, fielding_df=None):
    """
    Assign roles to players based on their historical performance
    
    Args:
        df: DataFrame with player predictions
        batting_df: Original batting data
        bowling_df: Original bowling data
        fielding_df: Original fielding data (optional)
        
    Returns:
        DataFrame with assigned roles
    """
    unique_players = df['fullName'].unique()
    roles = {}
    
    for player in unique_players:
        # Get player data
        player_batting = batting_df[batting_df['fullName'] == player]
        player_bowling = bowling_df[bowling_df['fullName'] == player]
        
        # Check if player is a wicketkeeper
        is_wk = False
        if fielding_df is not None:
            player_fielding = fielding_df[fielding_df['fullName'] == player]
            if not player_fielding.empty and 'stumping_FP' in player_fielding.columns:
                is_wk = player_fielding['stumping_FP'].sum() > 0
        
        # Calculate average performance
        avg_batting_fp = player_batting['Batting_FP'].mean() if not player_batting.empty else 0
        avg_bowling_fp = player_bowling['Bowling_FP'].mean() if not player_bowling.empty else 0
        
        # Assign role based on performance
        if is_wk:
            roles[player] = 'WK'
        elif avg_batting_fp >= 20 and avg_bowling_fp >= 20:
            roles[player] = 'AR'  # All-rounder
        elif avg_bowling_fp >= 20:
            roles[player] = 'BWL'  # Bowler
        else:
            roles[player] = 'BAT'  # Batsman
    
    # Add roles to DataFrame
    df['role'] = df['fullName'].map(roles)
    
    # Ensure all players have a role
    df['role'] = df['role'].fillna('BAT')
    
    return df

def calculate_player_form(df, player_name, recent_matches=5):
    """Calculate player form based on recent performance"""
    player_data = df[df['fullName'] == player_name].sort_values('match_id', ascending=False)
    
    if len(player_data) < 2:
        return 1.0  # Default multiplier if insufficient data
    
    # Use only recent matches
    recent_data = player_data.head(recent_matches)
    
    # Calculate average fantasy points
    avg_fp = recent_data['Total_FP'].mean()
    
    # Calculate form multiplier (normalized to range 0.8-1.2)
    all_player_avg = df.groupby('fullName')['Total_FP'].mean().mean()
    form_multiplier = 1.0 + (avg_fp - all_player_avg) / (all_player_avg * 2)
    
    # Clamp to reasonable range
    form_multiplier = max(0.8, min(1.2, form_multiplier))
    
    return form_multiplier

def calculate_venue_factor(batting_df, bowling_df, player_name, venue):
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

def calculate_opponent_factor(batting_df, bowling_df, player_name, opponent):
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

def apply_contextual_factors(df, batting_df, bowling_df, venue, home_team, away_team):
    """Apply contextual factors to predicted points"""
    df = df.copy()
    
    # Calculate and apply form, venue and opponent factors
    for i, row in df.iterrows():
        player = row['fullName']
        team = row['batting_team']
        opponent = home_team if team == away_team else away_team
        
        # Calculate factors
        form_factor = calculate_player_form(df, player)
        venue_factor = calculate_venue_factor(batting_df, bowling_df, player, venue)
        opponent_factor = calculate_opponent_factor(batting_df, bowling_df, player, opponent)
        
        # Apply factors
        df.loc[i, 'form_factor'] = form_factor
        df.loc[i, 'venue_factor'] = venue_factor
        df.loc[i, 'opponent_factor'] = opponent_factor
        
        # Adjust predicted points
        combined_factor = form_factor * venue_factor * opponent_factor
        df.loc[i, 'Adjusted_Total_FP'] = row['Total_FP'] * combined_factor
    
    return df

def build_role_constrained_team(
    df, match_venue, home_team, away_team,
    max_players=11, max_credits=100, max_per_team=7,
    min_roles={"WK": 1, "BAT": 2, "AR": 0, "BWL": 2},
    max_roles={"WK": 1, "BAT": 5, "AR": 2, "BWL": 5}):
    """
    Build optimal team with role constraints
    
    Args:
        df: DataFrame with player predictions
        match_venue: Venue for the match
        home_team: Home team name
        away_team: Away team name
        max_players: Maximum number of players (default: 11)
        max_credits: Maximum credits (default: 100)
        max_per_team: Maximum players from same team (default: 7)
        min_roles: Minimum players per role
        max_roles: Maximum players per role
    
    Returns:
        DataFrame with optimal team
    """
    team = []
    team_credits = 0
    team_counts = {home_team: 0, away_team: 0}
    role_counts = {'WK': 0, 'BAT': 0, 'AR': 0, 'BWL': 0}
    selected_players = set()
    
    # Filter out players with missing team info
    df = df[~df['batting_team'].isin(['Unknown'])]
    
    # Sort by adjusted fantasy points (best players first)
    df_sorted = df.sort_values(by='Adjusted_Total_FP', ascending=False)
    
    # First, select mandatory roles
    for role, min_count in min_roles.items():
        role_df = df_sorted[df_sorted['role'] == role]
        selected = 0
        
        for _, player in role_df.iterrows():
            if selected >= min_count:
                break
                
            name = player['fullName']
            credit = player['credit']
            team_name = player['batting_team']
            fp = player['Adjusted_Total_FP']
            
            if name in selected_players:
                continue
                
            if team_counts.get(team_name, 0) >= max_per_team:
                continue
                
            if team_credits + credit > max_credits:
                continue
                
            # Add player to team
            team.append({
                'name': name, 
                'team': team_name, 
                'role': role, 
                'credit': credit, 
                'total_fp': fp
            })
            team_credits += credit
            team_counts[team_name] = team_counts.get(team_name, 0) + 1
            role_counts[role] += 1
            selected_players.add(name)
            selected += 1
    
    # Then fill remaining slots with best available players
    while len(team) < max_players:
        best_player = None
        best_value = -1  # Value = fantasy points / credit
        
        for _, player in df_sorted.iterrows():
            name = player['fullName']
            credit = player['credit']
            role = player['role']
            team_name = player['batting_team']
            fp = player['Adjusted_Total_FP']
            
            # Skip if already selected
            if name in selected_players:
                continue
                
            # Skip if exceeds team constraints
            if team_counts.get(team_name, 0) >= max_per_team:
                continue
                
            # Skip if exceeds role constraints
            if role_counts[role] >= max_roles[role]:
                continue
                
            # Skip if exceeds credit constraint
            if team_credits + credit > max_credits:
                continue
                
            # Calculate value (points per credit)
            value = fp / credit if credit > 0 else 0
            
            if value > best_value:
                best_value = value
                best_player = {
                    'name': name,
                    'team': team_name,
                    'role': role,
                    'credit': credit,
                    'total_fp': fp
                }
        
        if best_player:
            team.append(best_player)
            team_credits += best_player['credit']
            team_counts[best_player['team']] = team_counts.get(best_player['team'], 0) + 1
            role_counts[best_player['role']] += 1
            selected_players.add(best_player['name'])
        else:
            # If no valid player found, break to avoid infinite loop
            break
    
    # Create team DataFrame
    team_df = pd.DataFrame(team)
    
    # Check if team is valid
    if len(team_df) < max_players:
        print(f"⚠️ Warning: Could only select {len(team_df)} players under constraints")
        print(f"Role distribution: {role_counts}")
        
    return team_df

def select_captain_vice_captain(team_df, batting_df, bowling_df):
    """
    Select captain and vice-captain based on form and matchup
    
    Args:
        team_df: DataFrame with selected team
        batting_df: Original batting data
        bowling_df: Original bowling data
        
    Returns:
        DataFrame with captain and vice-captain selected
    """
    team_df = team_df.copy()
    
    # Calculate captain score (combination of total points and consistency)
    captain_scores = []
    
    for _, player in team_df.iterrows():
        name = player['name']
        
        # Get player history
        player_batting = batting_df[batting_df['fullName'] == name]
        player_bowling = bowling_df[bowling_df['fullName'] == name]
        
        # Calculate consistency (inverse of coefficient of variation)
        batting_consistency = 1.0
        if not player_batting.empty and len(player_batting) > 2:
            batting_mean = player_batting['Batting_FP'].mean()
            batting_std = player_batting['Batting_FP'].std()
            if batting_mean > 0:
                batting_consistency = 1 / (batting_std / batting_mean) if batting_std > 0 else 1.0
        
        bowling_consistency = 1.0
        if not player_bowling.empty and len(player_bowling) > 2:
            bowling_mean = player_bowling['Bowling_FP'].mean()
            bowling_std = player_bowling['Bowling_FP'].std()
            if bowling_mean > 0:
                bowling_consistency = 1 / (bowling_std / bowling_mean) if bowling_std > 0 else 1.0
        
        # Overall consistency
        consistency = (batting_consistency + bowling_consistency) / 2
        
        # Captain score = points * consistency
        captain_score = player['total_fp'] * consistency
        
        captain_scores.append({
            'name': name,
            'total_fp': player['total_fp'],
            'consistency': consistency,
            'captain_score': captain_score
        })
    
    # Sort by captain score
    captain_scores = sorted(captain_scores, key=lambda x: x['captain_score'], reverse=True)
    
    # Assign captain and vice-captain
    if len(captain_scores) >= 2:
        captain = captain_scores[0]['name']
        vice_captain = captain_scores[1]['name']
        
        # Update team_df with captain and vice-captain information
        team_df['is_captain'] = team_df['name'] == captain
        team_df['is_vice_captain'] = team_df['name'] == vice_captain
        
        # Apply points multipliers
        team_df['adjusted_fp'] = team_df['total_fp'].copy()
        team_df.loc[team_df['is_captain'], 'adjusted_fp'] *= 2.0  # Captain gets 2x points
        team_df.loc[team_df['is_vice_captain'], 'adjusted_fp'] *= 1.5  # Vice-captain gets 1.5x points
    
    return team_df

def visualize_team_composition(team_df):
    """Visualize team composition by role and team"""
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot role distribution
    role_counts = team_df['role'].value_counts()
    ax1.bar(role_counts.index, role_counts.values, color=['blue', 'green', 'orange', 'red'])
    ax1.set_title('Team Composition by Role')
    ax1.set_ylabel('Number of Players')
    
    # Plot team distribution
    team_counts = team_df['team'].value_counts()
    ax2.bar(team_counts.index, team_counts.values, color=['purple', 'teal'])
    ax2.set_title('Team Composition by Team')
    ax2.set_ylabel('Number of Players')
    
    plt.tight_layout()
    plt.savefig('team_composition.png')
    plt.close()
    
    return fig

def print_team_details(team_df):
    """Print detailed information about the selected team"""
    team_df = team_df.sort_values(by=['is_captain', 'is_vice_captain', 'total_fp'], ascending=False)
    
    total_credits = team_df['credit'].sum()
    total_adjusted_fp = team_df['adjusted_fp'].sum()
    
    print("\n" + "="*60)
    print(f"🏏 DREAM11 FANTASY TEAM PREDICTION")
    print("="*60)
    
    for i, player in team_df.iterrows():
        captain_tag = " (C)" if player.get('is_captain', False) else ""
        vc_tag = " (VC)" if player.get('is_vice_captain', False) else ""
        role_emoji = {
            'WK': '🧤',
            'BAT': '🏏',
            'BWL': '🎯',
            'AR': '⚡'
        }.get(player['role'], '')
        
        print(f"{i+1}. {role_emoji} {player['name']}{captain_tag}{vc_tag} - {player['team']} - {player['total_fp']:.1f} pts - ${player['credit']}")
    
    print("-"*60)
    print(f"Total Credits: {total_credits:.1f}/100")
    print(f"Total Points (with C/VC): {total_adjusted_fp:.1f}")
    print("="*60)
    
    # Print team statistics
    role_counts = team_df['role'].value_counts()
    team_counts = team_df['team'].value_counts()
    
    print("\nTeam Composition:")
    for role, count in role_counts.items():
        print(f"- {role}: {count} players")
    
    print("\nTeam Distribution:")
    for team, count in team_counts.items():
        print(f"- {team}: {count} players")
    
    return team_df
