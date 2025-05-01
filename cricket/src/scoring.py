import numpy as np
import pandas as pd
import os

def merge_predictions(batting, bowling, model_bat, model_bowl, features_bat, features_bowl):
    batting = batting.copy()
    bowling = bowling.copy()
    batting['Predicted_FP'] = model_bat.predict(batting[features_bat])
    bowling['Predicted_Bowling_FP'] = model_bowl.predict(bowling[features_bowl])
    
    df_combined = pd.merge(
        batting,
        bowling[['match_id', 'fullName', 'Predicted_Bowling_FP']],
        on=['match_id', 'fullName'], how='outer'
    )
    
    df_combined['Predicted_FP'] = df_combined['Predicted_FP'].fillna(0)
    df_combined['Predicted_Bowling_FP'] = df_combined['Predicted_Bowling_FP'].fillna(0)
    df_combined['Total_FP'] = df_combined['Predicted_FP'] + df_combined['Predicted_Bowling_FP']
    df_combined['credit'] = np.random.uniform(7.0, 10.0, size=len(df_combined)).round(1)

    # Add 'Unknown' category if needed
    for col in ['batting_team', 'bowling_team']:
        if pd.api.types.is_categorical_dtype(df_combined[col]):
            df_combined[col] = df_combined[col].cat.add_categories(['Unknown'])

    df_combined['batting_team'] = df_combined['batting_team'].fillna(df_combined['bowling_team'])
    df_combined['bowling_team'] = df_combined['bowling_team'].fillna(df_combined['batting_team'])
    df_combined['batting_team'] = df_combined['batting_team'].fillna('Unknown')
    df_combined['bowling_team'] = df_combined['bowling_team'].fillna('Unknown')

    df_combined = df_combined[df_combined['batting_team'] != 'Unknown']

    return df_combined

def assign_roles(df):
    def role(row):
        if row['Predicted_FP'] >= 25 and row['Predicted_Bowling_FP'] >= 25:
            return 'AR'
        elif row['Predicted_Bowling_FP'] >= 25:
            return 'BWL'
        elif row['Predicted_FP'] >= 25:
            return 'BAT'
        return 'BAT'
    df['role'] = df.apply(role, axis=1)
    return df

def build_role_constrained_team(
    df, max_players=11, max_credits=100, max_per_team=7,
    min_roles={"WK": 1, "BAT": 2, "AR": 0, "BWL": 2},
    max_roles={"WK": 1, "BAT": 5, "AR": 2, "BWL": 5}):

    team = []
    team_credits = 0
    team_counts = {}
    role_counts = {'WK': 0, 'BAT': 0, 'AR': 0, 'BWL': 0}
    selected_players = set()  # Track selected players to prevent duplicates

    # Filter out players with 'Unknown' team
    df = df[df['batting_team'] != 'Unknown']

    # Filter to only include currently active players
    players_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/players.csv"))
    active_players = pd.read_csv(players_path)["players"].unique()
    df = df[df['fullName'].isin(active_players)]

    # Relax the constraints for ARs and other roles further
    min_roles = {"WK": 1, "BAT": 2, "AR": 0, "BWL": 1}
    max_roles = {"WK": 1, "BAT": 5, "AR": 3, "BWL": 5}

    df_sorted = df.sort_values(by='Total_FP', ascending=False)

    for _, player in df_sorted.iterrows():
        if len(team) >= max_players:
            continue

        name = player['fullName']
        credit = player['credit']
        role = player['role']
        fp = player['Total_FP']
        team_name = player['batting_team']

        if name in selected_players:  # Skip if the player is already in the team
            continue

        if role_counts[role] >= max_roles[role]:
            continue
        if team_counts.get(team_name, 0) >= max_per_team:
            continue
        if team_credits + credit > max_credits:
            continue

        team.append({'name': name, 'team': team_name, 'role': role, 'credit': credit, 'total_fp': fp})
        team_credits += credit
        team_counts[team_name] = team_counts.get(team_name, 0) + 1
        role_counts[role] += 1
        selected_players.add(name)  # Add player to selected set

    # If ARs are missing, skip enforcing AR role
    if role_counts['AR'] < 1:
        print("⚠️ Not enough ARs selected — skipping AR constraints...")

    # Ensure team size is 11, balance team if necessary
    if len(team) < max_players:
        for _, player in df_sorted.iterrows():
            if len(team) >= max_players:
                break
            if player['fullName'] in selected_players:
                continue
            if team_counts.get(player['batting_team'], 0) >= max_per_team:
                continue
            if role_counts.get(player['role'], 0) >= max_roles.get(player['role'], max_players):
                continue
            if team_credits + player['credit'] > max_credits:
                continue
            team.append({'name': player['fullName'], 'team': player['batting_team'], 'role': player['role'], 'credit': player['credit'], 'total_fp': player['Total_FP']})
            selected_players.add(player['fullName'])
            team_counts[player['batting_team']] = team_counts.get(player['batting_team'], 0) + 1
            role_counts[player['role']] += 1
            team_credits += player['credit']

    if len(team) < max_players:
        print("❌ Invalid team under constraints.")
        return pd.DataFrame()

    return pd.DataFrame(team)

def apply_captaincy_boost(df_team):
    captain = df_team.iloc[0]
    vice_captain = df_team.iloc[1]
    df_team['adjusted_fp'] = df_team['total_fp']
    df_team.loc[df_team['name'] == captain['name'], 'adjusted_fp'] *= 2
    df_team.loc[df_team['name'] == vice_captain['name'], 'adjusted_fp'] *= 1.5
    return df_team