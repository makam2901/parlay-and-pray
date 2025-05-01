# Basic setup
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
warnings.filterwarnings('ignore')

# Load dataset
batting = pd.read_csv("data/Batting_data.csv")
bowling = pd.read_csv("data/Bowling_data.csv")
fielding = pd.read_csv("data/Fielding_data.csv")
fantasy = pd.read_csv("data/Final_Fantasy_data.csv")
match = pd.read_csv("data/Match_details.csv")
players = pd.read_csv("data/players.csv")

batting['credit'] = np.random.uniform(7.0, 10.0, size=len(batting)).round(1)  # simulate realistic Dream11 range

batting.dropna(subset=['Batting_FP'], inplace=True)
batting = batting[batting['balls'] > 0]
categorical_cols = ['season', 'home_team', 'away_team', 'batting_team', 'bowling_team']
batting[categorical_cols] = batting[categorical_cols].astype('category')

features = ['runs', 'balls', 'fours', 'sixes', 'strike_rate', 'batting_position', 'batting_innings']
X_bat = batting[features]
y_bat = batting['Batting_FP']

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_bat, y_bat, test_size=0.2, random_state=42)

# Train model
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Predict on the full dataset for optimizer
batting['Predicted_FP'] = rf.predict(X_bat)

def build_best_team(df, max_players=11, max_credits=100, max_players_per_team=7):
    team = []
    team_credits = 0
    team_counts = {}
    
    # Sort players by predicted fantasy points (descending)
    df_sorted = df.sort_values(by='Predicted_FP', ascending=False)
    
    for _, player in df_sorted.iterrows():
        if len(team) >= max_players:
            break
        
        # Extract player info
        name = player['fullName']
        credit = player['credit']
        team_name = player['batting_team']
        fp = player['Predicted_FP']
        
        # Team-specific player limit
        if team_counts.get(team_name, 0) >= max_players_per_team:
            continue
        
        # Credit constraint
        if team_credits + credit > max_credits:
            continue
        
        # Add player
        team.append({
            'name': name,
            'team': team_name,
            'credit': credit,
            'predicted_fp': fp
        })
        
        team_credits += credit
        team_counts[team_name] = team_counts.get(team_name, 0) + 1

    return pd.DataFrame(team)

# Drop NA targets
bowling = bowling.dropna(subset=['Bowling_FP'])

# Define features and target
bowling_features = ['overs', 'maidens', 'conceded', 'wickets', 'economyRate', 'wides', 'noballs']
X_bowl = bowling[bowling_features]
y_bowl = bowling['Bowling_FP']

# Train/test split
Xb_train, Xb_test, yb_train, yb_test = train_test_split(X_bowl, y_bowl, test_size=0.2, random_state=42)

# Train model
rf_bowl = RandomForestRegressor(n_estimators=200, max_depth=10)
rf_bowl.fit(Xb_train, yb_train)

# Predict for all data
bowling['Predicted_Bowling_FP'] = rf_bowl.predict(X_bowl)

df_combined = pd.merge(batting, bowling[['match_id', 'fullName', 'Predicted_Bowling_FP']],
                       on=['match_id', 'fullName'], how='outer')

# Fill missing values
df_combined['Predicted_FP'] = df_combined['Predicted_FP'].fillna(0)
df_combined['Predicted_Bowling_FP'] = df_combined['Predicted_Bowling_FP'].fillna(0)

# Create total predicted fantasy points
df_combined['Total_FP'] = df_combined['Predicted_FP'] + df_combined['Predicted_Bowling_FP']
df_combined['credit'] = np.random.uniform(7.0, 10.0, size=len(df_combined)).round(1)

def assign_role(row):
    batting_fp = row.get('Predicted_FP', 0)
    bowling_fp = row.get('Predicted_Bowling_FP', 0)

    if bowling_fp >= 25 and batting_fp >= 25:
        return 'AR'  # All-Rounder
    elif bowling_fp >= 25:
        return 'BWL'  # Bowler
    elif batting_fp >= 25:
        return 'BAT'  # Batter
    else:
        return 'BAT'  # Default to BAT if unclear

df_combined['role'] = df_combined.apply(assign_role, axis=1)

def build_role_constrained_team(df, max_players=11, max_credits=100, max_per_team=7):
    team = []
    team_credits = 0
    team_counts = {}
    role_counts = {'WK': 0, 'BAT': 0, 'AR': 0, 'BWL': 0}

    # Simulate role 'WK' for keeper-like players (if needed)
    if 'WK' not in df['role'].unique():
        wk_candidates = df.sort_values(by='Total_FP', ascending=False).head(5)
        df.loc[wk_candidates.index[:1], 'role'] = 'WK'

    # Sort by total fantasy points
    df_sorted = df.sort_values(by='Total_FP', ascending=False)

    for _, player in df_sorted.iterrows():
        if len(team) >= max_players:
            break

        name = player['fullName']
        credit = player['credit']
        role = player['role']
        fp = player['Total_FP']
        team_name = player['batting_team']

        # Role limits
        if role_counts[role] >= {'WK': 4, 'BAT': 6, 'AR': 4, 'BWL': 6}[role]:
            continue

        # Team constraint
        if team_counts.get(team_name, 0) >= max_per_team:
            continue

        # Credit check
        if team_credits + credit > max_credits:
            continue

        # Add player
        team.append({
            'name': name,
            'team': team_name,
            'role': role,
            'credit': credit,
            'total_fp': fp
        })

        # Update counters
        team_credits += credit
        team_counts[team_name] = team_counts.get(team_name, 0) + 1
        role_counts[role] += 1

    # Validate minimum role constraints
    if (role_counts['WK'] < 1 or role_counts['BAT'] < 3 or 
        role_counts['AR'] < 1 or role_counts['BWL'] < 3 or len(team) != 11):
        print("❌ Could not build a valid team under constraints. Try expanding data or tweaking logic.")
        return pd.DataFrame()
    
    return pd.DataFrame(team)

df_combined = df_combined.sort_values(by='Total_FP', ascending=False).drop_duplicates(subset='fullName')
df_combined['batting_team'] = df_combined['batting_team'].fillna(df_combined['bowling_team'])

final_team = build_role_constrained_team(df_combined)

captain = final_team.iloc[0]
vice_captain = final_team.iloc[1]

final_team['adjusted_fp'] = final_team['total_fp']
final_team.loc[final_team['name'] == captain['name'], 'adjusted_fp'] *= 2
final_team.loc[final_team['name'] == vice_captain['name'], 'adjusted_fp'] *= 1.5

print("\n💥 Team with Captain & Vice-Captain Boost:")
print(final_team[['name', 'team','role', 'credit', 'total_fp', 'adjusted_fp']])
print(f"\n🎯 Adjusted Total FP: {final_team['adjusted_fp'].sum():.2f}")
