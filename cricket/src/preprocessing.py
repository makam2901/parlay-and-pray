import numpy as np

def preprocess_batting(batting_df):
    batting_df = batting_df.copy()
    batting_df.dropna(subset=['Batting_FP'], inplace=True)
    batting_df = batting_df[batting_df['balls'] > 0]
    batting_df['credit'] = np.random.uniform(7.0, 10.0, size=len(batting_df)).round(1)
    batting_df[['season', 'home_team', 'away_team', 'batting_team', 'bowling_team']] = \
        batting_df[['season', 'home_team', 'away_team', 'batting_team', 'bowling_team']].astype('category')
    
    features = ['runs', 'balls', 'fours', 'sixes', 'strike_rate', 'batting_position', 'batting_innings']
    X = batting_df[features]
    y = batting_df['Batting_FP']
    return X, y, batting_df

def preprocess_bowling(bowling_df):
    bowling_df = bowling_df.copy()
    bowling_df.dropna(subset=['Bowling_FP'], inplace=True)
    features = ['overs', 'maidens', 'conceded', 'wickets', 'economyRate', 'wides', 'noballs']
    X = bowling_df[features]
    y = bowling_df['Bowling_FP']
    return X, y, bowling_df