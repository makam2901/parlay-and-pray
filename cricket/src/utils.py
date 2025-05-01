import pandas as pd

def load_data():
    batting = pd.read_csv("data/Batting_data.csv")
    bowling = pd.read_csv("data/Bowling_data.csv")
    fielding = pd.read_csv("data/Fielding_data.csv")
    fantasy = pd.read_csv("data/Final_Fantasy_data.csv")
    match = pd.read_csv("data/Match_details.csv")
    players = pd.read_csv("data/players.csv")
    return batting, bowling, fielding, fantasy, match, players

def filter_match_data(batting, bowling, home_team, away_team):
    batting_filtered = batting[
        ((batting['home_team'] == home_team) & (batting['away_team'] == away_team)) |
        ((batting['home_team'] == away_team) & (batting['away_team'] == home_team))
    ]
    bowling_filtered = bowling[
        ((bowling['home_team'] == home_team) & (bowling['away_team'] == away_team)) |
        ((bowling['home_team'] == away_team) & (bowling['away_team'] == home_team))
    ]
    return batting_filtered, bowling_filtered