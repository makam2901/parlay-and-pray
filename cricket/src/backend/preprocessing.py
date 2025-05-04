import pandas as pd
import numpy as np
import os
import logging
# --- NEW IMPORTS ---
from google.cloud import secretmanager
import tempfile
import atexit
# --- END NEW IMPORTS ---

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Keep this global variable for cleanup reference ---
_temp_key_file_path = None

def _cleanup_keyfile():
    """Cleanup function to delete the temporary key file."""
    global _temp_key_file_path
    if _temp_key_file_path and os.path.exists(_temp_key_file_path):
        try:
            logging.info(f"Cleaning up temporary key file: {_temp_key_file_path}")
            os.remove(_temp_key_file_path)
            _temp_key_file_path = None # Avoid trying to delete again
        except Exception as e:
            logging.error(f"Error cleaning up key file {_temp_key_file_path}: {e}", exc_info=True)

# --- Register cleanup function ONCE when the module is loaded ---
atexit.register(_cleanup_keyfile)


class DataPreprocessor:
    """Handles data loading and preprocessing for Dream11 predictions"""

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
        logging.info("DataPreprocessor initialized.")

    def load_data(self):
        """Load the datasets using the service account key fetched from Secret Manager."""
        global _temp_key_file_path # Use global variable to store path for cleanup

        bucket_path = "gs://mlops-dream11-data" # Bucket in the other project
        # --- Define your Secret Manager secret resource name ---
        secret_resource_name = "projects/188982653956/secrets/gcs-cross-project-key/versions/latest" # Or specific version

        key_file_path = None # Initialize key_file_path

        try:
            logging.info(f"Attempting to fetch secret: {secret_resource_name}")
            client = secretmanager.SecretManagerServiceClient()
            # The client automatically uses Workload Identity credentials
            response = client.access_secret_version(request={"name": secret_resource_name})
            sa_key_bytes = response.payload.data
            logging.info(f"Successfully fetched secret payload ({len(sa_key_bytes)} bytes)")

            # Create a temporary file to store the key
            # Note: delete=False means we control deletion via atexit
            with tempfile.NamedTemporaryFile(delete=False, suffix=".json", mode='wb') as temp_key_file:
              key_file_path = temp_key_file.name
              temp_key_file.write(sa_key_bytes)
              # File is automatically closed when 'with' block exits

            _temp_key_file_path = key_file_path # Store path for atexit cleanup
            logging.info(f"Service account key saved temporarily to: {key_file_path}")

            # --- CRITICAL: Set the environment variable FOR THIS PROCESS ---
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = key_file_path
            logging.info(f"Set GOOGLE_APPLICATION_CREDENTIALS to: {key_file_path}")

            # --- The original checks below are now REMOVED as they caused the crash ---
            # if not key_file_path: ...
            # if not os.path.exists(key_file_path): ...

            logging.info(f"Attempting to load data from bucket: {bucket_path} using fetched credentials.")

            # gcsfs (used by pandas) will now automatically pick up credentials
            # from the file specified by the GOOGLE_APPLICATION_CREDENTIALS env var.
            # No need for storage_options={"token": ...}
            storage_options = {}

            # Define file paths
            batting_file = f"{bucket_path}/Batting_data.csv"
            bowling_file = f"{bucket_path}/Bowling_data.csv"
            fielding_file = f"{bucket_path}/Fielding_data.csv"
            match_file = f"{bucket_path}/Match_details.csv"
            players_file = f"{bucket_path}/players.csv"

            logging.info(f"Loading batting data from {batting_file}")
            self.batting = pd.read_csv(batting_file, storage_options=storage_options)
            logging.info(f"Loaded {len(self.batting)} batting records.")

            logging.info(f"Loading bowling data from {bowling_file}")
            self.bowling = pd.read_csv(bowling_file, storage_options=storage_options)
            logging.info(f"Loaded {len(self.bowling)} bowling records.")

            logging.info(f"Loading fielding data from {fielding_file}")
            self.fielding = pd.read_csv(fielding_file, storage_options=storage_options)
            logging.info(f"Loaded {len(self.fielding)} fielding records.")

            logging.info(f"Loading match data from {match_file}")
            self.match = pd.read_csv(match_file, storage_options=storage_options)
            logging.info(f"Loaded {len(self.match)} match records.")

            logging.info(f"Loading players data from {players_file}")
            self.players = pd.read_csv(players_file, storage_options=storage_options)
            logging.info(f"Loaded {len(self.players)} player records.")

            self.data_loaded = True
            logging.info("All datasets loaded successfully using credentials fetched from Secret Manager.")

        except Exception as e:
            logging.error(f"Failed during data loading using Secret Manager key {secret_resource_name}: {e}", exc_info=True)
            # Re-raise the exception so the application startup fails clearly if data loading fails
            raise
        # No finally block needed for cleanup; atexit handles it

    # --- Rest of the DataPreprocessor class methods remain the same ---
    # (get_team_players, prepare_match_data, preprocess_batting, etc.)

    def get_team_players(self, team_name):
        """Get current players for a team"""
        if not self.data_loaded:
            self.load_data() # Ensure data is loaded if accessed directly
        if self.players is None:
            raise ValueError("Players data not loaded.")
        team_players = self.players[self.players['team'] == team_name]['players'].tolist()
        if not team_players:
            # Instead of raising error, maybe return empty list or log warning?
            logging.warning(f"No players found for team: {team_name}")
            # raise ValueError(f"No players found for team: {team_name}")
        return team_players

    def prepare_match_data(self, home_team, away_team):
        """Prepare data for model training for specific teams"""
        if not self.data_loaded:
            self.load_data()

        if self.batting is None or self.bowling is None or self.fielding is None or self.players is None or self.match is None:
             raise ValueError("One or more dataframes failed to load. Cannot prepare match data.")


        logging.info(f"Preparing match data for {home_team} vs {away_team}...") # Changed print to logging

        # Get current players for both teams
        home_players = self.get_team_players(home_team)
        away_players = self.get_team_players(away_team)
        all_players = home_players + away_players

        if not all_players:
             logging.warning(f"No players identified for match {home_team} vs {away_team}. Returning empty dataframes.")
             return [], pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), None


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

        logging.info(f"Prepared data for {len(all_players)} players") # Changed print to logging
        logging.info(f"Selected venue: {venue}") # Changed print to logging

        return all_players, team_batting, team_bowling, team_fielding, venue

    def preprocess_batting(self, batting_df, year_split=2023):
        """Preprocess batting data with enhanced feature engineering"""
        batting_df = batting_df.copy()

        # Basic cleaning
        batting_df.dropna(subset=['Batting_FP'], inplace=True)
        # Ensure 'balls' column exists and handle potential non-numeric types before comparison
        if 'balls' in batting_df.columns:
             batting_df['balls'] = pd.to_numeric(batting_df['balls'], errors='coerce')
             batting_df.dropna(subset=['balls'], inplace=True) # Drop rows where balls couldn't be converted
             batting_df = batting_df[batting_df['balls'] > 0]
        else:
             logging.warning("Column 'balls' not found in batting data during preprocessing.")
             # Depending on requirements, either return empty df, raise error, or proceed without this filter
             return pd.DataFrame(), pd.Series(), pd.DataFrame(), pd.Series() # Example: return empty


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
        # Drop rows where season is NaN after conversion, as it's needed for split
        batting_df.dropna(subset=['season'], inplace=True)

        # Create recent form feature
        # Ensure match_id exists and is sortable
        if 'match_id' not in batting_df.columns:
             logging.error("Column 'match_id' required for sorting by time is missing.")
             raise ValueError("Missing 'match_id' column for recent form calculation.")
        batting_df = batting_df.sort_values(['fullName', 'match_id'])
        batting_df['recent_form'] = batting_df.groupby('fullName')['Batting_FP'].transform(
            lambda x: x.shift(1).rolling(window=5, min_periods=1).mean()
        )
        batting_df['recent_form'] = batting_df['recent_form'].fillna(0) # Fill NaNs resulting from rolling mean

        # Split data by year
        train_batting = batting_df[batting_df['season'] < year_split].copy() # Use .copy() to avoid SettingWithCopyWarning
        test_batting = batting_df[batting_df['season'] >= year_split].copy()

        # Define features
        features = [
            'runs', 'balls', 'fours', 'sixes', 'strike_rate', 'batting_position',
            'boundary_percentage', 'runs_per_ball', 'boundary_runs', 'non_boundary_runs',
            'recent_form', 'batting_innings'
        ]

        # Ensure all features exist and handle NaNs/Infs *before* returning
        X_train = train_batting.reindex(columns=features, fill_value=0)
        X_test = test_batting.reindex(columns=features, fill_value=0)
        y_train = train_batting['Batting_FP'].fillna(0) # Ensure target has no NaNs
        y_test = test_batting['Batting_FP'].fillna(0)

        # Clean features after ensuring columns exist
        X_train = self._clean_features(X_train)
        X_test = self._clean_features(X_test)


        # Check for empty dataframes after split/filtering
        if X_train.empty or X_test.empty:
             logging.warning(f"Batting preprocessing resulted in empty train or test set (Train: {len(X_train)}, Test: {len(X_test)}).")


        return X_train, y_train, X_test, y_test


    def preprocess_bowling(self, bowling_df, year_split=2023):
        """Preprocess bowling data with enhanced feature engineering"""
        bowling_df = bowling_df.copy()

        # Basic cleaning
        bowling_df.dropna(subset=['Bowling_FP'], inplace=True)

        # Feature engineering
        bowling_df['overs'] = pd.to_numeric(bowling_df['overs'], errors='coerce')
        # Drop rows where 'overs' is NaN as it's needed for balls_bowled
        bowling_df.dropna(subset=['overs'], inplace=True)

        bowling_df['balls_bowled'] = bowling_df['overs'].apply(
            lambda x: int(x) * 6 + int(round((x % 1) * 10)) if pd.notnull(x) else 0 # Corrected calculation for balls from overs
        )

        # Ensure 'conceded' exists and is numeric before calculating dot_balls
        if 'conceded' in bowling_df.columns:
             bowling_df['conceded'] = pd.to_numeric(bowling_df['conceded'], errors='coerce').fillna(0)
             bowling_df['dot_balls'] = bowling_df.apply(
                 lambda x: max(0, x['balls_bowled'] - x['conceded']) if x['balls_bowled'] > 0 else 0,
                 axis=1
             )
             bowling_df['dot_ball_percentage'] = bowling_df.apply(
                 lambda x: x['dot_balls'] / x['balls_bowled'] if x['balls_bowled'] > 0 else 0,
                 axis=1
             )
        else:
             logging.warning("Column 'conceded' not found. Skipping dot ball calculations.")
             bowling_df['dot_balls'] = 0
             bowling_df['dot_ball_percentage'] = 0


        # Ensure 'wickets' exists and is numeric before calculating wicket_rate
        if 'wickets' in bowling_df.columns:
             bowling_df['wickets'] = pd.to_numeric(bowling_df['wickets'], errors='coerce').fillna(0)
             bowling_df['wicket_rate'] = bowling_df.apply(
                 lambda x: x['wickets'] / x['balls_bowled'] if x['balls_bowled'] > 0 else 0,
                 axis=1
             )
        else:
             logging.warning("Column 'wickets' not found. Skipping wicket rate calculation.")
             bowling_df['wicket_rate'] = 0


        # Handle categorical data
        categorical_cols = ['season', 'home_team', 'away_team', 'batting_team', 'bowling_team', 'venue']
        for col in categorical_cols:
            if col in bowling_df.columns:
                bowling_df[col] = bowling_df[col].astype('category')

        # Convert season to numeric
        bowling_df['season'] = pd.to_numeric(bowling_df['season'], errors='coerce')
        bowling_df.dropna(subset=['season'], inplace=True)


        # Create recent form feature
        if 'match_id' not in bowling_df.columns:
             logging.error("Column 'match_id' required for sorting by time is missing.")
             raise ValueError("Missing 'match_id' column for recent form calculation.")
        bowling_df = bowling_df.sort_values(['fullName', 'match_id'])
        bowling_df['recent_form'] = bowling_df.groupby('fullName')['Bowling_FP'].transform(
            lambda x: x.shift(1).rolling(window=5, min_periods=1).mean()
        )
        bowling_df['recent_form'] = bowling_df['recent_form'].fillna(0)

        # Split data by year
        train_bowling = bowling_df[bowling_df['season'] < year_split].copy()
        test_bowling = bowling_df[bowling_df['season'] >= year_split].copy()

        # Define features
        features = [
            'overs', 'maidens', 'conceded', 'wickets', 'economyRate', 'wides', 'noballs',
            'dot_ball_percentage', 'wicket_rate', 'recent_form', 'bowling_innings'
        ]

        # Ensure all features exist and handle NaNs/Infs *before* returning
        X_train = train_bowling.reindex(columns=features, fill_value=0)
        X_test = test_bowling.reindex(columns=features, fill_value=0)
        y_train = train_bowling['Bowling_FP'].fillna(0) # Ensure target has no NaNs
        y_test = test_bowling['Bowling_FP'].fillna(0)

        # Clean features after ensuring columns exist
        X_train = self._clean_features(X_train)
        X_test = self._clean_features(X_test)

        # Check for empty dataframes after split/filtering
        if X_train.empty or X_test.empty:
             logging.warning(f"Bowling preprocessing resulted in empty train or test set (Train: {len(X_train)}, Test: {len(X_test)}).")


        return X_train, y_train, X_test, y_test

    def _clean_features(self, df):
        """Clean DataFrame by removing NaNs, infs, and extreme values"""
        df = df.copy()

        # Replace inf with NaN FIRST
        df.replace([np.inf, -np.inf], np.nan, inplace=True)

        # Fill remaining NaNs with 0 (or a suitable strategy like mean/median if appropriate)
        # Using 0 as per previous logic, but consider if mean/median makes more sense for some features
        df.fillna(0, inplace=True)

        # Handle extreme values - Check for numeric types before applying quantile
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        for col in numeric_cols:
            # Skip if column is all zeros or constant
            if df[col].nunique() <= 1:
                 continue

            try:
                 q_low = df[col].quantile(0.001)
                 q_high = df[col].quantile(0.999)
                 # Only clip if quantiles are meaningful (not same as min/max if constant)
                 if q_low < q_high:
                     # Ensure limits are finite before comparison (handle potential NaN quantiles)
                     finite_q_high = q_high if np.isfinite(q_high) else np.finfo(df[col].dtype).max
                     finite_q_low = q_low if np.isfinite(q_low) else np.finfo(df[col].dtype).min

                     # Clip using the original (potentially non-finite) quantiles
                     # This preserves behavior if quantiles are inf/-inf due to column values
                     df[col] = df[col].clip(lower=q_low, upper=q_high)

            except Exception as e:
                 logging.warning(f"Could not clip column {col}: {e}")


        return df