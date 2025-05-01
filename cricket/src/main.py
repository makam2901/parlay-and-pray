from utils import load_data, filter_match_data
from preprocessing import preprocess_batting, preprocess_bowling
from training import train_model
from scoring import (
    merge_predictions,
    assign_roles,
    build_role_constrained_team,
    apply_captaincy_boost
)

def main():
    # Get user input for the match teams
    home_team = input("Enter the home team: ")
    away_team = input("Enter the away team: ")

    # 1. Load all data
    print("📦 Loading data...")
    batting, bowling, fielding, fantasy, match, players = load_data()

    # 2. Filter data based on selected match teams
    print(f"🔍 Filtering data for match: {home_team} vs {away_team}...")
    batting_filtered, bowling_filtered = filter_match_data(batting, bowling, home_team, away_team)

    # 3. Preprocess
    print("🧹 Preprocessing...")
    X_bat, y_bat, batting_filtered = preprocess_batting(batting_filtered)
    X_bowl, y_bowl, bowling_filtered = preprocess_bowling(bowling_filtered)

    # 4. Train models
    print("🧠 Training models...")
    model_bat = train_model(X_bat, y_bat)
    model_bowl = train_model(X_bowl, y_bowl, n_estimators=200, max_depth=10)

    # 5. Scoring
    print("📈 Scoring players...")
    df_combined = merge_predictions(
        batting_filtered, bowling_filtered,
        model_bat, model_bowl,
        X_bat.columns, X_bowl.columns
    )
    df_combined = df_combined[df_combined['batting_team'] != 'Unknown']
    df_combined = assign_roles(df_combined)
    print("🔍 Role distribution:")
    print(df_combined['role'].value_counts())

    # 6. Get custom inputs from user
    min_bat = int(input("Enter minimum number of BAT (Batsmen): "))
    max_bat = int(input("Enter maximum number of BAT (Batsmen): "))
    min_ar = int(input("Enter minimum number of AR (All-rounders): "))
    max_ar = int(input("Enter maximum number of AR (All-rounders): "))
    min_bwl = int(input("Enter minimum number of BWL (Bowlers): "))
    max_bwl = int(input("Enter maximum number of BWL (Bowlers): "))

    # 7. Build team with custom role constraints
    print("👥 Building best team with custom constraints...")
    final_team = build_role_constrained_team(
        df_combined,
        min_roles={"WK": 1, "BAT": min_bat, "AR": min_ar, "BWL": min_bwl},
        max_roles={"WK": 1, "BAT": max_bat, "AR": max_ar, "BWL": max_bwl}
    )
    if final_team.empty:
        print("❌ Could not build a valid team.")
        return

    final_team = apply_captaincy_boost(final_team)

    # 8. Output
    print("\n💥 Final Dream11 Team (with Captain & Vice-Captain boosts):")
    print(final_team[['name', 'team', 'role', 'credit', 'total_fp', 'adjusted_fp']])
    print(f"\n🎯 Total Adjusted FP: {final_team['adjusted_fp'].sum():.2f}")

if __name__ == "__main__":
    main()