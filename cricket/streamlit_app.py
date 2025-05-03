# streamlit_app.py
import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
from PIL import Image
import io
import base64

# Page config
st.set_page_config(
    page_title="Dream11 Fantasy Team Predictor",
    page_icon="🏏",
    layout="wide"
)

# FastAPI backend URL
API_URL = "http://localhost:8000/predict_team/"

# Teams and venues
teams = [
    "CSK", "DC", "GT", "KKR", "LSG", "MI", "PBKS", "RCB", "RR", "SRH"
]

venues = [
    "MA Chidambaram Stadium, Chepauk, Chennai",
    "M.Chinnaswamy Stadium, Bengaluru",
    "Wankhede Stadium, Mumbai",
    "Arun Jaitley Stadium, Delhi",
    "Rajiv Gandhi International Stadium, Uppal, Hyderabad",
    "Narendra Modi Stadium, Motera, Ahmedabad",
    "Eden Gardens, Kolkata",
    "Bharat Ratna Shri Atal Bihari Vajpayee Ekana Cricket Stadium, Lucknow",
    "Barsapara Cricket Stadium, Guwahati",
    "Punjab Cricket Association IS Bindra Stadium, Mohali, Chandigarh"
]

# Team colors for visualization
team_colors = {
    "CSK": "#FFFF00",  # Yellow
    "DC": "#0080FF",   # Blue
    "GT": "#00008B",   # Dark Blue
    "KKR": "#800080",  # Purple
    "LSG": "#87CEEB",  # Sky Blue
    "MI": "#0000FF",   # Blue
    "PBKS": "#FF0000", # Red
    "RCB": "#FF0000",  # Red
    "RR": "#FFC0CB",   # Pink
    "SRH": "#FFA500"   # Orange
}

# Role icons
role_icons = {
    "WK": "🧤",
    "BAT": "🏏",
    "AR": "⚡",
    "BWL": "🎯"
}

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #1E3A8A;
        margin-top: 1rem;
    }
    .player-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 10px;
        margin: 5px 0;
        border-left: 5px solid #4CAF50;
    }
    .captain {
        border-left: 5px solid #FFD700;
        background-color: #FFFDE7;
    }
    .vice-captain {
        border-left: 5px solid #C0C0C0;
        background-color: #F5F5F5;
    }
    .stats-box {
        background-color: #e9ecef;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown("<h1 class='main-header'>🏏 Dream11 Fantasy Team Predictor</h1>", unsafe_allow_html=True)

# Main Layout - Split into sidebar and main area
with st.sidebar:
    st.header("Match Setup")
    
    # Team selection with logos
    home_team = st.selectbox("Select Home Team", teams)
    away_team = st.selectbox("Select Away Team", [t for t in teams if t != home_team])
    
    # Venue selection with smart defaults based on home team
    default_venue_index = 0
    team_home_venues = {
        "CSK": "MA Chidambaram Stadium, Chepauk, Chennai",
        "RCB": "M.Chinnaswamy Stadium, Bengaluru",
        "MI": "Wankhede Stadium, Mumbai",
        "DC": "Arun Jaitley Stadium, Delhi",
        "SRH": "Rajiv Gandhi International Stadium, Uppal, Hyderabad",
        "GT": "Narendra Modi Stadium, Motera, Ahmedabad",
        "KKR": "Eden Gardens, Kolkata",
        "LSG": "Bharat Ratna Shri Atal Bihari Vajpayee Ekana Cricket Stadium, Lucknow",
        "RR": "Barsapara Cricket Stadium, Guwahati",
        "PBKS": "Punjab Cricket Association IS Bindra Stadium, Mohali, Chandigarh"
    }
    
    if home_team in team_home_venues:
        default_venue = team_home_venues[home_team]
        default_venue_index = venues.index(default_venue)
    
    venue = st.selectbox("Select Venue", venues, index=default_venue_index)
    
    st.header("Team Composition")
    
    # Role counts with validation
    col1, col2 = st.columns(2)
    with col1:
        wk_count = st.number_input("Wicketkeepers (WK)", min_value=1, max_value=2, value=1)
        bat_count = st.number_input("Batsmen (BAT)", min_value=3, max_value=5, value=4)
    
    with col2:
        ar_count = st.number_input("All-Rounders (AR)", min_value=1, max_value=3, value=2)
        bwl_count = st.number_input("Bowlers (BWL)", min_value=3, max_value=5, value=4)
    
    # Validate total players
    total_players = wk_count + bat_count + ar_count + bwl_count
    if total_players != 11:
        st.warning(f"Total players: {total_players}/11. Please adjust to have exactly 11 players.")
    else:
        st.success(f"Total players: {total_players}/11 ✓")
    
    # Advanced options (collapsible)
    with st.expander("Advanced Options"):
        max_credits = st.slider("Max Credits", min_value=80, max_value=100, value=100, step=1)
        max_per_team = st.slider("Max Players per Team", min_value=4, max_value=7, value=7, step=1)
        st.checkbox("Apply Form Factors", value=True)
        st.checkbox("Apply Venue Factors", value=True)
        st.checkbox("Apply Opponent Factors", value=True)
    
    # Predict button
    predict_button = st.button("🔮 Predict Team", type="primary", disabled=total_players != 11)

# Initialize session state to store predictions
if "prediction_result" not in st.session_state:
    st.session_state.prediction_result = None
if "team_visualized" not in st.session_state:
    st.session_state.team_visualized = False

# Handle prediction
if predict_button:
    with st.spinner("Predicting optimal team..."):
        request_data = {
            "home_team": home_team,
            "away_team": away_team,
            "venue": venue,
            "train_new": True,
            "wk_count": int(wk_count),
            "bat_count": int(bat_count),
            "ar_count": int(ar_count),
            "bwl_count": int(bwl_count),
            "max_credits": max_credits,
            "max_per_team": max_per_team
        }
        
        try:
            response = requests.post(API_URL, json=request_data)
            response.raise_for_status()
            st.session_state.prediction_result = response.json()
            st.session_state.team_visualized = False
        except Exception as e:
            st.error(f"❌ Failed to predict team: {str(e)}")
            st.info("Make sure your backend API is running at " + API_URL)

# Display prediction results if availables
if st.session_state.prediction_result:
    result = {
        "team": st.session_state.prediction_result["team_raw"]
    }
    
    # Create tabs for different views
    tab1, tab2, tab3 = st.tabs(["Team Selection", "Team Statistics", "Player Details"])
    
    with tab1:
        st.markdown(f"### {home_team} vs {away_team}")
        st.markdown(f"**Venue:** {venue}")
        
        # Display team with styling
        home_players = [p for p in result["team"] if p["team"] == home_team]
        away_players = [p for p in result["team"] if p["team"] == away_team]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"#### {home_team} ({len(home_players)} players)")
            for player in sorted(home_players, key=lambda x: (-x.get("is_captain", False), -x.get("is_vice_captain", False), -x["total_fp"])):
                cap = " (C)" if player.get("is_captain") else ""
                vc = " (VC)" if player.get("is_vice_captain") else ""
                role = f"{role_icons.get(player['role'], '')} {player['role']}"
                
                # Determine card class
                card_class = "player-card"
                if player.get("is_captain"):
                    card_class += " captain"
                elif player.get("is_vice_captain"):
                    card_class += " vice-captain"
                
                st.markdown(f"""
                <div class='{card_class}'>
                    <b>{player['name']}{cap}{vc}</b><br>
                    {role} | 💰 {player['credit']} | ⭐ {round(player['total_fp'], 1)} pts
                </div>
                """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"#### {away_team} ({len(away_players)} players)")
            for player in sorted(away_players, key=lambda x: (-x.get("is_captain", False), -x.get("is_vice_captain", False), -x["total_fp"])):
                cap = " (C)" if player.get("is_captain") else ""
                vc = " (VC)" if player.get("is_vice_captain") else ""
                role = f"{role_icons.get(player['role'], '')} {player['role']}"
                
                # Determine card class
                card_class = "player-card"
                if player.get("is_captain"):
                    card_class += " captain"
                elif player.get("is_vice_captain"):
                    card_class += " vice-captain"
                
                st.markdown(f"""
                <div class='{card_class}'>
                    <b>{player['name']}{cap}{vc}</b><br>
                    {role} | 💰 {player['credit']} | ⭐ {round(player['total_fp'], 1)} pts
                </div>
                """, unsafe_allow_html=True)
        
        # Summary statistics
        st.markdown("### Team Summary")
        col1, col2, col3, col4 = st.columns(4)
        
        total_credits = sum(p["credit"] for p in result["team"])
        total_points = sum(p["adjusted_fp"] for p in result["team"])
        
        with col1:
            st.metric("Total Credits", f"{round(total_credits, 1)}/100")
        
        with col2:
            st.metric("Total Points", round(total_points, 1))
        
        with col3:
            captain = next((p["name"] for p in result["team"] if p.get("is_captain")), "Unknown")
            st.metric("Captain", f"{captain}")

        with col4:
            vc = next((p["name"] for p in result["team"] if p.get("is_vice_captain")), "Unknown")
            st.metric("Vice-Captain", f"{vc}")
    
    with tab2:
        if not st.session_state.team_visualized:
            # Create visualizations
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # Plot role distribution
            role_counts = {"WK": 0, "BAT": 0, "AR": 0, "BWL": 0}
            for player in result["team"]:
                role_counts[player["role"]] += 1
            
            ax1.bar(role_counts.keys(), role_counts.values(), color=['blue', 'green', 'orange', 'red'])
            ax1.set_title('Team Composition by Role')
            ax1.set_ylabel('Number of Players')
            
            # Plot team distribution
            team_counts = {home_team: 0, away_team: 0}
            for player in result["team"]:
                team_counts[player["team"]] += 1
            
            team_colors_list = [team_colors.get(team, "#888888") for team in team_counts.keys()]
            ax2.bar(team_counts.keys(), team_counts.values(), color=team_colors_list)
            ax2.set_title('Team Composition by Team')
            ax2.set_ylabel('Number of Players')
            
            plt.tight_layout()
            
            # Save to buffer and display
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            st.session_state.team_visualized = True
            st.session_state.team_viz_data = base64.b64encode(buf.read()).decode()
            plt.close()
        
        # Display visualization from session state
        if st.session_state.team_visualized:
            st.image(f"data:image/png;base64,{st.session_state.team_viz_data}")
        
        # Additional stats
        role_data = {}
        for player in result["team"]:
            role = player["role"]
            if role not in role_data:
                role_data[role] = []
            role_data[role].append(player["total_fp"])
        
        st.markdown("### Points Distribution by Role")
        role_stats = {
            "Role": [],
            "Count": [],
            "Avg Points": [],
            "Max Points": [],
            "Min Points": []
        }
        
        for role, points in role_data.items():
            role_stats["Role"].append(f"{role_icons.get(role, '')} {role}")
            role_stats["Count"].append(len(points))
            role_stats["Avg Points"].append(round(sum(points) / len(points), 1))
            role_stats["Max Points"].append(round(max(points), 1))
            role_stats["Min Points"].append(round(min(points), 1))
        
        stats_df = pd.DataFrame(role_stats)
        st.dataframe(stats_df, hide_index=True)
    
    with tab3:
        # Sort players by total fantasy points
        sorted_players = sorted(result["team"], key=lambda x: -x["total_fp"])
        
        st.markdown("### Player Details")
        for player in sorted_players:
            with st.expander(f"{player['name']} ({player['team']} - {player['role']})"):
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    st.markdown(f"**Role:** {role_icons.get(player['role'], '')} {player['role']}")
                    st.markdown(f"**Team:** {player['team']}")
                    st.markdown(f"**Credit Cost:** 💰 {player['credit']}")
                    if player.get("is_captain"):
                        st.markdown("**Captain:** Yes 👑")
                    elif player.get("is_vice_captain"):
                        st.markdown("**Vice Captain:** Yes 🥈")
                
                with col2:
                    st.markdown("**Fantasy Points**")
                    st.markdown(f"Base Points: {round(player['total_fp'], 1)}")
                    
                    if player.get("is_captain"):
                        st.markdown(f"Captain Bonus: {round(player['total_fp'], 1)} (2x)")
                    elif player.get("is_vice_captain"):
                        st.markdown(f"Vice-Captain Bonus: {round(player['total_fp'] * 0.5, 1)} (1.5x)")
                    
                    st.markdown(f"**Adjusted Points:** {round(player['adjusted_fp'], 1)}")
                    
                    # Display factors if available
                    if "form_factor" in player:
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Form Factor", round(player["form_factor"], 2))
                        with col2:
                            st.metric("Venue Factor", round(player["venue_factor"], 2))
                        with col3:
                            st.metric("Opponent Factor", round(player["opponent_factor"], 2))

# Display info message when no prediction is available
if not st.session_state.prediction_result and not predict_button:
    st.info("👈 Set up your match details and click 'Predict Team' to get started!")
    
    # Example visualization
    st.markdown("### How it works")
    st.write("""
    This app uses machine learning to predict the optimal Dream11 fantasy cricket team for an upcoming match.
    
    1. **Select the teams** and venue for the match
    2. **Customize** the team composition (WK, BAT, AR, BWL)
    3. **Get predictions** based on historical player performance
    4. **Analyze** the recommended team with detailed statistics
    
    The algorithm considers player form, venue factors, opponent matchups, and role constraints to build the optimal team.
    """)