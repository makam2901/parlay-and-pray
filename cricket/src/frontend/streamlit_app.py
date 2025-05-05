# src/frontend/streamlit_app.py
import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
from PIL import Image
import io
import base64
import os
import logging # Added logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- API URLs ---
# Use environment variables, fallback to localhost for local dev
PREDICT_API_URL = os.getenv("API_URL", "http://localhost:8000/predict_team/")
CHAT_API_URL = os.getenv("CHAT_API_URL", "http://localhost:8000/chat/") # Use env var

# --- Page Config ---
st.set_page_config(
    page_title="Dream11 Fantasy Team Predictor",
    page_icon="🏏",
    layout="wide"
)

# --- Constants (Teams, Venues, Colors, Icons) ---
teams = ["CSK", "DC", "GT", "KKR", "LSG", "MI", "PBKS", "RCB", "RR", "SRH"]
venues = [
    "MA Chidambaram Stadium, Chepauk, Chennai", "M.Chinnaswamy Stadium, Bengaluru",
    "Wankhede Stadium, Mumbai", "Arun Jaitley Stadium, Delhi",
    "Rajiv Gandhi International Stadium, Uppal, Hyderabad", "Narendra Modi Stadium, Motera, Ahmedabad",
    "Eden Gardens, Kolkata", "Bharat Ratna Shri Atal Bihari Vajpayee Ekana Cricket Stadium, Lucknow",
    "Barsapara Cricket Stadium, Guwahati", "Punjab Cricket Association IS Bindra Stadium, Mohali, Chandigarh"
]
team_home_venues = {
    "CSK": "MA Chidambaram Stadium, Chepauk, Chennai", "RCB": "M.Chinnaswamy Stadium, Bengaluru",
    "MI": "Wankhede Stadium, Mumbai", "DC": "Arun Jaitley Stadium, Delhi",
    "SRH": "Rajiv Gandhi International Stadium, Uppal, Hyderabad", "GT": "Narendra Modi Stadium, Motera, Ahmedabad",
    "KKR": "Eden Gardens, Kolkata", "LSG": "Bharat Ratna Shri Atal Bihari Vajpayee Ekana Cricket Stadium, Lucknow",
    "RR": "Barsapara Cricket Stadium, Guwahati", # Assuming RR primary home
    "PBKS": "Punjab Cricket Association IS Bindra Stadium, Mohali, Chandigarh"
}
team_colors = {
    "CSK": "#FFFF00", "DC": "#0080FF", "GT": "#00008B", "KKR": "#800080",
    "LSG": "#87CEEB", "MI": "#0000FF", "PBKS": "#FF0000", "RCB": "#FF0000",
    "RR": "#FFC0CB", "SRH": "#FFA500"
}
role_icons = {"WK": "🧤", "BAT": "🏏", "AR": "⚡", "BWL": "🎯"}

# --- Custom CSS ---
st.markdown("""
    <style>
    .main-header { /* Keep header */
        font-size: 2.5rem; font-weight: 700; color: #1E3A8A;
        text-align: center; margin-bottom: 1rem;
    }
    .stChatInputContainer > div > textarea { /* Ensure chat input height is reasonable */
        min-height: 50px !important;
    }
    .stChatMessage { /* Add some spacing to chat messages */
         margin-bottom: 1rem;
    }
     .player-card {
        background-color: #f8f9fa; border-radius: 10px; padding: 10px;
        margin: 5px 0; border-left: 5px solid #4CAF50;
    }
    .captain { border-left: 5px solid #FFD700; background-color: #FFFDE7; }
    .vice-captain { border-left: 5px solid #C0C0C0; background-color: #F5F5F5; }
    </style>
""", unsafe_allow_html=True)

# --- Header ---
st.markdown("<h1 class='main-header'>🏏 Dream11 Fantasy Team Predictor & Chat</h1>", unsafe_allow_html=True)

# --- Session State Initialization ---
if "prediction_result" not in st.session_state:
    st.session_state.prediction_result = None
if "team_visualized" not in st.session_state:
    st.session_state.team_visualized = False
if "team_viz_data" not in st.session_state:
    st.session_state.team_viz_data = None
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hi! Ask me anything about cricket, or use the sidebar to predict a fantasy team."}]

# --- Sidebar (Always Visible) ---
with st.sidebar:
    st.header("Match Setup")
    home_team = st.selectbox("Select Home Team", teams, key="home_team")
    away_team = st.selectbox("Select Away Team", [t for t in teams if t != home_team], key="away_team")

    default_venue_index = 0
    if home_team in team_home_venues:
        default_venue = team_home_venues[home_team]
        try:
            default_venue_index = venues.index(default_venue)
        except ValueError:
            default_venue_index = 0 # Fallback if venue not in list

    venue = st.selectbox("Select Venue", venues, index=default_venue_index, key="venue")

    st.header("Team Composition")
    col1_side, col2_side = st.columns(2) # Renamed to avoid conflict
    with col1_side:
        wk_count = st.number_input("Wicketkeepers (WK)", min_value=1, max_value=4, value=1, key="wk_count")
        bat_count = st.number_input("Batsmen (BAT)", min_value=2, max_value=6, value=4, key="bat_count")
    with col2_side:
        ar_count = st.number_input("All-Rounders (AR)", min_value=1, max_value=4, value=2, key="ar_count")
        bwl_count = st.number_input("Bowlers (BWL)", min_value=2, max_value=6, value=4, key="bwl_count")

    total_players = wk_count + bat_count + ar_count + bwl_count
    if total_players != 11:
        st.warning(f"Total players: {total_players}/11. Adjust roles for exactly 11 players.")
    else:
        st.success(f"Total players: {total_players}/11 ✓")

    with st.expander("Advanced Options"):
        # Corrected slider uses floats
        max_credits = st.slider("Max Credits", min_value=80.0, max_value=100.0, value=100.0, step=0.5, key="max_credits")
        max_per_team = st.slider("Max Players per Team", min_value=4, max_value=7, value=7, step=1, key="max_per_team")

    predict_button = st.button("🔮 Predict Team", type="primary", disabled=(total_players != 11), key="predict_button")

    # Add button to clear results and go back to chat
    if st.session_state.prediction_result is not None:
        if st.button("Clear Results / Back to Chat"):
            st.session_state.prediction_result = None
            st.session_state.team_visualized = False # Reset viz
            st.session_state.team_viz_data = None
            st.rerun()

# --- Handle Prediction Button Click Logic ---
if predict_button:
    if total_players != 11:
        st.error("Please ensure the total number of players is exactly 11.")
    else:
        with st.spinner("Predicting optimal team..."):
            request_data = {
                "home_team": home_team,
                "away_team": away_team,
                "venue": venue,
                "wk_count": int(wk_count),
                "bat_count": int(bat_count),
                "ar_count": int(ar_count),
                "bwl_count": int(bwl_count),
                "max_credits": float(max_credits), # Send as float
                "max_per_team": int(max_per_team)
            }
            try:
                if not PREDICT_API_URL:
                    st.error("Prediction API URL is not configured!")
                else:
                    logging.info(f"Sending prediction request to: {PREDICT_API_URL}")
                    logging.debug(f"Prediction request data: {request_data}")
                    response = requests.post(PREDICT_API_URL, json=request_data)
                    response.raise_for_status()
                    result_data = response.json()
                    logging.info("Prediction successful.")
                    logging.debug(f"Prediction response data keys: {result_data.keys()}")

                    # Process prediction result
                    team_raw = result_data.get("team_raw")
                    team_summary = result_data.get("team_summary")

                    if not team_raw or not team_summary:
                        st.error("Prediction response is missing expected data ('team_raw' or 'team_summary').")
                        st.session_state.prediction_result = None
                    else:
                        captain_name = team_summary.get("captain")
                        vice_captain_name = team_summary.get("vice_captain")

                        # Add flags/factors to raw data
                        for player in team_raw:
                            player["is_captain"] = (player.get("name") == captain_name)
                            player["is_vice_captain"] = (player.get("name") == vice_captain_name)

                        result_data["team_raw"] = team_raw
                        st.session_state.prediction_result = result_data
                        st.session_state.team_visualized = False # Reset viz flag
                        st.rerun() # Rerun to display results

            except requests.exceptions.HTTPError as http_err:
                 logging.error(f"HTTP error during prediction: {http_err}", exc_info=True)
                 error_detail = str(http_err)
                 try:
                     error_detail = http_err.response.json().get("detail", str(http_err))
                 except json.JSONDecodeError: pass
                 st.error(f"❌ Failed to predict team (HTTP {http_err.response.status_code}): {error_detail}")
                 st.session_state.prediction_result = None # Keep chat visible
            except requests.exceptions.RequestException as req_err:
                 logging.error(f"Request error during prediction: {req_err}", exc_info=True)
                 st.error(f"❌ Failed to connect to prediction API: {req_err}")
                 st.info(f"Is the backend running at {PREDICT_API_URL}?")
                 st.session_state.prediction_result = None # Keep chat visible
            except Exception as e:
                 logging.error(f"Unexpected error during prediction processing: {e}", exc_info=True)
                 st.error(f"❌ An unexpected error occurred: {str(e)}")
                 st.session_state.prediction_result = None # Keep chat visible


# --- Main Area Layout (Conditional) ---
# **** LAYOUT CHANGE ****
# Removed the top-level col_main, col_results columns.
# Now we display EITHER chat OR results based on session state.

if st.session_state.prediction_result is None:
    # --- STATE 1: Show Chat Interface ---
    st.subheader("Chat with Cricket Bot")

    # Display chat messages from history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # React to user input
    if prompt := st.chat_input("Ask me about cricket..."):
        # Display user message
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Prepare request for backend chat
        chat_request_data = {
            "message": prompt,
            "history": [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages[:-1]]
        }

        # Call backend chat endpoint
        try:
            logging.info(f"Sending chat request to: {CHAT_API_URL}")
            if not CHAT_API_URL or "localhost" in CHAT_API_URL:
                 logging.warning("Chat API URL appears to be using localhost or is not set. Ensure CHAT_API_URL env var is correctly set in deployment.")
                 # Optionally display a less technical warning to the user if needed
                 # st.warning("Chat service might not be configured correctly for this environment.")

            response = requests.post(CHAT_API_URL, json=chat_request_data)
            response.raise_for_status()
            chat_response_data = response.json()
            assistant_reply = chat_response_data.get("reply")
            backend_history = chat_response_data.get("history", [])
            st.session_state.messages = [{"role": m["role"], "content": m["content"]} for m in backend_history]

            # Display assistant response
            if st.session_state.messages and st.session_state.messages[-1]["role"] == "assistant":
                 with st.chat_message("assistant"):
                     st.markdown(st.session_state.messages[-1]["content"])
            else:
                 logging.warning("Backend did not return expected history format. Displaying direct reply.")
                 with st.chat_message("assistant"):
                     st.markdown(assistant_reply if assistant_reply else "Sorry, I couldn't get a response.")
                 if assistant_reply and (not st.session_state.messages or st.session_state.messages[-1]["role"] == "user"):
                      st.session_state.messages.append({"role": "assistant", "content": assistant_reply})

        except requests.exceptions.RequestException as e:
            logging.error(f"Chat API request failed: {e}", exc_info=True)
            st.error(f"❌ Failed to connect to the chat bot: {e}")
            error_msg = f"Sorry, I couldn't connect to the chat service. Please check the backend status and ensure the CHAT_API_URL environment variable is set correctly. (Attempted: {CHAT_API_URL})"
            st.session_state.messages.append({"role": "assistant", "content": error_msg})
            # Display the error in the chat interface
            with st.chat_message("assistant"):
                st.markdown(error_msg)
        except Exception as e:
            logging.error(f"Error processing chat response: {e}", exc_info=True)
            st.error(f"❌ An error occurred: {str(e)}")
            error_msg = f"Sorry, an unexpected error occurred while chatting: {e}."
            st.session_state.messages.append({"role": "assistant", "content": error_msg})
            # Display the error in the chat interface
            with st.chat_message("assistant"):
                st.markdown(error_msg)

else:
    # --- STATE 2: Show Prediction Results ---
    st.subheader("Prediction Results")
    result = st.session_state.prediction_result
    team_raw_data = result.get("team_raw", [])
    team_summary_data = result.get("team_summary", {})

    if not team_raw_data:
        st.warning("No team data available in the prediction result.")
        # Button to go back is now handled by the sidebar button
    else:
        # Create tabs for different views
        tab1, tab2, tab3 = st.tabs(["Team Selection", "Team Statistics", "Player Details"])

        with tab1:
            st.markdown(f"### {home_team} vs {away_team}")
            st.markdown(f"**Venue:** {venue}")

            home_players = [p for p in team_raw_data if p.get("team") == home_team]
            away_players = [p for p in team_raw_data if p.get("team") == away_team]

            col_t1, col_t2 = st.columns(2) # Columns within tab are fine

            # Function to display player card
            def display_player_card(player):
                cap = " (C)" if player.get("is_captain") else ""
                vc = " (VC)" if player.get("is_vice_captain") else ""
                role_str = player.get('role', 'N/A')
                role_display = f"{role_icons.get(role_str, '')} {role_str}"
                credit_val = player.get('credit', 0)
                fp_val = player.get('adjusted_fp', player.get('total_fp', 0))

                card_class = "player-card"
                if player.get("is_captain"): card_class += " captain"
                elif player.get("is_vice_captain"): card_class += " vice-captain"

                st.markdown(f"""
                <div class='{card_class}'>
                    <b>{player.get('name', 'Unknown')}{cap}{vc}</b><br>
                    {role_display} | 💰 {credit_val:.1f} | ⭐ {fp_val:.1f} pts
                </div>
                """, unsafe_allow_html=True)

            with col_t1:
                st.markdown(f"#### {home_team} ({len(home_players)} players)")
                for player in sorted(home_players, key=lambda x: (-x.get("is_captain", False), -x.get("is_vice_captain", False), -x.get('adjusted_fp', x.get('total_fp', 0)))):
                     display_player_card(player)

            with col_t2:
                st.markdown(f"#### {away_team} ({len(away_players)} players)")
                for player in sorted(away_players, key=lambda x: (-x.get("is_captain", False), -x.get("is_vice_captain", False), -x.get('adjusted_fp', x.get('total_fp', 0)))):
                     display_player_card(player)

            # Summary statistics
            st.markdown("### Team Summary")
            sum_col1, sum_col2, sum_col3, sum_col4 = st.columns(4) # Columns within tab fine

            total_credits = team_summary_data.get('total_credits', 0)
            adj_total_points = sum(p.get('adjusted_fp', p.get('total_fp', 0)) for p in team_raw_data)
            captain = team_summary_data.get("captain", "Unknown")
            vc = team_summary_data.get("vice_captain", "Unknown")

            with sum_col1: st.metric("Total Credits", f"{total_credits:.1f}/{max_credits}")
            with sum_col2: st.metric("Total Adj. Points", f"{adj_total_points:.1f}")
            with sum_col3: st.metric("Captain", captain)
            with sum_col4: st.metric("Vice-Captain", vc)

        with tab2:
            # Visualizations
            if not st.session_state.team_visualized or not st.session_state.team_viz_data:
                try:
                    logging.info("Generating team visualizations...")
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
                    # Role distribution
                    role_counts = pd.Series([p.get('role') for p in team_raw_data]).value_counts().reindex(role_icons.keys(), fill_value=0)
                    ax1.bar(role_counts.index, role_counts.values, color=['skyblue', 'lightgreen', 'gold', 'salmon'])
                    ax1.set_title('Team Composition by Role')
                    ax1.set_ylabel('Number of Players')
                    # Team distribution
                    team_counts = pd.Series([p.get('team') for p in team_raw_data]).value_counts().reindex([home_team, away_team], fill_value=0)
                    team_colors_list = [team_colors.get(t, "#888888") for t in team_counts.index]
                    ax2.bar(team_counts.index, team_counts.values, color=team_colors_list)
                    ax2.set_title('Team Composition by Team')
                    ax2.set_ylabel('Number of Players')

                    plt.tight_layout()
                    buf = io.BytesIO()
                    plt.savefig(buf, format='png', bbox_inches='tight')
                    buf.seek(0)
                    st.session_state.team_viz_data = base64.b64encode(buf.read()).decode()
                    st.session_state.team_visualized = True
                    plt.close(fig)
                    logging.info("Visualizations generated.")
                except Exception as viz_err:
                    logging.error(f"Error generating visualization: {viz_err}", exc_info=True)
                    st.warning("Could not generate team visualizations.")

            if st.session_state.team_visualized and st.session_state.team_viz_data:
                st.image(f"data:image/png;base64,{st.session_state.team_viz_data}")
            else:
                st.info("Visualizations could not be generated or are pending.")

            # Points Distribution by Role
            st.markdown("### Points Distribution by Role")
            role_data = {}
            # ... (logic as before) ...
            for player in team_raw_data:
                role = player.get("role")
                points = player.get('adjusted_fp', player.get('total_fp', 0)) # Use adjusted points
                if role:
                    if role not in role_data: role_data[role] = []
                    role_data[role].append(points)
            role_stats_list = []
            for role, points_list in role_data.items():
                if points_list:
                    role_stats_list.append({
                        "Role": f"{role_icons.get(role, '')} {role}", "Count": len(points_list),
                        "Avg Points": round(sum(points_list) / len(points_list), 1),
                        "Max Points": round(max(points_list), 1), "Min Points": round(min(points_list), 1)
                    })
            if role_stats_list:
                 stats_df = pd.DataFrame(role_stats_list)
                 st.dataframe(stats_df, hide_index=True)
            else:
                 st.info("No role point data available.")


        with tab3:
            st.markdown("### Player Details")
            sorted_players = sorted(team_raw_data, key=lambda x: -x.get('adjusted_fp', x.get('total_fp', 0)))

            for player in sorted_players:
                player_name = player.get('name', 'Unknown')
                player_team = player.get('team', 'N/A')
                player_role = player.get('role', 'N/A')
                with st.expander(f"{player_name} ({player_team} - {player_role})"):
                    p_col1, p_col2 = st.columns([1, 2]) # Level 1 nesting (OK)

                    with p_col1:
                        st.markdown(f"**Role:** {role_icons.get(player_role, '')} {player_role}")
                        st.markdown(f"**Team:** {player_team}")
                        st.markdown(f"**Credit Cost:** 💰 {player.get('credit', 0):.1f}")
                        if player.get("is_captain"): st.markdown("**Captain:** Yes 👑")
                        elif player.get("is_vice_captain"): st.markdown("**Vice Captain:** Yes 🥈")

                    with p_col2:
                        st.markdown("**Fantasy Points Breakdown**")
                        base_fp = player.get('total_fp', 0)
                        adj_fp = player.get('adjusted_fp', base_fp)
                        st.markdown(f"Base Points (Predicted): {base_fp:.1f}")

                        # Display factors if available
                        if "form_factor" in player and player["form_factor"] is not None:
                             # **** NESTED COLUMN FIX ****
                             # Removed the st.columns(3) here and placed st.metric directly
                            st.metric("Form Factor", f"{player.get('form_factor', 1.0):.2f}", delta=f"{((player.get('form_factor', 1.0)-1)*100):.1f}%", delta_color="off")
                            st.metric("Venue Factor", f"{player.get('venue_factor', 1.0):.2f}", delta=f"{((player.get('venue_factor', 1.0)-1)*100):.1f}%", delta_color="off")
                            st.metric("Opponent Factor", f"{player.get('opponent_factor', 1.0):.2f}", delta=f"{((player.get('opponent_factor', 1.0)-1)*100):.1f}%", delta_color="off")
                        else:
                            st.caption("Contextual factors not available for this player.")

                        st.markdown(f"**Adjusted Points (Used for Selection): {adj_fp:.1f}**")

                        # Show C/VC bonus
                        final_points = adj_fp
                        if player.get("is_captain"):
                            final_points *= 2
                            st.markdown(f"Captain Bonus: +{adj_fp:.1f} pts (Total: {final_points:.1f})")
                        elif player.get("is_vice_captain"):
                            final_points *= 1.5
                            st.markdown(f"Vice-Captain Bonus: +{adj_fp * 0.5:.1f} pts (Total: {final_points:.1f})")