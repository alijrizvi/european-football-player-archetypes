import json
import os
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv
from google import genai
from google.genai import types

# Load local environment variables from .env if present
load_dotenv()

# Import local modular tools
from src.tools import (
    analyze_squad_balance,
    find_similar_clones,
    get_top_archetype_players,
)

# -----------------------------------------------------------------------------
# 1. PAGE CONFIGURATION & THEME
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Copilot: European Footballer Archetypes",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("⚽ European Footballers Archetype | Tactical Decision Engine")
st.caption("➡️ Your Personal LLM Assistant: Transforming Us Football Fans into Tactical Decision Scientists!")

# -----------------------------------------------------------------------------
# 2. GEMINI AI CLIENT CONFIGURATION
# -----------------------------------------------------------------------------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY")
client = None
api_ready = False

if GEMINI_API_KEY:
    try:
        client = genai.Client(api_key=GEMINI_API_KEY)
        api_ready = True
    except Exception as e:
        st.warning(f"⚠️ Error configuring Gemini API client: {e}")

SYSTEM_INSTRUCTION = """
You are a Senior Decision Scientist and Tactical Scouting Director for European football.
You reason over 3D spatial vector embeddings, principal component scores, and HAC cluster archetypes.

When asked to find similar players, analyze squad lineups, or pull top players for a role, 
ALWAYS use your provided function calling tools (find_similar_clones, analyze_squad_balance, get_top_archetype_players)
to ground your answers strictly on the mathematical outputs of the dataset.

Maintain a concise, objective, analytical tone.
"""

# -----------------------------------------------------------------------------
# 3. DATA LOADING & INITIALIZATION
# -----------------------------------------------------------------------------
@st.cache_data
def load_app_data():
    parquet_path = os.path.join("data", "processed", "player_matrix_clean.parquet")
    json_path = os.path.join("data", "processed", "archetype_profiles.json")

    # Fallback paths for flat root testing
    if not os.path.exists(parquet_path):
        parquet_path = "player_matrix_clean.parquet"
    if not os.path.exists(json_path):
        json_path = "archetype_profiles.json"

    df = pd.read_parquet(parquet_path)
    with open(json_path, "r", encoding="utf-8") as f:
        profiles = json.load(f)

    return df, profiles


try:
    df_players, archetype_profiles = load_app_data()
except Exception as e:
    st.error(f"Error loading project artifacts: {e}")
    st.stop()

# Define Cluster taxonomy labels
CLUSTER_NAMES = {
    1: "〽️ Final-Third Penetrators / Primary Creators",
    2: "⤴️ Vertical Ball-Carriers / Transition Engines",
    3: "⚡️ Direct Wingers / 1v1 Take-On Specialists (Boy - Aren't They Everyone's Favorites?! 🔥)"
}

# -----------------------------------------------------------------------------
# 4. SIDEBAR CONTROLS
# -----------------------------------------------------------------------------
st.sidebar.header("🕹️ Tactical Engine Controls")

selected_league = st.sidebar.multiselect(
    "Filter Leagues",
    options=df_players["league_name"].str.replace("_", " ").str.title().unique(),
    default=df_players["league_name"].str.replace("_", " ").str.title().unique(),
)

filtered_df = df_players[df_players["league_name"].str.replace("_", " ").str.title().isin(selected_league)]

st.sidebar.markdown("---")
st.sidebar.subheader("📊 Dataset Telemetry")
st.sidebar.metric("Total Players Analyzed", len(filtered_df))
st.sidebar.metric("Tactical Archetypes", len(CLUSTER_NAMES))

# -----------------------------------------------------------------------------
# 5. MAIN INTERFACE: TABS
# -----------------------------------------------------------------------------
tab1, tab2, tab3 = st.tabs(
    ["🌌 3D Archetype Space", "📊 Cluster Profiles", "🤖 AI Tactical Copilot"]
)

# -----------------------------------------------------------------------------
# TAB 1: 3D SPATIAL VECTOR PLOT
# -----------------------------------------------------------------------------
with tab1:
    st.subheader("3D PCA Player Archetype Space")
    st.markdown(
        "Interactive spatial projection of player profiles across Principal Components. "
        "Players with similar tactical signatures cluster closer together in Euclidean space."
    )

    filtered_df_copy = filtered_df.copy()
    filtered_df_copy["Archetype"] = filtered_df_copy["cluster"].map(CLUSTER_NAMES)

    fig_3d = px.scatter_3d(
        filtered_df_copy,
        x="PCA1",
        y="PCA2",
        z="PCA3",
        color="Archetype",
        hover_name="player",
        hover_data=["team", "league_name", "verticality_index", "pitch_tilt_ratio"],
        title="European Footballer Spatial Vector Distribution",
        opacity=0.8,
        height=700,
    )
    fig_3d.update_layout(margin=dict(l=0, r=0, b=0, t=40))
    st.plotly_chart(fig_3d, use_container_width=True)

# -----------------------------------------------------------------------------
# TAB 2: CLUSTER PROFILE RADAR CHARTS
# -----------------------------------------------------------------------------
with tab2:
    st.subheader("Archetype Tactical Radar Benchmarks")

    # Map raw JSON / parquet column keys to clean, professional display titles
    METRIC_MAP = {
        "mean_progressive_action_score": "Progressive Action Score",
        "total_displacement_hypotenuse": "Total Distance Covered",
        "total_delta_x": "Total Distance Covered",  # Fallback key mapping
        "prog_action_ratio": "Progressive Action %",
        "verticality_index": "Verticality Index",
        "pitch_tilt_ratio": "Pitch Tilt",
        "mean_delta_x": "Forward Yards Gained"      # Fallback key mapping
    }

    # Custom Cluster Palette
    CLUSTER_COLORS = {
        1: "#1f77b4",  # Blue
        2: "#ff7f0e",  # Orange
        3: "#2ca02c"   # Green
    }

    # Dropdown with individual clusters + "Compare All 3 Clusters Together"
    dropdown_options = [
        "Compare All 3 Clusters Together",
        1,
        2,
        3
    ]

    selected_option = st.selectbox(
        "Select Tactical Cluster Archetype",
        options=dropdown_options,
        format_func=lambda x: (
            "Compare All 3 Clusters Together"
            if x == "Compare All 3 Clusters Together"
            else f"Cluster {x}: {CLUSTER_NAMES[x]}"
        ),
        index=0  # Default to showing all 3 together
    )

    st.markdown("""
- **Progressive Action Score**: Evaluates the overall quality and forward threat level of a player's progressive passes, carries, and dribbles based on historical success models.
- **Total Distance Covered**: Represents the net horizontal spatial displacement (the direct hypotenuse vector distance) a player achieved via forward actions on the pitch.
- **Progressive Action %**: The percentage of a player's total ball involvements that successfully pushed the team forward, highlighting how direct or progress-minded they are.
- **Verticality Index**: Measures a player's vertical directness by calculating the ratio of forward yardage gained against their total overall distance traveled.
- **Pitch Tilt (Attacking Share)**: Calculates the share of a player's progressive involvements that successfully entered or occurred within the opponent's final third (last 20% of the field).
""")

    fig_radar = go.Figure()

    if selected_option == "Compare All 3 Clusters Together":
        # Loop through all clusters to build an overlaid multi-trace radar chart
        for c_id in [1, 2, 3]:
            cluster_info = archetype_profiles.get(str(c_id), {})
            if cluster_info and "percentiles" in cluster_info:
                pcts = cluster_info["percentiles"]
                # Clean up display keys
                categories = [METRIC_MAP.get(k, k.replace("_", " ").title()) for k in pcts.keys()]
                values = list(pcts.values())

                fig_radar.add_trace(
                    go.Scatterpolar(
                        r=values + [values[0]],
                        theta=categories + [categories[0]],
                        fill="toself",
                        name=f"Cluster {c_id}",
                        line=dict(color=CLUSTER_COLORS[c_id], width=2),
                        opacity=0.45
                    )
                )

        radar_title = "Percentile Benchmark Profile — All 3 Tactical Archetypes Overlay"

    else:
        # Plot single selected cluster
        c_id = selected_option
        cluster_info = archetype_profiles.get(str(c_id), {})
        if cluster_info and "percentiles" in cluster_info:
            pcts = cluster_info["percentiles"]
            categories = [METRIC_MAP.get(k, k.replace("_", " ").title()) for k in pcts.keys()]
            values = list(pcts.values())

            fig_radar.add_trace(
                go.Scatterpolar(
                    r=values + [values[0]],
                    theta=categories + [categories[0]],
                    fill="toself",
                    name=f"Cluster {c_id}",
                    line=dict(color=CLUSTER_COLORS[c_id], width=3),
                    fillcolor=CLUSTER_COLORS[c_id],
                    opacity=0.5
                )
            )

        radar_title = f"Percentile Benchmark Profile — {CLUSTER_NAMES[c_id]}"

    fig_radar.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                ticksuffix="%"
            )
        ),
        showlegend=True if selected_option == "Compare All 3 Clusters Together" else False,
        height=550,
        title=radar_title,
        template="plotly_white"
    )

    st.plotly_chart(fig_radar, use_container_width=True)
    
# -----------------------------------------------------------------------------
# TAB 3: AI COPILOT INTERFACE (GEMINI INTEGRATION)
# -----------------------------------------------------------------------------
with tab3:
    st.subheader("🤖 Your Personal Assistant Manager")
    st.markdown(
        "Grounded in this project's data, analytics, and Machine Learning models. Consider this one of your team's *complementary* Data Scientists! "
        "Ask tactical questions, request 3D Euclidean player replacements, or evaluate squad archetype balance."
    )

    if not api_ready or client is None:
        st.warning(
            "⚠️ Gemini API Key not detected. Please set `GEMINI_API_KEY` in Streamlit Cloud Secrets "
            "or your local environment to activate the AI Copilot layer."
        )
    else:
        # Initialize Chat Session State
        if "messages" not in st.session_state:
            st.session_state.messages = [
                {
                    "role": "assistant",
                    "content": "Salam! I am your Tactical Decision Copilot. Ask me for player replacements, archetype leaderboards, or squad balance analysis.",
                }
            ]

        # Render chat history
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        # User Prompt Input
        if user_query := st.chat_input("e.g., 'Find me 3 players similar to Luka Modric'"):
            st.session_state.messages.append({"role": "user", "content": user_query})
            with st.chat_message("user"):
                st.markdown(user_query)

            with st.chat_message("assistant"):
                with st.spinner("Reasoning over 3D vector space and querying tools..."):
                    try:
                        # Configure tools and system instructions
                        config = types.GenerateContentConfig(
                            system_instruction=SYSTEM_INSTRUCTION,
                            temperature=0.2,
                            top_p=0.95,
                            max_output_tokens=1024,
                            tools=[
                                find_similar_clones,
                                analyze_squad_balance,
                                get_top_archetype_players,
                            ],
                        )

                        # Reconstruct conversation history for multi-turn chat
                        contents = []
                        for m in st.session_state.messages:
                            role = "user" if m["role"] == "user" else "model"
                            contents.append(
                                types.Content(
                                    role=role,
                                    parts=[types.Part.from_text(text=m["content"])],
                                )
                            )

                        response = client.models.generate_content(
                            model="gemini-3.8-flash",
                            contents=contents,
                            config=config,
                        )

                        response_text = response.text or "I completed the tool analysis for your tactical query."
                        st.markdown(response_text)
                        st.session_state.messages.append(
                            {"role": "assistant", "content": response_text}
                        )
                    except Exception as err:
                        st.error(f"Copilot Error: {err}")
