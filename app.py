import json
import os
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import google.generativeai as genai
from dotenv import load_dotenv

# Load local environment variables from .env if present
load_dotenv()

# Import local modular tools
from src.tools import (
    find_similar_clones,
    analyze_squad_balance,
    get_top_archetype_players,
)
from src.utils import initialize_gemini_client

# -----------------------------------------------------------------------------
# 1. PAGE CONFIGURATION & THEME
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="European Tactical Decision Engine",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("⚽ European Tactical Decision Engine & Archetype Copilot")
st.caption("3D Spatial Vector Archetype Analytics & Grounded AI Scouting Agent")

# -----------------------------------------------------------------------------
# 2. GEMINI AI STUDIO CONFIGURATION
# -----------------------------------------------------------------------------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY")
api_ready = False

if GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        api_ready = True
    except Exception as e:
        st.warning(f"⚠️ Error configuring Gemini API: {e}")

# Store System Instructions as a valid Python multi-line string
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
    parquet_path = os.path.join("data", "processed", "player_pca_matrix.parquet")
    json_path = os.path.join("data", "processed", "archetype_profiles.json")

    # Fallback paths for flat root testing
    if not os.path.exists(parquet_path):
        parquet_path = "player_pca_matrix.parquet"
    if not os.path.exists(json_path):
        json_path = "archetype_profiles.json"

    df = pd.read_parquet(parquet_path)
    with open(json_path, "r") as f:
        profiles = json.load(f)

    return df, profiles


try:
    df_players, archetype_profiles = load_app_data()
except Exception as e:
    st.error(f"Error loading project artifacts: {e}")
    st.stop()

# Define Cluster taxonomy labels
CLUSTER_NAMES = {
    0: "Tempo-Setting Recyclers",
    1: "Vertical Ball-Carriers / Transition Engines",
    2: "Final-Third Penetrators / Primary Creators",
    3: "Direct Wingers / 1v1 Take-On Specialists",
    4: "Defensive Anchor / Ball Recovery Specialists",
}

# -----------------------------------------------------------------------------
# 4. SIDEBAR CONTROLS
# -----------------------------------------------------------------------------
st.sidebar.header("🕹️ Tactical Engine Controls")

selected_league = st.sidebar.multiselect(
    "Filter Leagues",
    options=df_players["league_name"].unique(),
    default=df_players["league_name"].unique(),
)

filtered_df = df_players[df_players["league_name"].isin(selected_league)]

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
    selected_cluster_id = st.selectbox(
        "Select Tactical Cluster Archetype",
        options=list(CLUSTER_NAMES.keys()),
        format_func=lambda x: f"Cluster {x}: {CLUSTER_NAMES[x]}",
    )

    cluster_info = archetype_profiles.get(str(selected_cluster_id), {})
    if cluster_info:
        pcts = cluster_info["percentiles"]
        categories = list(pcts.keys())
        values = list(pcts.values())

        fig_radar = go.Figure()
        fig_radar.add_trace(
            go.Scatterpolar(
                r=values + [values[0]],
                theta=categories + [categories[0]],
                fill="toself",
                name=f"Cluster {selected_cluster_id}",
            )
        )
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
            showlegend=False,
            height=500,
            title=f"Percentile Benchmark Profile — {CLUSTER_NAMES[selected_cluster_id]}",
        )
        st.plotly_chart(fig_radar, use_container_width=True)

# -----------------------------------------------------------------------------
# TAB 3: AI COPILOT INTERFACE (GEMINI INTEGRATION)
# -----------------------------------------------------------------------------
with tab3:
    st.subheader("🤖 Grounded Decision Intelligence Copilot")
    st.markdown(
        "Ask tactical questions, request 3D Euclidean player replacements, or evaluate squad archetype balance."
    )

    if not api_ready:
        st.warning(
            "⚠️ Gemini API Key not detected. Please set `GEMINI_API_KEY` in Streamlit Cloud Secrets "
            "to activate the AI Copilot layer."
        )
    else:
        # Initialize Gemini Model with System Instructions & Function Calling Tools
        model = genai.GenerativeModel(
            model_name="gemini-1.5-pro",
            generation_config={
                "temperature": 0.2,
                "top_p": 0.95,
                "max_output_tokens": 1024,
            },
            system_instruction=SYSTEM_INSTRUCTION,
            tools=[find_similar_clones, analyze_squad_balance, get_top_archetype_players],
        )

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
                        chat = model.start_chat(enable_automatic_function_calling=True)
                        response = chat.send_message(user_query)
                        st.markdown(response.text)
                        st.session_state.messages.append(
                            {"role": "assistant", "content": response.text}
                        )
                    except Exception as err:
                        st.error(f"Copilot Error: {err}")
