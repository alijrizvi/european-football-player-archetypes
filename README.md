# ⚽ European Footballer Archetype | Tactical Decision Engine

An End-to-End Decision Intelligence System & AI Copilot transforming raw European football tracking data into actionable 3D spatial vector embeddings, cluster profiles, and automated scouting replacements.

## Medium Case Study: 

<*European Footballers Player Arcehtype Design*>

---

# 📌 Executive Overview

Modern football scouting requires moving beyond traditional superficial stats into multidimensional spatial vector analytics. This decision intelligence engine processes progressive actions, pitch tilt, and horizontal displacement across European football leagues using Dimensionality Reduction (PCA) and Hierarchical Agglomerative Clustering (HAC).

Integrated with Google Gemini 2.5 Flash via dynamic function calling tools, the system acts as an AI Tactical Scouting Director capable of executing 3D Euclidean player similarity searches, squad balance audits, and role-based leaderboards in real time.

---

# 🔑 Key Features & Technical Architecture

## 🌌 3D Spatial Vector Embedding (PCA): 

Reduces complex multi-variable spatial performance metrics into a 3D Euclidean vector space for intuitive spatial clustering.

## 📊 Unsupervised Cluster Profiling (HAC): Segments players into distinct tactical archetypes:

* Cluster 1: Final-Third Penetrators / Primary Creators

* Cluster 2: Vertical Ball-Carriers / Transition Engines

* Cluster 3: Direct Wingers / 1v1 Take-On Specialists

## 🎯 Tactical Archetype Radar Profiler: 

Interactive Percentile Radar benchmarks comparing Progressive Action Scores, Distance Covered, Pitch Tilt, and Verticality Index.

## 🔍 Euclidean Player Vector Clone Finder: 

Nearest-neighbor similarity engine locating low-cost tactical equivalents to target players across European leagues.

## 🤖 Grounded AI Tactical Copilot: 

LLM decision interface integrated via google-genai SDK using tool calling to ground responses directly on dataset outputs.

---

# 🛠️ Tech Stack & Methods

## Analytics & Data Processing: 

Python 3.10+, Pandas, NumPy, Scikit-Learn (PCA, HAC Clustering), PyArrow (Parquet)

## Visualization Engine: 

Plotly Express & Plotly Graph Objects (3D Scatter, Radar Profiler, Interactive Visuals)

## Web & UI Framework: 

Streamlit Dashboard Engine

## LLM & Agentic AI Layer: 

Google GenAI SDK (google-genai), Gemini 2.5 Flash, Tool/Function Calling

---

# 📂 Repository Structure

├── data/
│   └── processed/
│       ├── player_matrix_clean.parquet    # Standardized player spatial metrics
│       └── archetype_profiles.json        # Pre-computed cluster percentile benchmarks
├── src/
│   ├── __init__.py
│   ├── tools.py                           # Native Python tools for Gemini Function Calling
│   └── utils.py                           # API client initializers & environment loaders
├── app.py                                 # Core Streamlit multi-tab decision application
├── requirements.txt                       # Version-pinned dependencies
├── .env.example                           # Template for API keys
└── README.md                              # Technical repository documentation

---

# 🚀 Quickstart & Local Installation

## 1. Clone the Repository

git clone https://github.com/alijrizvi/end-to-end_decision_intelligence_copilot_llm.git
cd end-to-end_decision_intelligence_copilot_llm

## 2. Set Up Virtual Environment & Dependencies

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

## 3. Environment Configuration

`Create a .env file in the root directory:`

GEMINI_API_KEY="your_google_gemini_api_key_here"

## 4. Launch the Streamlit App

streamlit run app.py

---

# 🔗 Key Links & References

## Live Demo: 

Streamlit Cloud App (Add your live link here)

## Author Profile: 

Ali J. Rizvi — GitHub | LinkedIn | Goodreads

# 📜 License

Distributed under the MIT License. See LICENSE for more information.
