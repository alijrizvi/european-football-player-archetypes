import os
from typing import Any, Dict, List
import numpy as np
import pandas as pd


def _load_player_data() -> pd.DataFrame:
    """Helper function to load parquet dataset with root-level fallbacks."""
    paths = [
        os.path.join("data", "processed", "player_matrix_clean.parquet"),
        os.path.join("data", "processed", "player_pca_matrix.parquet"),
        "player_matrix_clean.parquet",
        "player_pca_matrix.parquet",
    ]
    for path in paths:
        if os.path.exists(path):
            return pd.read_parquet(path)
    raise FileNotFoundError("Could not find player matrix parquet dataset.")


# Unified Cluster Taxonomy Mapping
CLUSTER_TAXONOMY = {
    1: "Final-Third Penetrators / Primary Creators",
    2: "Vertical Ball-Carriers / Transition Engines",
    3: "Direct Wingers / 1v1 Take-On Specialists",
}


def find_similar_clones(player_name: str, top_n: int = 5) -> List[Dict[str, Any]]:
    """Calculates 3D Euclidean distance across PCA coordinates (PCA1, PCA2, PCA3)
    to identify the top N nearest tactical clones for a given player.

    Args:
        player_name: The name of the target player to search for.
        top_n: Number of nearest tactical clone alternatives to return.

    Returns:
        List of dictionaries containing player attributes and vector distances.
    """
    try:
        df = _load_player_data()
    except Exception as e:
        return [{"error": f"Data loading failed: {str(e)}"}]

    # Case-insensitive lookup
    target_matches = df[df["player"].str.lower().str.contains(player_name.lower(), na=False)]
    if target_matches.empty:
        return [{"error": f"Player '{player_name}' was not found in the dataset matrix."}]

    target = target_matches.iloc[0]
    target_vec = np.array([target["PCA1"], target["PCA2"], target["PCA3"]])

    # Vectorized 3D Euclidean distance calculation
    all_vecs = df[["PCA1", "PCA2", "PCA3"]].values
    df_temp = df.copy()
    df_temp["euclidean_distance"] = np.linalg.norm(all_vecs - target_vec, axis=1)

    # Sort by closest distance and exclude the queried player
    clones = (
        df_temp[df_temp["player"] != target["player"]]
        .sort_values("euclidean_distance")
        .head(top_n)
    )

    selected_cols = [
        "player",
        "team",
        "league_name",
        "cluster",
        "euclidean_distance",
        "verticality_index",
        "pitch_tilt_ratio",
    ]
    
    # Filter available columns defensively
    valid_cols = [col for col in selected_cols if col in clones.columns]
    
    return clones[valid_cols].to_dict(orient="records")


def analyze_squad_balance(player_list: List[str]) -> Dict[str, Any]:
    """Evaluates a starting XI or squad list against archetype cluster distributions.
    Identifies cluster over-concentrations and flags missing tactical profiles.

    Args:
        player_list: List of player names comprising the squad.

    Returns:
        Dictionary report detailing archetype distribution, tactical gaps, and squad averages.
    """
    try:
        df = _load_player_data()
    except Exception as e:
        return {"error": f"Data loading failed: {str(e)}"}

    search_players = [p.lower() for p in player_list]
    squad_df = df[df["player"].str.lower().isin(search_players)]
    
    found_names = set(squad_df["player"].str.lower())
    missing_players = list(set(search_players) - found_names)

    if squad_df.empty:
        return {
            "error": "None of the specified players were found in the dataset matrix.",
            "missing_players": missing_players,
        }

    cluster_counts = squad_df["cluster"].value_counts().to_dict()
    present_archetypes = {
        CLUSTER_TAXONOMY.get(c, f"Cluster {c}"): count
        for c, count in cluster_counts.items()
    }

    all_clusters = set(CLUSTER_TAXONOMY.keys())
    missing_cluster_ids = all_clusters - set(cluster_counts.keys())
    missing_archetypes = [
        CLUSTER_TAXONOMY.get(c, f"Cluster {c}") for c in missing_cluster_ids
    ]

    prog_col = "total_prog_actions" if "total_prog_actions" in squad_df.columns else "mean_progressive_action_score"

    squad_report = {
        "squad_size_found": len(squad_df),
        "missing_players_not_in_database": missing_players,
        "archetype_distribution": present_archetypes,
        "tactical_gaps": missing_archetypes,
        "squad_averages": {
            "mean_verticality_index": float(squad_df["verticality_index"].mean()) if "verticality_index" in squad_df.columns else 0.0,
            "mean_pitch_tilt_ratio": float(squad_df["pitch_tilt_ratio"].mean()) if "pitch_tilt_ratio" in squad_df.columns else 0.0,
            "total_progressive_actions": float(squad_df[prog_col].sum()) if prog_col in squad_df.columns else 0.0,
        },
    }

    return squad_report


def get_top_archetype_players(
    cluster_id: int, top_n: int = 5, metric_sort: str = "verticality_index"
) -> List[Dict[str, Any]]:
    """Retrieves the top N highest-performing players within a specific tactical cluster/archetype.

    Args:
        cluster_id: Numeric ID of the tactical cluster archetype (1, 2, or 3).
        top_n: Number of top players to retrieve.
        metric_sort: Primary column metric name to sort players by descending.

    Returns:
        List of dictionaries containing top archetype player profiles.
    """
    try:
        df = _load_player_data()
    except Exception as e:
        return [{"error": f"Data loading failed: {str(e)}"}]

    valid_clusters = sorted(df["cluster"].unique().tolist())
    if cluster_id not in valid_clusters:
        return [{"error": f"Cluster ID {cluster_id} invalid. Must be one of {valid_clusters}."}]

    cluster_df = df[df["cluster"] == cluster_id]

    # Fallback sorting metric if specified column does not exist
    if metric_sort not in cluster_df.columns:
        metric_sort = "verticality_index" if "verticality_index" in cluster_df.columns else cluster_df.columns[0]

    top_players = cluster_df.sort_values(by=metric_sort, ascending=False).head(top_n).copy()
    top_players["archetype_label"] = CLUSTER_TAXONOMY.get(cluster_id, f"Cluster {cluster_id}")

    selected_cols = [
        "player",
        "team",
        "league_name",
        "archetype_label",
        metric_sort,
        "verticality_index",
        "pitch_tilt_ratio",
    ]
    
    # Filter unique available columns
    valid_cols = list(dict.fromkeys([col for col in selected_cols if col in top_players.columns]))

    return top_players[valid_cols].to_dict(orient="records")
