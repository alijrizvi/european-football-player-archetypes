import json
import numpy as np
import pandas as pd


def find_similar_clones(player_name: str, top_n: int = 5) -> pd.DataFrame:
    """Calculates 3D Euclidean distance across PCA coordinates (PCA1, PCA2, PCA3)
    to identify the top N nearest tactical clones for a given player.
    """
    parquet_path = "data/processed/player_pca_matrix.parquet"
    if not pd.io.common.file_exists(parquet_path):
        parquet_path = "player_pca_matrix.parquet"

    df = pd.read_parquet(parquet_path)

    # Case-insensitive lookup
    target_matches = df[df["player"].str.lower().str.contains(player_name.lower())]
    if target_matches.empty:
        raise ValueError(
            f"Player '{player_name}' was not found in the dataset matrix."
        )

    target = target_matches.iloc[0]
    target_vec = np.array([target["PCA1"], target["PCA2"], target["PCA3"]])

    # Vectorized 3D Euclidean distance calculation
    all_vecs = df[["PCA1", "PCA2", "PCA3"]].values
    df["euclidean_distance"] = np.linalg.norm(all_vecs - target_vec, axis=1)

    # Sort by closest distance and exclude the queried player
    clones = (
        df[df["player"] != target["player"]]
        .sort_values("euclidean_distance")
        .head(top_n)
    )

    return clones[
        [
            "player",
            "team",
            "league_name",
            "cluster",
            "euclidean_distance",
            "verticality_index",
            "pitch_tilt_ratio",
        ]
    ]


def analyze_squad_balance(player_list: list) -> dict:
    """Evaluates a starting XI or squad list against archetype cluster distributions.
    Identifies cluster over-concentrations and flags missing tactical profiles.
    """
    parquet_path = "data/processed/player_pca_matrix.parquet"
    if not pd.io.common.file_exists(parquet_path):
        parquet_path = "player_pca_matrix.parquet"

    df = pd.read_parquet(parquet_path)

    cluster_taxonomy = {
        0: "Tempo-Setting Recyclers (High Volume / Lateral Passers)",
        1: "Vertical Ball-Carriers / Transition Engines (High Carries / High Delta X)",
        2: "Final-Third Penetrators / Primary Creators (High Pitch Tilt / Box Entries)",
        3: "Direct Wingers / 1v1 Take-On Specialists (High Dribbles / Wide Y Displacement)",
        4: "Defensive Anchor / Ball Recovery Specialists",
    }

    squad_df = df[df["player"].str.lower().isin([p.lower() for p in player_list])]
    missing_players = set([p.lower() for p in player_list]) - set(
        squad_df["player"].str.lower()
    )

    cluster_counts = squad_df["cluster"].value_counts().to_dict()
    present_archetypes = {
        cluster_taxonomy.get(c, f"Cluster {c}"): count
        for c, count in cluster_counts.items()
    }

    all_clusters = set(range(5))
    missing_cluster_ids = all_clusters - set(cluster_counts.keys())
    missing_archetypes = [
        cluster_taxonomy.get(c, f"Cluster {c}") for c in missing_cluster_ids
    ]

    squad_report = {
        "squad_size_found": len(squad_df),
        "missing_players_not_in_database": list(missing_players),
        "archetype_distribution": present_archetypes,
        "tactical_gaps": missing_archetypes,
        "squad_averages": {
            "mean_verticality_index": float(squad_df["verticality_index"].mean()),
            "mean_pitch_tilt_ratio": float(squad_df["pitch_tilt_ratio"].mean()),
            "total_progressive_actions": float(
                squad_df["total_prog_actions"].sum()
            ),
        },
    }

    return squad_report


def get_top_archetype_players(
    cluster_id: int, top_n: int = 5, metric_sort: str = "total_prog_actions"
) -> pd.DataFrame:
    """Retrieves the top N highest-performing players within a specific tactical cluster/archetype."""
    parquet_path = "data/processed/player_pca_matrix.parquet"
    if not pd.io.common.file_exists(parquet_path):
        parquet_path = "player_pca_matrix.parquet"

    df = pd.read_parquet(parquet_path)

    if cluster_id not in df["cluster"].unique():
        valid_clusters = sorted(df["cluster"].unique().tolist())
        raise ValueError(
            f"Cluster ID {cluster_id} invalid. Must be one of {valid_clusters}."
        )

    cluster_df = df[df["cluster"] == cluster_id]

    if metric_sort not in cluster_df.columns:
        metric_sort = "total_prog_actions"

    top_players = cluster_df.sort_values(by=metric_sort, ascending=False).head(
        top_n
    )

    cluster_taxonomy = {
        0: "Tempo-Setting Recyclers",
        1: "Vertical Ball-Carriers / Transition Engines",
        2: "Final-Third Penetrators / Primary Creators",
        3: "Direct Wingers / 1v1 Take-On Specialists",
        4: "Defensive Anchor / Ball Recovery Specialists",
    }

    top_players["archetype_label"] = cluster_taxonomy.get(
        cluster_id, f"Cluster {cluster_id}"
    )

    return top_players[
        [
            "player",
            "team",
            "league_name",
            "archetype_label",
            "total_prog_actions",
            "verticality_index",
            "pitch_tilt_ratio",
        ]
    ]
