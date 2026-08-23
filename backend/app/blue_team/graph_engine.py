"""NetworkX graph features for the Blue Team detector (PRD §7.3).

Approved design: the graph is built from TRAIN-split transactions only, to
avoid test-set structure leaking into features used for training. Test rows
look up precomputed train-graph statistics by entity id (beneficiary_id /
device_id / customer_id), defaulting to 0 for any entity never seen in train.
"""

from itertools import combinations
from typing import Dict

import networkx as nx
import pandas as pd


def build_train_transaction_graph(train_df: pd.DataFrame) -> nx.DiGraph:
    """Directed graph: edge customer_id -> beneficiary_id for every distinct
    payer/payee pair in the train split.
    """
    graph = nx.DiGraph()
    edges = train_df[["customer_id", "beneficiary_id"]].drop_duplicates()
    graph.add_edges_from(edges.itertuples(index=False, name=None))
    return graph


def compute_beneficiary_degrees(graph: nx.DiGraph) -> Dict[str, Dict[str, int]]:
    """Beneficiary in-degree/out-degree (train-graph only)."""
    return {"in_degree": dict(graph.in_degree()), "out_degree": dict(graph.out_degree())}


def compute_shared_device_stats(train_df: pd.DataFrame) -> Dict:
    """Per-device distinct-customer count, plus the exact (device_id,
    customer_id) pairs observed in train — used later to exclude a
    transaction's own customer from its device's shared count.
    """
    device_customer_count = train_df.groupby("device_id")["customer_id"].nunique().to_dict()
    device_customer_pairs = train_df[["device_id", "customer_id"]].drop_duplicates()
    return {"count": device_customer_count, "pairs": device_customer_pairs}


def build_customer_similarity_graph(train_df: pd.DataFrame) -> nx.Graph:
    """Undirected customer-customer graph (train-only): an edge connects two
    customers if they send to the same beneficiary, or use the same device.
    """
    graph = nx.Graph()
    for _, customer_ids in train_df.groupby("beneficiary_id")["customer_id"].unique().items():
        graph.add_edges_from(combinations(sorted(set(customer_ids)), 2))
    for _, customer_ids in train_df.groupby("device_id")["customer_id"].unique().items():
        graph.add_edges_from(combinations(sorted(set(customer_ids)), 2))
    # Ensure every train-side customer is a node even with no shared entities.
    graph.add_nodes_from(train_df["customer_id"].unique())
    return graph


def compute_two_hop_fraud_risk(similarity_graph: nx.Graph, train_df: pd.DataFrame) -> Dict[str, float]:
    """For each train-graph customer node: fraction of its exactly-2-hop
    neighbors (via shared beneficiary or shared device) with >=1 fraud
    transaction in the train split. 0.0 if it has no 2-hop neighbors.
    """
    has_fraud_train = train_df.groupby("customer_id")["is_fraud"].max().to_dict()

    risk: Dict[str, float] = {}
    for node in similarity_graph.nodes():
        lengths = nx.single_source_shortest_path_length(similarity_graph, node, cutoff=2)
        two_hop = [n for n, d in lengths.items() if d == 2]
        if not two_hop:
            risk[node] = 0.0
        else:
            risk[node] = sum(has_fraud_train.get(n, 0) for n in two_hop) / len(two_hop)
    return risk


def compute_graph_features(train_df: pd.DataFrame) -> Dict:
    """Orchestrates all graph feature computation on the train split only."""
    txn_graph = build_train_transaction_graph(train_df)
    degrees = compute_beneficiary_degrees(txn_graph)
    device_stats = compute_shared_device_stats(train_df)
    similarity_graph = build_customer_similarity_graph(train_df)
    two_hop_risk = compute_two_hop_fraud_risk(similarity_graph, train_df)
    return {
        "beneficiary_in_degree": degrees["in_degree"],
        "beneficiary_out_degree": degrees["out_degree"],
        "device_customer_count": device_stats["count"],
        "device_customer_pairs": device_stats["pairs"],
        "two_hop_fraud_risk": two_hop_risk,
        "train_row_count": len(train_df),
    }


def apply_graph_features(df: pd.DataFrame, graph_features: Dict) -> pd.DataFrame:
    """Vectorized merge of precomputed train-graph statistics onto any split
    (train or test), defaulting to 0 for entities never seen in train.
    """
    df = df.copy()
    df["beneficiary_in_degree"] = df["beneficiary_id"].map(graph_features["beneficiary_in_degree"]).fillna(0)
    df["beneficiary_out_degree"] = df["beneficiary_id"].map(graph_features["beneficiary_out_degree"]).fillna(0)

    device_count = df["device_id"].map(graph_features["device_customer_count"]).fillna(0)
    pairs = graph_features["device_customer_pairs"].copy()
    pairs["_in_train"] = True
    merged = df[["device_id", "customer_id"]].merge(pairs, on=["device_id", "customer_id"], how="left")
    self_in_train = merged["_in_train"].notna().to_numpy()
    df["shared_device_count"] = (device_count.to_numpy() - self_in_train.astype(int)).clip(min=0)

    df["two_hop_fraud_risk"] = df["customer_id"].map(graph_features["two_hop_fraud_risk"]).fillna(0.0)
    return df
