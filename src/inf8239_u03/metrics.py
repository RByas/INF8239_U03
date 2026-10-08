from __future__ import annotations

import pandas as pd


def hit_rate_at_k(recommendations: dict[int, list[int]], truth: pd.DataFrame, k: int = 10) -> float:
    hits = []
    for row in truth.itertuples():
        if row.userId in recommendations:
            hits.append(int(row.movieId in recommendations[row.userId][:k]))
    return sum(hits) / len(hits) if hits else 0.0


def catalog_coverage(recommendations: dict[int, list[int]], catalog_size: int) -> float:
    recommended = {item for values in recommendations.values() for item in values}
    return len(recommended) / catalog_size if catalog_size else 0.0


def precision_at_k(recommendations: dict[int, list[int]], truth: pd.DataFrame, k: int = 10) -> float:
    relevant = truth.groupby("userId")["movieId"].apply(set)
    values = [
        len(set(recommendations[user][:k]) & items) / k
        for user, items in relevant.items()
        if user in recommendations
    ]
    return sum(values) / len(values) if values else 0.0


def mean_item_popularity(recommendations: dict[int, list[int]], item_counts: pd.Series) -> float:
    """Promedio de valoraciones en entrenamiento de los ítems recomendados (alto = popularidad dominante)."""
    counts = [float(item_counts.get(item, 0)) for values in recommendations.values() for item in values]
    return sum(counts) / len(counts) if counts else 0.0
