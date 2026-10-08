"""Comparación controlada de LAB09: alpha, factores y tres semillas, con tabla de Pareto costo-utilidad."""

from __future__ import annotations

import argparse
from itertools import combinations

import pandas as pd

from inf8239_u03.config import MOVIELENS_DIR, ROOT
from inf8239_u03.data import load_movielens
from inf8239_u03.recommenders import profile_recommendations, run_experiment

parser = argparse.ArgumentParser()
parser.add_argument("--factors", type=int, nargs="+", default=[10, 20, 40])
parser.add_argument("--alphas", type=float, nargs="+", default=[0.25, 0.5, 0.75])
parser.add_argument("--seeds", type=int, nargs="+", default=[42, 7, 123])
parser.add_argument("--epochs", type=int, default=12)
parser.add_argument("--reuse", action="store_true", help="recalcula tablas desde reports/lab09_runs.csv sin reentrenar")
args = parser.parse_args()

reports = ROOT / "reports"
reports.mkdir(exist_ok=True)

if args.reuse:
    runs = pd.read_csv(reports / "lab09_runs.csv")
    profiles = pd.read_csv(reports / "lab09_profiles.csv")
else:
    ratings, movies = load_movielens(MOVIELENS_DIR)
    rows, profile_frames = [], []
    for factors in args.factors:
        for seed in args.seeds:
            print(f"Entrenando factors={factors} seed={seed} ...", flush=True)
            run_rows, recommender = run_experiment(ratings, movies, factors, args.epochs, args.alphas, seed)
            rows.extend(run_rows)
            for alpha in args.alphas:
                recommender.alpha = alpha
                frame = profile_recommendations(recommender, movies)
                profile_frames.append(frame.assign(factors=factors, seed=seed, alpha=alpha))
    runs = pd.DataFrame(rows)
    runs.to_csv(reports / "lab09_runs.csv", index=False)
    profiles = pd.concat(profile_frames, ignore_index=True)
    profiles.to_csv(reports / "lab09_profiles.csv", index=False)

# Popularidad y contenido no dependen de factores ni semilla: se conservan una sola vez.
runs["config"] = runs.apply(
    lambda r: r["method"] if r["method"] in ("popularity", "content")
    else f"{r['method']} f={r['factors']}" + (f" a={r['alpha']}" if r["method"] == "hybrid" else ""),
    axis=1,
)
metrics = ["rmse", "hit_rate_at_10", "precision_at_10", "catalog_coverage", "mean_item_popularity", "train_seconds", "model_size_kb"]
summary = runs.groupby("config", sort=False).agg(
    method=("method", "first"),
    factors=("factors", "first"),
    alpha=("alpha", "first"),
    seeds=("seed", "nunique"),
    **{f"{m}_mean": (m, "mean") for m in metrics},
    hit_rate_at_10_std=("hit_rate_at_10", "std"),
    catalog_coverage_std=("catalog_coverage", "std"),
).reset_index()
summary.loc[summary["method"].isin(["popularity", "content"]), "factors"] = pd.NA


# Utilidad en dos dimensiones (aciertos y amplitud del catálogo); costo en tiempo y tamaño.
UTILITY = ["hit_rate_at_10_mean", "catalog_coverage_mean"]
COST = ["train_seconds_mean", "model_size_kb_mean"]


def dominated(row: pd.Series, table: pd.DataFrame) -> str:
    """Dominada si otra configuración es igual o mejor en toda utilidad e igual o menor en todo costo, y estrictamente mejor en algo."""
    for _, other in table.iterrows():
        if other["config"] == row["config"]:
            continue
        no_worse = all(other[c] >= row[c] for c in UTILITY) and all(other[c] <= row[c] for c in COST)
        better = any(other[c] > row[c] for c in UTILITY) or any(other[c] < row[c] for c in COST)
        if no_worse and better:
            return other["config"]
    return ""


summary["dominated_by"] = summary.apply(dominated, axis=1, table=summary)
summary["pareto_optimal"] = summary["dominated_by"].eq("")
summary = summary.sort_values("hit_rate_at_10_mean", ascending=False)
summary.to_csv(reports / "lab09_summary.csv", index=False)
summary[["config", "hit_rate_at_10_mean", "catalog_coverage_mean", "train_seconds_mean", "model_size_kb_mean", "pareto_optimal", "dominated_by"]].to_csv(
    reports / "lab09_pareto.csv", index=False
)

# Estabilidad entre semillas (Jaccard medio del Top-10 híbrido) y dependencia del contenido.
stability = []
for (profile, factors, alpha), group in profiles[profiles["method"].eq("hybrid")].groupby(["profile", "factors", "alpha"]):
    lists = [set(g["movieId"]) for _, g in group.groupby("seed")]
    pairs = [len(a & b) / len(a | b) for a, b in combinations(lists, 2)]
    content = profiles[(profiles["profile"].eq(profile)) & profiles["method"].eq("content") & profiles["factors"].eq(factors) & profiles["alpha"].eq(alpha)]
    content_sets = {seed: set(g["movieId"]) for seed, g in content.groupby("seed")}
    overlap = [len(set(g["movieId"]) & content_sets[seed]) / 10 for seed, g in group.groupby("seed")]
    stability.append({
        "profile": profile,
        "factors": factors,
        "alpha": alpha,
        "history_size": int(group["history_size"].iloc[0]),
        "personalized": bool(group["personalized"].iloc[0]),
        "jaccard_between_seeds": sum(pairs) / len(pairs) if pairs else 1.0,
        "overlap_with_content_top10": sum(overlap) / len(overlap),
        "any_seen_item": bool(group["already_seen"].any()),
    })
stability = pd.DataFrame(stability)
stability.to_csv(reports / "lab09_profile_stability.csv", index=False)

pd.set_option("display.width", 200)
print("\nResumen (media sobre semillas):")
print(summary[["config", "rmse_mean", "hit_rate_at_10_mean", "hit_rate_at_10_std", "catalog_coverage_mean", "mean_item_popularity_mean", "train_seconds_mean", "model_size_kb_mean", "pareto_optimal"]].to_string(index=False))
print("\nEstabilidad de perfiles (factors=20):")
print(stability[stability["factors"].eq(20)].to_string(index=False))
