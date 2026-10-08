from __future__ import annotations

import argparse
import json

import pandas as pd

from inf8239_u03.config import MOVIELENS_DIR, RANDOM_STATE, ROOT
from inf8239_u03.data import load_movielens
from inf8239_u03.recommenders import profile_recommendations, run_experiment

parser = argparse.ArgumentParser()
parser.add_argument("--factors", type=int, default=20)
parser.add_argument("--epochs", type=int, default=12)
parser.add_argument("--alpha", type=float, default=0.75)
parser.add_argument("--seed", type=int, default=RANDOM_STATE)
args = parser.parse_args()

ratings, movies = load_movielens(MOVIELENS_DIR)
rows, recommender = run_experiment(ratings, movies, args.factors, args.epochs, [args.alpha], args.seed)
results = pd.DataFrame(rows)
hybrid = results[results["method"].eq("hybrid")].iloc[0]

metrics = {
    "rmse": hybrid["rmse"],
    "rmse_global_mean": hybrid["rmse_global_mean"],
    "rmse_item_mean": hybrid["rmse_item_mean"],
    "hit_rate_at_10": hybrid["hit_rate_at_10"],
    "precision_at_10": hybrid["precision_at_10"],
    "catalog_coverage": hybrid["catalog_coverage"],
    "mean_item_popularity": hybrid["mean_item_popularity"],
    "train_seconds": hybrid["train_seconds"],
    "model_size_kb": hybrid["model_size_kb"],
    "evaluated_users": int(hybrid["evaluated_users"]),
    "factors": args.factors,
    "epochs": args.epochs,
    "alpha": args.alpha,
    "seed": args.seed,
    "cold_start_policy": "popularidad por cantidad y media; no personalizada",
}
metrics = {key: value.item() if hasattr(value, "item") else value for key, value in metrics.items()}

reports = ROOT / "reports"
runs = reports / "lab09_runs"
runs.mkdir(parents=True, exist_ok=True)
name = f"f{args.factors}_e{args.epochs}_a{args.alpha}_s{args.seed}"
(runs / f"metrics_{name}.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
results.to_csv(runs / f"methods_{name}.csv", index=False)
(reports / "hybrid_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

profiles = profile_recommendations(recommender, movies)
profiles.to_csv(runs / f"profiles_{name}.csv", index=False)
fallback_ids = recommender.popularity(-1, 10)
movies.set_index("movieId").loc[fallback_ids].reset_index().to_csv(reports / "cold_start_fallback.csv", index=False)

print(json.dumps(metrics, indent=2))
print("\nComparación de estrategias (mismo corte y semilla):")
print(results[["method", "alpha", "rmse", "hit_rate_at_10", "precision_at_10", "catalog_coverage", "mean_item_popularity", "train_seconds", "model_size_kb"]].to_string(index=False))
print("\nRecomendaciones híbridas de ejemplo por perfil:")
sample = profiles[profiles["method"].eq("hybrid")]
for (profile, user_id), group in sample.groupby(["profile", "userId"], sort=False):
    first = group.iloc[0]
    label = "personalizada" if first["personalized"] else "NO personalizada (fallback de popularidad)"
    print(f"\n{profile} · userId={user_id} · historial={first['history_size']} · {label}")
    print(group[["rank", "title", "already_seen"]].to_string(index=False))
