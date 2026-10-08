import pandas as pd

from inf8239_u03.config import MOVIELENS_DIR, ROOT
from inf8239_u03.data import load_movielens
from inf8239_u03.recommenders import ContentRecommender, weighted_popularity

QUERIES = ["Toy Story (1995)", "Heat (1995)", "Sabrina (1995)"]  # géneros distintos
TOP_N = 10
COLS = ["title", "count", "mean", "weighted_score"]

ratings, movies = load_movielens(MOVIELENS_DIR)
reports = ROOT / "reports"
reports.mkdir(exist_ok=True)

# 1. Baseline de popularidad suavizada (no personalizado)
popular_all = weighted_popularity(ratings, movies)
popular_all.head(TOP_N)[COLS].to_csv(reports / "popular_top10.csv")
# Evidencia del suavizado: el mismo ranking ordenado por media simple
by_mean = popular_all.sort_values("mean", ascending=False).head(TOP_N)
by_mean[COLS].to_csv(reports / "popular_mean_top10.csv")
single_rating = int((popular_all["count"] == 1).sum())
print("Popularidad suavizada:\n", popular_all.head(TOP_N)[COLS])
print("\nMismo ranking por media simple (sin suavizar):\n", by_mean[COLS])
print(f"\nPelículas con una sola valoración: {single_rating}")

# 2. Recomendación por contenido, con desempate por popularidad suavizada
# El suavizado se aplica a todas las películas valoradas (no solo a las que superan el umbral),
# para que `count` sea el número real de valoraciones; una película sin valoraciones recibe la media global.
prior = ratings["rating"].mean()
popular_every = weighted_popularity(ratings, movies, quantile=0.80, min_count=0)
recommender = ContentRecommender().fit(movies)
frames, tie_rows = [], []
for title in QUERIES:
    candidates = recommender.recommend(title, len(movies)).copy()
    candidates["content_score"] = candidates["content_score"].round(6)
    candidates = candidates.merge(
        popular_every[["count", "weighted_score"]], left_on="movieId", right_index=True, how="left"
    )
    candidates["count"] = candidates["count"].fillna(0).astype(int)
    candidates["weighted_score"] = candidates["weighted_score"].fillna(prior)
    top_score = candidates["content_score"].max()
    n_ties = int((candidates["content_score"] == top_score).sum())
    top = candidates.sort_values(
        ["content_score", "weighted_score"], ascending=False, kind="stable"
    ).head(TOP_N)
    top.insert(0, "rank", range(1, len(top) + 1))
    top.insert(0, "query", title)
    frames.append(top)
    genres = movies.loc[movies["title"] == title, "genres"].iloc[0]
    tie_rows.append({"query": title, "genres": genres, "top_score": top_score, "tied_movies": n_ties})
    print(f"\nSimilares a {title} ({n_ties} películas empatan en el puntaje máximo {top_score}):")
    print(top[["rank", "title", "genres", "content_score", "count", "weighted_score"]].to_string(index=False))

pd.concat(frames).to_csv(reports / "content_recommendations.csv", index=False)
pd.DataFrame(tie_rows).to_csv(reports / "content_ties.csv", index=False)

# 3. Cold start de ítems: películas sin ninguna valoración
rated_ids = set(ratings["movieId"].unique())
cold = movies[~movies["movieId"].isin(rated_ids)]
cold.to_csv(reports / "cold_start_items.csv", index=False)
print(f"\nPelículas sin valoraciones (cold start de ítems): {len(cold)}")