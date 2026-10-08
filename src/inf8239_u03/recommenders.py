from __future__ import annotations

from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import root_mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity

from inf8239_u03.metrics import (
    catalog_coverage,
    hit_rate_at_k,
    mean_item_popularity,
    precision_at_k,
)


def weighted_popularity(
    ratings: pd.DataFrame, movies: pd.DataFrame, quantile: float = 0.80, min_count: float | None = None
) -> pd.DataFrame:
    """Promedio bayesiano; `min_count=None` filtra por el mismo cuantil usado para suavizar."""
    stats = ratings.groupby("movieId")["rating"].agg(["mean", "count"])
    global_mean = float(ratings["rating"].mean())
    minimum = float(stats["count"].quantile(quantile))
    stats["weighted_score"] = (
        stats["count"] / (stats["count"] + minimum) * stats["mean"]
        + minimum / (stats["count"] + minimum) * global_mean
    )
    threshold = minimum if min_count is None else min_count
    return stats[stats["count"] >= threshold].join(movies.set_index("movieId")).sort_values("weighted_score", ascending=False)


class ContentRecommender:
    def fit(self, movies: pd.DataFrame) -> ContentRecommender:
        self.movies = movies.reset_index(drop=True).copy()
        self.movies["genres_text"] = self.movies["genres"].str.replace("|", " ", regex=False)
        self.vectorizer = TfidfVectorizer()
        self.matrix = self.vectorizer.fit_transform(self.movies["genres_text"])
        return self

    def recommend(self, title: str, k: int = 10) -> pd.DataFrame:
        matches = self.movies.index[self.movies["title"].eq(title)]
        if len(matches) == 0:
            raise KeyError(f"Título no encontrado: {title}")
        index = int(matches[0])
        scores = cosine_similarity(self.matrix[index], self.matrix).ravel()
        order = [value for value in scores.argsort()[::-1] if value != index][:k]
        result = self.movies.loc[order, ["movieId", "title", "genres"]].copy()
        result["content_score"] = scores[order]
        return result.reset_index(drop=True)


class MatrixFactorization:
    def __init__(self, factors: int = 20, learning_rate: float = 0.01, regularization: float = 0.05, seed: int = 42):
        self.factors = factors
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.seed = seed

    def fit(self, ratings: pd.DataFrame, epochs: int = 12) -> MatrixFactorization:
        self.users = sorted(ratings["userId"].unique())
        self.items = sorted(ratings["movieId"].unique())
        self.user_index = {value: index for index, value in enumerate(self.users)}
        self.item_index = {value: index for index, value in enumerate(self.items)}
        rng = np.random.default_rng(self.seed)
        self.user_factors = rng.normal(0, 0.1, (len(self.users), self.factors))
        self.item_factors = rng.normal(0, 0.1, (len(self.items), self.factors))
        self.global_mean = float(ratings["rating"].mean())
        observations = [(self.user_index[r.userId], self.item_index[r.movieId], float(r.rating)) for r in ratings.itertuples()]
        for _ in range(epochs):
            rng.shuffle(observations)
            for user, item, rating in observations:
                error = rating - self.predict_indices(user, item)
                previous_user = self.user_factors[user].copy()
                self.user_factors[user] += self.learning_rate * (error * self.item_factors[item] - self.regularization * self.user_factors[user])
                self.item_factors[item] += self.learning_rate * (error * previous_user - self.regularization * self.item_factors[item])
        return self

    def predict_indices(self, user_index: int, item_index: int) -> float:
        return float(self.global_mean + self.user_factors[user_index] @ self.item_factors[item_index])

    def predict(self, user_id: int, movie_id: int) -> float:
        return self.predict_indices(self.user_index[user_id], self.item_index[movie_id])

    @property
    def size_bytes(self) -> int:
        return int(self.user_factors.nbytes + self.item_factors.nbytes)

    def top_n(self, user_id: int, seen: set[int], k: int = 10) -> pd.DataFrame:
        if user_id not in self.user_index:
            return pd.DataFrame(columns=["movieId", "collaborative_score"])
        items = np.asarray(self.items)
        scores = self.global_mean + self.item_factors @ self.user_factors[self.user_index[user_id]]
        unseen = ~np.isin(items, list(seen))
        items, scores = items[unseen], scores[unseen]
        order = np.argsort(-scores, kind="stable")[:k]
        return pd.DataFrame({"movieId": items[order], "collaborative_score": scores[order]})


def temporal_leave_one_out(ratings: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    ordered = ratings.sort_values(["userId", "timestamp"])
    test = ordered.groupby("userId", sort=False).tail(1)
    train = ordered.drop(test.index)
    return train.reset_index(drop=True), test.reset_index(drop=True)


# --- LAB09: híbrido, fallback de usuario frío y evaluación comparativa ---

METHODS = ("popularity", "content", "collaborative", "hybrid")


def min_max(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return values
    return (values - values.min()) / max(values.max() - values.min(), 1e-9)


def popularity_ranking(train: pd.DataFrame) -> list[int]:
    stats = train.groupby("movieId")["rating"].agg(["count", "mean"])
    return stats.sort_values(["count", "mean"], ascending=False).index.astype(int).tolist()


class GenreSpace:
    """TF-IDF de géneros; el perfil de usuario pondera lo que valoró por encima de 2.5."""

    def fit(self, movies: pd.DataFrame) -> GenreSpace:
        genres = movies["genres"].str.replace("|", " ", regex=False)
        self.matrix = TfidfVectorizer().fit_transform(genres)
        self.row = {int(movie_id): index for index, movie_id in enumerate(movies["movieId"])}
        return self

    @property
    def size_bytes(self) -> int:
        return int(self.matrix.data.nbytes + self.matrix.indices.nbytes + self.matrix.indptr.nbytes)

    def profile(self, history: pd.DataFrame) -> np.ndarray | None:
        history = history[history["movieId"].isin(self.row)]
        if history.empty:
            return None
        rows = [self.row[int(item)] for item in history["movieId"]]
        weights = np.clip(history["rating"].to_numpy() - 2.5, 0.1, None)
        return np.asarray(self.matrix[rows].multiply(weights[:, None]).sum(axis=0) / weights.sum())

    def scores(self, profile: np.ndarray, movie_ids) -> np.ndarray:
        rows = [self.row[int(item)] for item in movie_ids]
        return cosine_similarity(profile, self.matrix[rows]).ravel()


class HybridRecommender:
    """Cuatro estrategias sobre el mismo catálogo de entrenamiento.

    Un usuario sin factores latentes (usuario frío) recibe popularidad: lista no personalizada.
    """

    def __init__(self, model: MatrixFactorization, space: GenreSpace, train: pd.DataFrame, alpha: float = 0.75, pool: int = 100):
        self.model = model
        self.space = space
        self.alpha = alpha
        self.pool = pool
        self.popular = popularity_ranking(train)
        # Candidatos de contenido ordenados por popularidad: los empates de similitud se resuelven a favor del más valorado.
        self.candidates = np.asarray(self.popular)
        self.history = {int(user): group[["movieId", "rating"]] for user, group in train.groupby("userId")}

    def is_personalized(self, user_id: int) -> bool:
        return user_id in self.model.user_index

    def seen(self, user_id: int) -> set[int]:
        history = self.history.get(user_id)
        return set() if history is None else set(history["movieId"].astype(int))

    def popularity(self, user_id: int, k: int = 10) -> list[int]:
        seen = self.seen(user_id)
        return [item for item in self.popular if item not in seen][:k]

    def content(self, user_id: int, k: int = 10) -> list[int]:
        history = self.history.get(user_id)
        profile = None if history is None else self.space.profile(history)
        if profile is None:
            return self.popularity(user_id, k)
        candidates = self.candidates[~np.isin(self.candidates, list(self.seen(user_id)))]
        scores = self.space.scores(profile, candidates)
        return candidates[np.argsort(-scores, kind="stable")[:k]].astype(int).tolist()

    def collaborative(self, user_id: int, k: int = 10) -> list[int]:
        if not self.is_personalized(user_id):
            return self.popularity(user_id, k)
        return self.model.top_n(user_id, self.seen(user_id), k)["movieId"].astype(int).tolist()

    def hybrid(self, user_id: int, k: int = 10) -> list[int]:
        if not self.is_personalized(user_id):
            return self.popularity(user_id, k)
        pool = self.model.top_n(user_id, self.seen(user_id), self.pool)
        profile = self.space.profile(self.history[user_id])
        content = np.zeros(len(pool)) if profile is None else min_max(self.space.scores(profile, pool["movieId"]))
        score = self.alpha * min_max(pool["collaborative_score"]) + (1 - self.alpha) * content
        return pool["movieId"].to_numpy()[np.argsort(-score, kind="stable")[:k]].astype(int).tolist()

    def recommend(self, method: str, user_id: int, k: int = 10) -> list[int]:
        return getattr(self, method)(user_id, k)


def evaluable_test(model: MatrixFactorization, test: pd.DataFrame) -> pd.DataFrame:
    return test[test["userId"].isin(model.user_index) & test["movieId"].isin(model.item_index)]


def rating_errors(model: MatrixFactorization, train: pd.DataFrame, evaluable: pd.DataFrame) -> dict[str, float]:
    """RMSE del modelo frente a dos líneas base sin aprendizaje latente."""
    truth = evaluable["rating"]
    item_mean = train.groupby("movieId")["rating"].mean()
    return {
        "rmse": float(root_mean_squared_error(truth, [model.predict(r.userId, r.movieId) for r in evaluable.itertuples()])),
        "rmse_global_mean": float(root_mean_squared_error(truth, np.full(len(truth), train["rating"].mean()))),
        "rmse_item_mean": float(root_mean_squared_error(truth, evaluable["movieId"].map(item_mean))),
    }


def evaluate_method(recommender: HybridRecommender, method: str, evaluable: pd.DataFrame, item_counts: pd.Series, k: int = 10) -> dict[str, float]:
    start = perf_counter()
    recommendations = {int(user): recommender.recommend(method, int(user), k) for user in evaluable["userId"].unique()}
    return {
        f"hit_rate_at_{k}": hit_rate_at_k(recommendations, evaluable, k),
        f"precision_at_{k}": precision_at_k(recommendations, evaluable, k),
        "catalog_coverage": catalog_coverage(recommendations, len(item_counts)),
        "mean_item_popularity": mean_item_popularity(recommendations, item_counts),
        "eval_seconds": perf_counter() - start,
    }


def run_experiment(
    ratings: pd.DataFrame,
    movies: pd.DataFrame,
    factors: int,
    epochs: int,
    alphas: list[float],
    seed: int,
    k: int = 10,
) -> tuple[list[dict], HybridRecommender]:
    """Entrena una vez por semilla y evalúa las cuatro estrategias con el mismo corte temporal."""
    train, test = temporal_leave_one_out(ratings)
    start = perf_counter()
    model = MatrixFactorization(factors=factors, seed=seed).fit(train, epochs=epochs)
    train_seconds = perf_counter() - start
    space = GenreSpace().fit(movies)
    evaluable = evaluable_test(model, test)
    item_counts = train.groupby("movieId").size()
    errors = rating_errors(model, train, evaluable)
    base = {"factors": factors, "epochs": epochs, "seed": seed, "evaluated_users": int(evaluable["userId"].nunique())}
    costs = {
        "popularity": (0.0, len(item_counts) * 8),
        "content": (0.0, space.size_bytes),
        "collaborative": (train_seconds, model.size_bytes),
        "hybrid": (train_seconds, model.size_bytes + space.size_bytes),
    }
    rows, recommender = [], None
    for alpha in alphas:
        recommender = HybridRecommender(model, space, train, alpha=alpha)
        methods = METHODS if alpha == alphas[0] else ("hybrid",)
        for method in methods:
            seconds, size = costs[method]
            learned = method in ("collaborative", "hybrid")
            rows.append({
                **base,
                "method": method,
                "alpha": alpha if method == "hybrid" else np.nan,
                "rmse": errors["rmse"] if learned else np.nan,
                "rmse_global_mean": errors["rmse_global_mean"],
                "rmse_item_mean": errors["rmse_item_mean"],
                **evaluate_method(recommender, method, evaluable, item_counts, k),
                "train_seconds": seconds,
                "model_size_kb": size / 1024,
            })
    return rows, recommender


def select_profiles(recommender: HybridRecommender) -> dict[str, int]:
    """Usuario con más historial, con menos historial y un usuario nuevo inexistente."""
    sizes = pd.Series({user: len(history) for user, history in recommender.history.items()})
    return {
        "historial_amplio": int(sizes.idxmax()),
        "historial_pequeno": int(sizes.sort_values(kind="stable").index[0]),
        "usuario_nuevo": int(sizes.index.max()) + 1,
    }


def profile_recommendations(recommender: HybridRecommender, movies: pd.DataFrame, k: int = 10) -> pd.DataFrame:
    titles = movies.set_index("movieId")["title"]
    frames = []
    for profile, user_id in select_profiles(recommender).items():
        seen = recommender.seen(user_id)
        for method in METHODS:
            items = recommender.recommend(method, user_id, k)
            frames.append(pd.DataFrame({
                "profile": profile,
                "userId": user_id,
                "history_size": len(seen),
                "personalized": recommender.is_personalized(user_id),
                "method": method,
                "rank": range(1, len(items) + 1),
                "movieId": items,
                "title": titles.reindex(items).to_numpy(),
                "already_seen": [item in seen for item in items],
            }))
    return pd.concat(frames, ignore_index=True)
