import numpy as np
import pytest

from inf8239_u03.recommenders import (
    METHODS,
    GenreSpace,
    HybridRecommender,
    MatrixFactorization,
    profile_recommendations,
    run_experiment,
    temporal_leave_one_out,
)


def test_temporal_split_keeps_last_event(sample_ratings):
    train, test = temporal_leave_one_out(sample_ratings)
    assert len(test) == sample_ratings["userId"].nunique()
    assert all(test.groupby("userId")["timestamp"].max() >= train.groupby("userId")["timestamp"].max())


def test_matrix_factorization_predicts_finite_value(sample_ratings):
    model = MatrixFactorization(factors=3, seed=42).fit(sample_ratings, epochs=2)
    assert np.isfinite(model.predict(1, 1))


def test_top_n_excludes_seen_and_reports_size(sample_ratings):
    model = MatrixFactorization(factors=3, seed=42).fit(sample_ratings, epochs=2)
    result = model.top_n(1, {1, 2}, 10)
    assert not set(result["movieId"]) & {1, 2}
    assert result["collaborative_score"].is_monotonic_decreasing
    assert model.size_bytes == (3 + 5) * 3 * 8


@pytest.fixture
def recommender(sample_ratings, sample_movies):
    model = MatrixFactorization(factors=3, seed=42).fit(sample_ratings, epochs=2)
    return HybridRecommender(model, GenreSpace().fit(sample_movies), sample_ratings, alpha=0.5)


@pytest.mark.parametrize("method", METHODS)
def test_methods_exclude_seen_items(recommender, method):
    items = recommender.recommend(method, 1, 5)
    assert items
    assert not set(items) & recommender.seen(1)
    assert len(items) == len(set(items))


def test_new_user_gets_non_personalized_popularity(recommender):
    assert not recommender.is_personalized(99)
    assert recommender.hybrid(99, 3) == recommender.popular[:3]


def test_profiles_cover_three_cases(recommender, sample_movies):
    profiles = profile_recommendations(recommender, sample_movies, k=2)
    assert set(profiles["profile"]) == {"historial_amplio", "historial_pequeno", "usuario_nuevo"}
    assert not profiles["already_seen"].any()
    assert not profiles.loc[profiles["profile"].eq("usuario_nuevo"), "personalized"].any()


def test_experiment_reports_every_method(sample_ratings, sample_movies):
    rows, _ = run_experiment(sample_ratings, sample_movies, factors=2, epochs=1, alphas=[0.25, 0.75], seed=1)
    methods = [row["method"] for row in rows]
    assert methods.count("hybrid") == 2
    assert set(methods) == set(METHODS)
    assert all(row["model_size_kb"] > 0 for row in rows)
