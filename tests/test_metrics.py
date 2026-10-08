import pandas as pd

from inf8239_u03.metrics import (
    catalog_coverage,
    hit_rate_at_k,
    mean_item_popularity,
    precision_at_k,
)


def test_ranking_metrics():
    recs = {1: [3, 2], 2: [4, 1]}
    truth = pd.DataFrame({"userId": [1, 2], "movieId": [3, 5]})
    assert hit_rate_at_k(recs, truth, 2) == 0.5
    assert catalog_coverage(recs, 5) == 0.8


def test_precision_and_popularity():
    recs = {1: [3, 2], 2: [4, 1]}
    truth = pd.DataFrame({"userId": [1, 2], "movieId": [3, 5]})
    assert precision_at_k(recs, truth, 2) == 0.25
    assert mean_item_popularity(recs, pd.Series({1: 4, 2: 2, 3: 1, 4: 1})) == 2.0
