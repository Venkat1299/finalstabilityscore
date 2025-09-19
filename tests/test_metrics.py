import numpy as np
from stabilityscore.metrics import stability_score

def test_stability_score():
    metrics = [0.8, 0.82, 0.78, 0.81]
    baseline = 0.8
    score = stability_score(metrics, baseline)
    assert 0 <= score <= 1
