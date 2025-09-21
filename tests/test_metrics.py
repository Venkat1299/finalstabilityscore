import numpy as np
from finalstabilityscore.metrics import stability_score

def test_stability_score():
    metrics = [0.8, 0.9, 0.85, 0.88, 0.82]
    baseline = 0.85
    score = stability_score(metrics, baseline)
    assert 0 <= score <= 1
