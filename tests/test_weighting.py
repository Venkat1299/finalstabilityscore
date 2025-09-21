import numpy as np
from finalstabilityscore.weighting import learn_weights

def test_learn_weights():
    results = {
        "noise": [0.8, 0.9, 0.85],
        "bootstrap": [0.82, 0.88, 0.85]
    }
    baseline = 0.85
    weights = learn_weights(results, baseline)
    assert len(weights) == 2
    assert all(0 <= w <= 1 for w in weights.values())
    assert abs(sum(weights.values()) - 1.0) < 1e-6
