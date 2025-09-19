import numpy as np
from stabilityscore.weighting import learn_weights

def dummy_results():
    return {
        "bootstrap": [0.80, 0.82, 0.79, 0.81],
        "noise": [0.75, 0.85, 0.70, 0.90],
        "missingness": [0.78, 0.77, 0.79, 0.76],
        "outliers": [0.60, 0.95, 0.55, 0.90]
    }

def test_learn_weights_sum_to_one():
    baseline = 0.80
    results = dummy_results()
    weights = learn_weights(results, baseline)
    total_weight = sum(weights.values())
    assert abs(total_weight - 1.0) < 1e-6, "Weights should sum to 1"

def test_learn_weights_non_negative():
    baseline = 0.80
    results = dummy_results()
    weights = learn_weights(results, baseline)
    for w in weights.values():
        assert w >= 0, "Weights should be non-negative"

def test_learn_weights_keys_match():
    baseline = 0.80
    results = dummy_results()
    weights = learn_weights(results, baseline)
    assert set(weights.keys()) == set(results.keys()), "Weight keys should match perturbation types"
