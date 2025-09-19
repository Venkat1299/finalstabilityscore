import numpy as np
from stabilityscore.core import stability_framework

def dummy_model(X, y):
    return np.mean(y)

def test_stability_framework():
    X = np.random.rand(100, 5)
    y = np.random.randint(0, 2, 100)
    result = stability_framework(X, y, dummy_model)
    assert "stability_overall" in result
    assert 0 <= result["stability_overall"] <= 1
