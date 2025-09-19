import numpy as np
from stabilityscore.perturbations import run_all_tests

def dummy_model(X, y):
    return np.mean(y)

def test_run_all_tests():
    X = np.random.rand(100, 5)
    y = np.random.randint(0, 2, 100)
    results = run_all_tests(X, y, dummy_model, n_iter=5, parallel=False)
    assert all(len(v) == 5 for v in results.values())
