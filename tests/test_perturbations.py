import numpy as np
from sklearn.ensemble import RandomForestClassifier
from finalstabilityscore.perturbations import run_all_perturbations

def test_run_all_perturbations():
    X = np.random.rand(100, 5)
    y = np.random.randint(0, 2, 100)
    model = RandomForestClassifier(n_estimators=10, random_state=42)

    scores = run_all_perturbations(model, X, y, n_iter=3)
    assert len(scores) == 4  # Should have 4 perturbation types
    assert all(0 <= score <= 1 for score in scores.values())
