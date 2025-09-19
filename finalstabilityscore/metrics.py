import numpy as np

def stability_score(metrics, baseline, alpha=1e-6):
    std_dev = np.std(metrics)
    denom = max(abs(baseline), alpha)
    score = 1 - (std_dev / denom)
    return float(np.clip(score, 0, 1))

# Optional: built-in scoring functions
from sklearn.metrics import roc_auc_score, mean_squared_error, r2_score

def classification_auc(X, y, model):
    return roc_auc_score(y, model.predict_proba(X)[:, 1])

def regression_rmse(X, y, model):
    return mean_squared_error(y, model.predict(X), squared=False)

def regression_r2(X, y, model):
    return r2_score(y, model.predict(X))
