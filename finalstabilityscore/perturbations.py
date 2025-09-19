import numpy as np
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from .metrics import stability_score

def bootstrap(X, y, model_fn):
    idx = np.random.choice(len(y), len(y), replace=True)
    return model_fn(X[idx], y[idx])

def noise(X, y, model_fn):
    noise = np.random.normal(0, 0.1 * X.std(axis=0), X.shape)
    return model_fn(X + noise, y)

def missingness(X, y, model_fn):
    X_m = X.copy()
    mask = np.random.rand(*X.shape) < 0.1
    X_m[mask] = np.nan
    X_m = np.where(np.isnan(X_m), np.nanmean(X_m, axis=0), X_m)
    return model_fn(X_m, y)

def outliers(X, y, model_fn):
    X_o = X.copy()
    idx = np.random.choice(len(X), int(0.05 * len(X)), replace=False)
    X_o[idx] *= 10
    return model_fn(X_o, y)

def run_all_perturbations(model, X, y, n_iter=10, test_size=0.2, random_state=42):
    """
    Run all perturbations on a given model and return stability scores.
    
    Parameters:
    -----------
    model : sklearn model
        The model to test
    X : array-like
        Feature matrix
    y : array-like
        Target vector
    n_iter : int, default=10
        Number of iterations for each perturbation
    test_size : float, default=0.2
        Test set size for train/test split
    random_state : int, default=42
        Random state for reproducibility
        
    Returns:
    --------
    dict : Dictionary with perturbation names as keys and stability scores as values
    """
    
    def perturb_bootstrap(X, y):
        idx = np.random.choice(len(X), len(X), replace=True)
        return X[idx], y[idx]

    def perturb_noise(X, y):
        noise = np.random.normal(0, 0.1, X.shape)
        return X + noise, y

    def perturb_missing(X, y):
        X_copy = X.copy()
        mask = np.random.rand(*X.shape) < 0.1
        X_copy[mask] = 0
        return X_copy, y

    def perturb_outliers(X, y):
        X_copy = X.copy()
        n_outliers = int(0.05 * len(X))
        idx = np.random.choice(len(X), n_outliers, replace=False)
        X_copy[idx] += np.random.normal(10, 5, X_copy[idx].shape)
        return X_copy, y

    perturbations = {
        "Bootstrapping": perturb_bootstrap,
        "Noise": perturb_noise,
        "Missingness": perturb_missing,
        "Outliers": perturb_outliers
    }

    scores = {}
    for name, perturb_fn in perturbations.items():
        acc_list = []
        for i in range(n_iter):
            X_pert, y_pert = perturb_fn(X, y)
            X_train, X_test, y_train, y_test = train_test_split(
                X_pert, y_pert, test_size=test_size, random_state=random_state + i
            )
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            acc = accuracy_score(y_test, preds)
            acc_list.append(acc)
        baseline = np.mean(acc_list)
        score = stability_score(acc_list, baseline)
        scores[name] = score

    return scores

def run_all_tests(X, y, model_fn, n_iter=20, parallel=True):
    tests = {
        "bootstrap": bootstrap,
        "noise": noise,
        "missingness": missingness,
        "outliers": outliers
    }
    if parallel:
        return {
            k: Parallel(n_jobs=-1)(delayed(fn)(X, y, model_fn) for _ in range(n_iter))
            for k, fn in tests.items()
        }
    else:
        return {
            k: [fn(X, y, model_fn) for _ in range(n_iter)]
            for k, fn in tests.items()
        }
