import numpy as np

def learn_weights(results, baseline, alpha=1e-6):
    """
    Learn dynamic weights for each perturbation type based on variability.

    Parameters:
    - results: dict of perturbation results, e.g. {"noise": [...], "bootstrap": [...]}
    - baseline: original model score on clean data
    - alpha: small constant to prevent division by zero

    Returns:
    - weights: dict of normalized weights for each perturbation type
    """
    raw_weights = {}
    for perturb_type, scores in results.items():
        std_dev = np.std(scores)
        denom = max(abs(baseline), alpha)
        variability = std_dev / denom
        raw_weights[perturb_type] = 1 / (variability + alpha)

    total = sum(raw_weights.values())
    normalized_weights = {k: v / total for k, v in raw_weights.items()}
    return normalized_weights
