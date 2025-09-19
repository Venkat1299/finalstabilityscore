from .perturbations import run_all_tests
from .metrics import stability_score
from .utils import log_info

def stability_framework(X, y, model_fn, alpha=1e-6, parallel=True):
    baseline = model_fn(X, y)
    log_info(f"Baseline score: {baseline:.4f}")

    results = run_all_tests(X, y, model_fn, parallel=parallel)
    scores = {k: stability_score(v, baseline, alpha) for k, v in results.items()}
    stability_overall = sum(scores.values()) / len(scores)

    return {
        "baseline": baseline,
        "scores": scores,
        "stability_overall": stability_overall
    }
