# FinalStabilityScore 📊

A comprehensive package for evaluating model stability under perturbations with radar plot visualizations.

## 🔍 What It Does

- **4 Perturbation Tests**: Bootstrapping, Noise, Missingness, and Outliers
- **Overall Stability Score**: Computes bounded stability score using: `Stability Score = 1 - σ(R) / max(|R₀|, α)`
- **Radar Plot Visualization**: Interactive radar charts showing perturbation scores
- **Dashboard**: Text summary + radar plot visualization
- **Model Agnostic**: Works with any sklearn-compatible model

## 🚀 Installation

```bash
pip install -e .
```

## 📖 Quick Usage

```python
from finalstabilityscore import run_all_perturbations, plot_dashboard, plot_radar
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler

# Load and prepare data
iris = load_iris()
X, y = iris.data, iris.target
X_scaled = StandardScaler().fit_transform(X)

# Create model
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Run stability analysis
scores = run_all_perturbations(model, X_scaled, y, n_iter=10)
overall_score = np.mean(list(scores.values()))

# Generate visualizations
plot_radar("RandomForest", scores)  # Radar plot
plot_dashboard("RandomForest", {    # Dashboard
    "baseline": overall_score,
    "scores": scores,
    "stability_overall": overall_score
})
```

## 📊 Output

- **Overall Stability Score**: Single metric (0-1, higher is better)
- **4 Perturbation Scores**: Individual scores for each perturbation type
- **Radar Plot**: Visual representation of stability across perturbations
- **Dashboard**: Text summary + radar plot visualization
# finalstabilityscore
