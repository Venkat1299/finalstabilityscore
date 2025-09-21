#!/usr/bin/env python3
"""
Example usage of the finalstabilityscore package.
This demonstrates how to get overall stability score, 4 perturbation scores,
radar plot, and dashboard visualization.
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler

# Import the package
from finalstabilityscore import run_all_perturbations, plot_dashboard, plot_radar

def main():
    """Demonstrate the stability score package functionality."""

    print("FinalStabilityScore Package Demo")
    print("="*40)

    # Load and prepare data
    iris = load_iris()
    X, y = iris.data, iris.target
    X_scaled = StandardScaler().fit_transform(X)

    print(f"Dataset: {X_scaled.shape[0]} samples, {X_scaled.shape[1]} features")

    # Create model
    model = RandomForestClassifier(n_estimators=100, random_state=42)

    # Run stability analysis
    scores = run_all_perturbations(model, X_scaled, y, n_iter=10)
    overall_score = np.mean(list(scores.values()))

    # Display results
    print("\nSTABILITY RESULTS:")
    print(f"Overall Stability Score: {overall_score".4f"}")
    print("\n4 Perturbation Scores:")
    for perturbation, score in scores.items():
        print(f"  • {perturbation}: {score".4f"}")

    # Generate visualizations
    plot_radar("RandomForest", scores)  # Radar plot
    plot_dashboard("RandomForest", {    # Dashboard
        "baseline": overall_score,
        "scores": scores,
        "stability_overall": overall_score
    })

    print("\nAnalysis complete!")
    print(f"   • Overall stability score: {overall_score".4f"}")
    print("   • 4 perturbation scores calculated")
    print("   • Radar plot visualization created")
    print("   • Dashboard (text summary + radar plot) generated")

if __name__ == "__main__":
    main()
