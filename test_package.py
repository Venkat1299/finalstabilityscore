#!/usr/bin/env python3
"""
Simple test to verify the finalstabilityscore package works correctly.
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler

# Import the package
from finalstabilityscore import run_all_perturbations, plot_dashboard, plot_radar

def main():
    print("Testing FinalStabilityScore Package...")

    # Load and prepare data
    iris = load_iris()
    X, y = iris.data, iris.target
    X_scaled = StandardScaler().fit_transform(X)

    print("Dataset: {} samples, {} features".format(X_scaled.shape[0], X_scaled.shape[1]))

    # Create model
    model = RandomForestClassifier(n_estimators=100, random_state=42)

    # Run stability analysis
    scores = run_all_perturbations(model, X_scaled, y, n_iter=5)
    overall_score = np.mean(list(scores.values()))

    # Display results
    print("STABILITY RESULTS:")
    print("Overall Stability Score: {:.4f}".format(overall_score))
    print("4 Perturbation Scores:")
    for perturbation, score in scores.items():
        print("  {}: {:.4f}".format(perturbation, score))

    print("Package test completed successfully!")

if __name__ == "__main__":
    main()
