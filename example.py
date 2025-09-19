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
    
    print("🔬 FinalStabilityScore Package Demo")
    print("="*40)
    
    # Load and prepare data
    iris = load_iris()
    X, y = iris.data, iris.target
    
    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    print(f"Dataset: {X_scaled.shape[0]} samples, {X_scaled.shape[1]} features")
    
    # Create a model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    
    # Run stability analysis with 4 perturbations
    print("\n🔄 Running stability analysis...")
    scores = run_all_perturbations(model, X_scaled, y, n_iter=10)
    
    # Calculate overall stability score
    overall_score = np.mean(list(scores.values()))
    
    # Display results
    print(f"\n📊 STABILITY RESULTS:")
    print(f"Overall Stability Score: {overall_score:.4f}")
    print(f"\n4 Perturbation Scores:")
    for perturbation, score in scores.items():
        print(f"  • {perturbation}: {score:.4f}")
    
    # Create results dictionary for dashboard
    results = {
        "baseline": overall_score,
        "scores": scores,
        "stability_overall": overall_score
    }
    
    # Generate radar plot
    print(f"\n📈 Generating Radar Plot...")
    plot_radar("RandomForest", scores)
    
    # Generate dashboard (text summary + radar plot)
    print(f"\n📋 Generating Dashboard...")
    plot_dashboard("RandomForest", results)
    
    print(f"\n✅ Analysis complete!")
    print(f"   • Overall stability score: {overall_score:.4f}")
    print(f"   • 4 perturbation scores calculated")
    print(f"   • Radar plot visualization created")
    print(f"   • Dashboard (text summary + radar plot) generated")

if __name__ == "__main__":
    main()
