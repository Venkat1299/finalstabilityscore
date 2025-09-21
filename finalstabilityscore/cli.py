import argparse
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import roc_auc_score, mean_squared_error, r2_score
from .perturbations import run_all_perturbations
from .plots import plot_dashboard

def create_model_function(task_type):
    """Create a model function based on task type."""
    if task_type == "regression":
        def model_fn(X, y):
            model = LinearRegression().fit(X, y)
            return r2_score(y, model.predict(X))
        return model_fn
    else:  # classification
        def model_fn(X, y):
            model = LogisticRegression().fit(X, y)
            if len(np.unique(y)) == 2:  # binary classification
                return roc_auc_score(y, model.predict_proba(X)[:, 1])
            else:  # multiclass - use accuracy
                from sklearn.metrics import accuracy_score
                return accuracy_score(y, model.predict(X))
        return model_fn

def main():
    parser = argparse.ArgumentParser(description="Evaluate model stability under perturbations")
    parser.add_argument("--input", required=True, help="Path to CSV file")
    parser.add_argument("--target", required=True, help="Target column name")
    parser.add_argument("--task", choices=["classification", "regression"], default="classification",
                       help="Task type (default: classification)")
    parser.add_argument("--n_iter", type=int, default=10, help="Number of iterations (default: 10)")
    args = parser.parse_args()

    # Load and preprocess data
    df = pd.read_csv(args.input)
    X = df.drop(columns=[args.target])
    y = df[args.target]

    # Encode categorical variables
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    for col in X.select_dtypes(include="object").columns:
        X[col] = LabelEncoder().fit_transform(X[col])
    if y.dtype == "object":
        y = LabelEncoder().fit_transform(y)
    X = StandardScaler().fit_transform(X)

    # Create model function
    model_fn = create_model_function(args.task)

    # Create sklearn model for perturbations
    if args.task == "regression":
        model = LinearRegression()
    else:
        model = LogisticRegression()

    # Run stability analysis
    scores = run_all_perturbations(model, X, y, n_iter=args.n_iter)
    overall_score = np.mean(list(scores.values()))

    # Create results dictionary for dashboard
    results = {
        "baseline": overall_score,
        "scores": scores,
        "stability_overall": overall_score
    }

    # Generate dashboard
    plot_dashboard("CLI_Run", results)

    # Print summary
    print(f"\n📊 CLI Stability Analysis Results")
    print(f"Dataset: {args.input}")
    print(f"Task: {args.task}")
    print(f"Overall Stability Score: {overall_score:.4f}")
    print(f"\nPerturbation Scores:")
    for perturbation, score in scores.items():
        print(f"  • {perturbation}: {score:.4f}")

if __name__ == "__main__":
    main()
