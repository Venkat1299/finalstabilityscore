import argparse
import pandas as pd
from sklearn.linear_model import LogisticRegression, LinearRegression
from .core import stability_framework
from .metrics import classification_auc, regression_r2
from .utils import preprocess_dataset
from .plots import plot_dashboard

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--task", choices=["classification", "regression"], default="classification")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    X, y = preprocess_dataset(df, args.target)

    if args.task == "regression":
        model_fn = lambda X, y: regression_r2(X, y, LinearRegression().fit(X, y))
    else:
        model_fn = lambda X, y: classification_auc(X, y, LogisticRegression().fit(X, y))

    res = stability_framework(X, y, model_fn)
    plot_dashboard("CLI_Run", res)

if __name__ == "__main__":
    main()
