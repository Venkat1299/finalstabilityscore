import logging
from sklearn.preprocessing import LabelEncoder, StandardScaler

logging.basicConfig(level=logging.INFO)

def log_info(msg):
    logging.info(msg)

def preprocess_dataset(df, target_col):
    X = df.drop(columns=[target_col])
    y = df[target_col]
    for col in X.select_dtypes(include="object").columns:
        X[col] = LabelEncoder().fit_transform(X[col])
    if y.dtype == "object":
        y = LabelEncoder().fit_transform(y)
    X = StandardScaler().fit_transform(X)
    return X, y
