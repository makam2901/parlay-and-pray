import pandas as pd

def load_and_preprocess(path: str) -> pd.DataFrame:
    # Load dataset
    df = pd.read_csv(path, header=None, na_values="?")

    # Assign column names
    df.columns = [
        "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
        "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"
    ]

    # Drop rows with missing values
    df.dropna(inplace=True)

    # Binary classification: 0 = no disease, 1 = disease
    df["target"] = df["target"].apply(lambda x: 1 if x > 0 else 0)

    return df