"""
data/generator.py
Dedicated module for synthetic dataset generation and custom CSV ingestion/preprocessing
built with NumPy and Pandas.
"""
import io
import numpy as np
import pandas as pd


def generate_synthetic_data(preset: str = "moons", n_samples: int = 240, noise: float = 0.18, random_state: int = 42):
    """
    Generates synthetic datasets for classification experiments.
    Returns:
        df (pd.DataFrame): Full dataset including features and target column
        feature_names (list[str]): Names of feature columns
        target_name (str): Name of target column
        class_names (list[str]): Human-readable class labels
    """
    rng = np.random.RandomState(random_state)
    target_name = "Target"
    n_half = n_samples // 2
    n_rem = n_samples - n_half

    if preset == "moons":
        theta0 = np.linspace(0, np.pi, n_half)
        theta1 = np.linspace(0, np.pi, n_rem)
        x0 = np.column_stack([np.cos(theta0), np.sin(theta0)]) + rng.normal(0, noise, size=(n_half, 2))
        x1 = np.column_stack([1.0 - np.cos(theta1), 0.5 - np.sin(theta1)]) + rng.normal(0, noise, size=(n_rem, 2))
        X = np.vstack([x0, x1])
        y = np.hstack([np.zeros(n_half, dtype=int), np.ones(n_rem, dtype=int)])
        feature_names = ["Curvature_X", "Curvature_Y"]
        class_names = ["Moon A (0)", "Moon B (1)"]

    elif preset == "circles":
        theta0 = rng.uniform(0, 2 * np.pi, n_half)
        theta1 = rng.uniform(0, 2 * np.pi, n_rem)
        r0 = 1.0 + rng.normal(0, noise, size=n_half)
        r1 = 0.42 + rng.normal(0, noise * 0.8, size=n_rem)
        x0 = np.column_stack([r0 * np.cos(theta0), r0 * np.sin(theta0)])
        x1 = np.column_stack([r1 * np.cos(theta1), r1 * np.sin(theta1)])
        X = np.vstack([x0, x1])
        y = np.hstack([np.zeros(n_half, dtype=int), np.ones(n_rem, dtype=int)])
        feature_names = ["Radius_X", "Radius_Y"]
        class_names = ["Outer Ring (0)", "Inner Core (1)"]

    elif preset == "blobs":
        spread = 0.65 + noise * 2.5
        x0 = rng.normal(loc=[-1.4, -1.2], scale=spread, size=(n_half, 2))
        x1 = rng.normal(loc=[1.4, 1.2], scale=spread, size=(n_rem, 2))
        X = np.vstack([x0, x1])
        y = np.hstack([np.zeros(n_half, dtype=int), np.ones(n_rem, dtype=int)])
        feature_names = ["Dimension_1", "Dimension_2"]
        class_names = ["Cluster Alpha (0)", "Cluster Beta (1)"]

    elif preset == "xor":
        X = rng.uniform(-2.0, 2.0, size=(n_samples, 2))
        y = ((X[:, 0] * X[:, 1]) > 0).astype(int)
        X += rng.normal(0, noise, size=X.shape)
        feature_names = ["Coord_U", "Coord_V"]
        class_names = ["Quadrant Even (0)", "Quadrant Odd (1)"]

    elif preset == "churn":
        age = rng.uniform(20, 70, size=n_samples) + rng.normal(0, noise * 10, size=n_samples)
        usage = rng.uniform(5, 85, size=n_samples) + rng.normal(0, noise * 15, size=n_samples)
        tickets = np.clip(np.round(rng.uniform(0, 7, size=n_samples) + rng.normal(0, noise * 2, size=n_samples)), 0, 12)
        logit = -3.0 + 0.045 * age - 0.04 * usage + 0.75 * tickets + rng.normal(0, noise * 2, size=n_samples)
        probs = 1.0 / (1.0 + np.exp(-logit))
        y = (probs >= 0.5).astype(int)
        X = np.column_stack([age, usage, tickets])
        feature_names = ["Customer_Age", "Monthly_Usage_Hrs", "Support_Tickets"]
        target_name = "Churn_Status"
        class_names = ["Retained (0)", "Churned (1)"]

    else:
        X = rng.normal(0, 1.2, size=(n_samples, 3))
        logits = 1.2 * X[:, 0] - 0.9 * X[:, 1] + 0.6 * X[:, 2] + rng.normal(0, noise * 2, size=n_samples)
        y = (logits >= 0).astype(int)
        feature_names = ["Feature_1", "Feature_2", "Feature_3"]
        class_names = ["Class 0", "Class 1"]

    # Shuffle dataset rows
    perm = rng.permutation(len(y))
    X = X[perm]
    y = y[perm]

    df = pd.DataFrame(X, columns=feature_names)
    df[target_name] = y
    return df, feature_names, target_name, class_names


def load_and_preprocess_csv(csv_content: str, target_col: str = None, feature_cols: list = None):
    """
    Parses a CSV string, cleans missing values with column medians, and encodes binary targets.
    """
    df = pd.read_csv(io.StringIO(csv_content))
    df = df.dropna(how="all").reset_index(drop=True)

    if df.shape[1] < 2:
        raise ValueError("Dataset must contain at least 1 feature column and 1 target column.")

    if not target_col or target_col not in df.columns:
        target_col = df.columns[-1]

    if not feature_cols:
        feature_cols = [c for c in df.columns if c != target_col]
    else:
        feature_cols = [c for c in feature_cols if c in df.columns and c != target_col]

    # Convert features to numeric and impute missing values with column median
    X_df = df[feature_cols].apply(pd.to_numeric, errors="coerce")
    X_df = X_df.fillna(X_df.median(numeric_only=True)).fillna(0.0)

    # Encode target column into binary {0, 1}
    raw_target = df[target_col].astype(str).str.strip()
    unique_vals = sorted(raw_target.unique())

    if len(unique_vals) >= 2:
        pos_label = unique_vals[1]
        y = (raw_target == pos_label).astype(int).values
        class_names = [f"{unique_vals[0]} (0)", f"{unique_vals[1]} (1)"]
    else:
        y = pd.to_numeric(raw_target, errors="coerce").fillna(0).astype(int).values
        class_names = ["Class 0", "Class 1"]

    clean_df = X_df.copy()
    clean_df[target_col] = y

    return clean_df, list(X_df.columns), target_col, class_names
