import datetime
import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures


def load_data(path: str = "static/car_details_from_car_dehkho.csv") -> pd.DataFrame:
    """
    Load the car dataset from the specified CSV file.
    
    Args:
        path: Path to the CSV file
        
    Returns:
        A pandas DataFrame containing the loaded data
    """
    df = pd.read_csv(path)
    return df


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add engineered features to the dataset: car_age, km_per_year, and brand.
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with added engineered features
    """
    out = df.copy()
    this_year = datetime.datetime.now().year
    out["car_age"] = this_year - out["year"]
    out.loc[out["car_age"] == 0, "car_age"] = 1  # avoid 0‑div for new cars
    out["km_per_year"] = out["km_driven"] / out["car_age"]
    out["brand"] = out["name"].str.split().str[0]
    return out


CATEGORICAL_RAW = ["fuel", "seller_type", "transmission", "owner", "brand"]


def get_available_features(df: pd.DataFrame) -> list[str]:
    """
    Get a list of all available features in the dataset after preprocessing.
    
    Args:
        df: Input DataFrame
        
    Returns:
        List of feature names
    """
    df = add_engineered_features(df)
    df_enc = pd.get_dummies(df, columns=CATEGORICAL_RAW, drop_first=True)
    return sorted(df_enc.drop(columns=["selling_price", "name"]).columns)


def preprocess_data(df: pd.DataFrame, selected: list[str]):
    """
    Preprocess the data by adding engineered features and encoding categorical variables.
    
    Args:
        df: Input DataFrame
        selected: List of selected feature names
        
    Returns:
        X_all: Feature matrix
        y_log: Log-transformed target variable
    """
    df = add_engineered_features(df)
    df_enc = pd.get_dummies(df, columns=CATEGORICAL_RAW, drop_first=True)

    X_all = df_enc.drop(columns=["selling_price", "name"])
    y_log = np.log1p(df_enc["selling_price"]).astype(np.float64)

    if selected:
        X_all = X_all[[c for c in selected if c in X_all.columns]]
    return X_all, y_log


def standard_scale(df_train: pd.DataFrame, df_test: pd.DataFrame):
    """
    Apply Z-score normalization to numeric columns; dummy variables remain unchanged.
    
    Args:
        df_train: Training data
        df_test: Testing data
        
    Returns:
        Scaled training data, scaled testing data, and list of numeric columns
    """
    tr, te = df_train.copy(), df_test.copy()
    num_cols = tr.select_dtypes(include=[np.number]).columns
    mean = tr[num_cols].mean()
    std = tr[num_cols].std().replace(0, 1)
    tr[num_cols] = (tr[num_cols] - mean) / std
    te[num_cols] = (te[num_cols] - mean) / std
    return tr, te, num_cols


def train_test_split(X, y, test_size=0.2, seed=42):
    """
    Split data into training and testing sets.
    
    Args:
        X: Feature matrix
        y: Target vector
        test_size: Fraction of data to use for testing
        seed: Random seed for reproducibility
        
    Returns:
        X_train, X_test, y_train, y_test
    """
    np.random.seed(seed)
    idx = np.random.permutation(len(X))
    cut = int(len(X) * test_size)
    te, tr = idx[:cut], idx[cut:]
    return X.iloc[tr], X.iloc[te], y.iloc[tr], y.iloc[te]


def design_matrix_builder(X_df: pd.DataFrame, numeric_cols: list[str], degree: int):
    """
    Create a function that transforms data into a design matrix for linear regression.
    
    Args:
        X_df: DataFrame containing features
        numeric_cols: List of numeric column names
        degree: Polynomial degree for feature expansion
        
    Returns:
        Function that transforms a DataFrame into a design matrix
    """
    poly = PolynomialFeatures(degree=degree, include_bias=False) if degree > 1 else None

    def transform(df: pd.DataFrame):
        num = df[numeric_cols].to_numpy(dtype=np.float64)
        num_poly = poly.fit_transform(num) if poly else num
        dummies = df.drop(columns=numeric_cols).to_numpy(dtype=np.float64)
        return np.hstack([np.ones((len(df), 1)), num_poly, dummies])

    return transform
