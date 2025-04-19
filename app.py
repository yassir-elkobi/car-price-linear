import datetime

from flask import Flask, render_template, request, flash
import pandas as pd
import numpy as np
import matplotlib
import math
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from sklearn.preprocessing import PolynomialFeatures

matplotlib.use('Agg')
app = Flask(__name__)
app.secret_key = 'secret_key'


# ======= DATA LOADING & PREPROCESSING FUNCTIONS =======

def load_data(csv_path='static/car_details_from_car_dehkho.csv'):
    """Load CSV dataset."""
    try:
        df = pd.read_csv(csv_path)
        return df
    except Exception as e:
        raise RuntimeError(f"Error loading data: {e}")


def get_available_features(df):
    """
    Generate a list of available features after:
    1. Performing feature engineering (car_age, km_per_year)
    2. Encoding categorical columns
    3. Removing unwanted columns (name, selling_price)
    
    Returns a sorted list of feature names.
    """
    # Create a copy to avoid modifying the original dataframe
    df = df.copy()

    # One-hot encode categorical features
    categorical_features = ['fuel', 'seller_type', 'transmission', 'owner']
    df_encoded = pd.get_dummies(df, columns=categorical_features, drop_first=True)

    # Get feature list, excluding target and identifier columns
    available_features = list(df_encoded.drop(['selling_price', 'name'], axis=1).columns)
    available_features.sort()

    return available_features


def preprocess_data(df, selected_features):
    """
    Preprocess the data:
      - Add engineered features (car_age, km_per_year)
      - Encode categorical features
      - Drop columns not needed ('name' and target variable)
      - Filter the features based on user selection
    Returns features X and target y.
    """
    # Create a copy to avoid modifying the original
    df = df.copy()

    # Encode categorical features
    categorical_features = ['fuel', 'seller_type', 'transmission', 'owner']
    df_encoded = pd.get_dummies(df, columns=categorical_features, drop_first=True)

    # Separate features and target
    X = df_encoded.drop(['selling_price', 'name'], axis=1)
    y = df_encoded['selling_price']

    # If the user selected a subset, filter X accordingly
    if selected_features:
        # Make sure all selected features exist in X
        valid_features = [f for f in selected_features if f in X.columns]
        if len(valid_features) != len(selected_features):
            missing = set(selected_features) - set(valid_features)
            print(f"Warning: Features {missing} not found in dataset")
        X = X[valid_features]

    return X, y


def train_test_split(X, y, test_size=0.2, random_state=42):
    """Split the data into training and testing sets."""
    np.random.seed(random_state)
    indices = np.random.permutation(len(X))
    test_size = int(len(X) * test_size)
    test_indices = indices[:test_size]
    train_indices = indices[test_size:]

    X_train = X.iloc[train_indices]
    X_test = X.iloc[test_indices]
    y_train = y.iloc[train_indices]
    y_test = y.iloc[test_indices]

    return X_train, X_test, y_train, y_test


# ======= FEATURE SCALING FUNCTIONS =======

def mean_normalization(X):
    """
    Apply mean normalization: (x - mean) / (max - min)
    """
    # Convert to float64 to ensure proper numeric operations
    X_values = np.array(X, dtype=np.float64)

    # Calculate statistics
    mean = np.mean(X_values, axis=0)
    min_val = np.min(X_values, axis=0)
    max_val = np.max(X_values, axis=0)
    range_val = max_val - min_val

    # Avoid division by zero
    range_val[range_val < 1e-10] = 1.0

    # Normalize
    X_norm = (X_values - mean) / range_val

    return X_norm, mean, range_val


def standardization(X):
    """
    Apply standardization: (x - mean) / std
    """
    # Convert to float64 to ensure proper numeric operations
    X_values = np.array(X, dtype=np.float64)

    # Calculate mean and std
    mean = np.mean(X_values, axis=0)
    std = np.std(X_values, axis=0)

    # Avoid division by zero
    std[std < 1e-10] = 1.0

    # Standardize
    X_std = (X_values - mean) / std

    return X_std, mean, std


def apply_feature_scaling(X_train, X_test, scaling_method='none'):
    """
    Apply the specified feature scaling method to the data.
    Returns scaled data and scaling parameters.
    """
    # Convert to numpy arrays first
    X_train_values = np.array(X_train.values if hasattr(X_train, 'values') else X_train, dtype=np.float64)
    X_test_values = np.array(X_test.values if hasattr(X_test, 'values') else X_test, dtype=np.float64)

    if scaling_method == 'none':
        return X_train_values, X_test_values, None

    if scaling_method == 'mean_normalization':
        X_train_scaled, mean, range_val = mean_normalization(X_train_values)
        X_test_scaled = (X_test_values - mean) / range_val
        scaling_params = {'method': 'mean_normalization', 'mean': mean, 'range': range_val}
    elif scaling_method == 'standardization':
        X_train_scaled, mean, std = standardization(X_train_values)
        X_test_scaled = (X_test_values - mean) / std
        scaling_params = {'method': 'standardization', 'mean': mean, 'std': std}

    return X_train_scaled, X_test_scaled, scaling_params


# ======= POLYNOMIAL FEATURES =======

def create_polynomial_features(X, degree):
    """
    Create polynomial features up to the specified degree using scikit-learn.
    """
    if degree == 1:
        return X

    # Ensure X is a proper numeric array
    X = np.array(X, dtype=np.float64)

    # Use scikit-learn's PolynomialFeatures
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    X_poly = poly.fit_transform(X)

    return X_poly


# ======= LINEAR REGRESSION WITH GRADIENT DESCENT =======

def compute_cost(X, y, theta, lambda_=0):
    """
    Compute the cost function for linear regression.
    Includes regularization when lambda_ > 0.
    """
    m = len(y)
    predictions = X.dot(theta)
    errors = predictions - y

    # Calculate error term with protection against overflow
    squared_errors = np.zeros_like(errors)
    for i in range(len(errors)):
        squared_errors[i] = errors[i] ** 2

    error_sum = np.sum(squared_errors)

    # Calculate regularization term with protection against overflow
    reg_term = 0
    if lambda_ > 0:
        # Don't regularize the bias term (theta[0])
        reg_sum = 0
        for i in range(1, len(theta)):
            reg_sum += theta[i] ** 2
        reg_term = (lambda_ / (2 * m)) * reg_sum

    return (1 / (2 * m)) * error_sum + reg_term


def gradient_descent(X, y, theta, alpha, num_iters, lambda_=0):
    """
    Run gradient descent to optimize theta.
    Includes regularization when lambda_ > 0.
    Returns the final theta and cost history.
    """
    m = len(y)
    # Ensure all inputs are numeric arrays
    X = np.array(X, dtype=np.float64)
    y = np.array(y, dtype=np.float64)
    theta = np.array(theta, dtype=np.float64)
    alpha = float(alpha)
    lambda_ = float(lambda_)

    cost_history = []
    theta_history = []

    for _ in range(num_iters):
        try:

            predictions = X.dot(theta)
            errors = predictions - y

            # Gradient calculation with regularization
            gradient = (1 / m) * X.T.dot(errors)

            if lambda_ > 0:
                # Don't regularize the bias term
                regularization = (lambda_ / m) * theta
                regularization[0] = 0  # Don't regularize bias
                gradient += regularization

            # Calculate cost with numeric protection
            current_cost = compute_cost(X, y, theta, lambda_)
            cost_history.append(current_cost)
            theta_history.append(theta.copy())

        except Exception as e:
            print(f"Warning: Error in gradient descent iteration: {e}")
            # Keep going with current theta
            cost_history.append(1e10 if not cost_history else cost_history[-1])
            theta_history.append(theta.copy())

    return theta, cost_history, theta_history


def predict(X, theta):
    """
    Make predictions using the linear model.
    """
    return X.dot(theta)


# ======= CROSS VALIDATION =======

def k_fold_cross_validation(X, y, k=5, alpha=0.01, num_iters=1000, lambda_=0):
    """
    Perform k-fold cross-validation.
    Returns average MSE across folds.
    """
    # Ensure proper numeric arrays
    X = np.array(X, dtype=np.float64)
    y = np.array(y, dtype=np.float64)

    m = len(y)
    fold_size = m // k
    np.random.seed(42)
    indices = np.random.permutation(m)
    mse_scores = []

    for i in range(k):
        start_idx = i * fold_size
        end_idx = start_idx + fold_size if i < k - 1 else m

        test_indices = indices[start_idx:end_idx]
        train_indices = np.concatenate([indices[:start_idx], indices[end_idx:]])

        X_train_fold = X[train_indices]
        y_train_fold = y[train_indices]
        X_test_fold = X[test_indices]
        y_test_fold = y[test_indices]

        # Add bias term
        X_train_fold_bias = np.hstack((np.ones((X_train_fold.shape[0], 1)), X_train_fold))
        X_test_fold_bias = np.hstack((np.ones((X_test_fold.shape[0], 1)), X_test_fold))

        # Initialize theta
        theta = np.zeros(X_train_fold_bias.shape[1], dtype=np.float64)

        # Train model with gradient descent
        theta, _, _ = gradient_descent(X_train_fold_bias, y_train_fold, theta, alpha, num_iters, lambda_)

        # Predict and calculate MSE
        y_pred = predict(X_test_fold_bias, theta)
        mse = np.mean((y_pred - y_test_fold) ** 2)
        mse_scores.append(mse)

    return np.mean(mse_scores)


# ======= PLOTTING FUNCTIONS =======

def plot_actual_vs_predicted(y_test, y_pred):
    """
    Plot actual vs predicted values.
    """
    img = BytesIO()
    plt.figure(figsize=(8, 5))

    # Use alpha for better visualization with overlapping points
    plt.scatter(y_test, y_pred, alpha=0.5, color='blue')

    # Add reference line (perfect predictions)
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')

    # Format axes for better readability
    plt.ticklabel_format(style='plain', axis='both', useOffset=False)

    # Labels and title
    plt.xlabel('Actual Selling Price')
    plt.ylabel('Predicted Selling Price')
    plt.title('Actual vs Predicted Selling Price')

    plt.tight_layout()
    plt.savefig(img, format='png')
    plt.close()
    img.seek(0)
    return base64.b64encode(img.getvalue()).decode('utf8')


def plot_cost_history(cost_history):
    """
    Plot the cost function value vs iterations for gradient descent.
    """
    img = BytesIO()
    plt.figure(figsize=(8, 5))
    plt.plot(cost_history, color='green')
    plt.xlabel("Iteration")
    plt.ylabel("Cost")
    plt.title("Cost Function History")
    plt.tight_layout()
    plt.savefig(img, format='png')
    plt.close()
    img.seek(0)
    return base64.b64encode(img.getvalue()).decode('utf8')


def plot_gradient_descent_path(theta_history, cost_history=None):
    """
    Visualize the gradient descent path for the first two parameters.
    """
    # Only visualize the first two parameters (bias term and one feature)
    theta_history = np.array(theta_history)
    img = BytesIO()

    if theta_history.shape[1] >= 2:
        plt.figure(figsize=(8, 5))
        plt.plot(theta_history[:, 0], theta_history[:, 1], 'o-', color='red', markersize=3)
        plt.xlabel('Theta 0 (Bias)')
        plt.ylabel('Theta 1')
        plt.title("Gradient Descent Path")
        plt.tight_layout()
        plt.savefig(img, format='png')
        plt.close()
        img.seek(0)
        return base64.b64encode(img.getvalue()).decode('utf8')
    return None


# ======= METRICS =======

def calculate_metrics(y_true, y_pred):
    """
    Calculate regression metrics:
    - Mean Squared Error (MSE)
    - R-squared (R²)
    """
    # Protect against overflow in squared error calculations
    errors = y_true - y_pred
    squared_errors = np.zeros_like(errors)
    for i in range(len(errors)):
        if abs(errors[i]) > 1e5:
            squared_errors[i] = 1e10  # Cap extremely large errors
        else:
            squared_errors[i] = errors[i] * errors[i]

    mse = np.mean(squared_errors)

    # R² = 1 - (SSres / SStot)
    ss_res = np.sum(squared_errors)

    # Calculate total sum of squares with protection
    y_mean = np.mean(y_true)
    deviations = y_true - y_mean
    squared_deviations = np.zeros_like(deviations)
    for i in range(len(deviations)):
        if abs(deviations[i]) > 1e5:
            squared_deviations[i] = 1e10
        else:
            squared_deviations[i] = deviations[i] * deviations[i]

    ss_tot = np.sum(squared_deviations)

    # Protect against division by zero
    if ss_tot == 0:
        r2 = 0
    else:
        r2 = 1 - (ss_res / ss_tot)

    return mse, r2


# ======= GLOBAL DATA LOAD =======
df_original = load_data()
available_features = get_available_features(df_original)


# ======= FLASK ROUTES =======
@app.route('/', methods=["GET", "POST"])
def index():
    if request.method == "POST":
        try:
            # Get algorithm type (linear or polynomial)
            algorithm_type = request.form.get("algorithm_type", "linear")

            # Get polynomial degree
            poly_degree = int(request.form.get("poly_degree", 1))
            if poly_degree < 1 or poly_degree > 4:
                flash("Polynomial degree must be between 1 and 4. Using degree=1 as default.")
                poly_degree = 1

            # Get regularization setting
            use_regularization = request.form.get("use_regularization") == "on"
            lambda_ = float(request.form.get("lambda", 0.1)) if use_regularization else 0

            # Get cross-validation setting
            use_cv = request.form.get("use_cv") == "on"

            # Get feature scaling method
            scaling_method = request.form.get("scaling_method", "none")

            # Get selected features
            selected_features = request.form.getlist("features")
            if not selected_features:
                selected_features = available_features

            # Hyperparameters for gradient descent
            alpha = float(request.form.get("alpha", 0.01))
            num_iters = int(request.form.get("num_iters", 1000))

            # Preprocess and split data
            X, y = preprocess_data(df_original, selected_features)
            X_train, X_test, y_train, y_test = train_test_split(X, y)

            # Apply feature scaling
            X_train_scaled, X_test_scaled, scaling_params = apply_feature_scaling(X_train, X_test, scaling_method)

            # Convert to numpy arrays if not already
            y_train_np = y_train.values if isinstance(y_train, pd.Series) else y_train
            y_test_np = y_test.values if isinstance(y_test, pd.Series) else y_test

            # Scale target variables to avoid numeric overflow
            y_scale = max(np.max(np.abs(y_train_np)), 1e5)  # Ensure sufficient scaling
            y_train_scaled = y_train_np / y_scale
            y_test_scaled = y_test_np / y_scale

            # Adjust alpha based on scaling
            adjusted_alpha = alpha / (y_scale / 1e5)  # Scale alpha inversely with y_scale

            # Create polynomial features if needed
            if algorithm_type == "polynomial" and poly_degree > 1:
                X_train_poly = create_polynomial_features(X_train_scaled, poly_degree)
                X_test_poly = create_polynomial_features(X_test_scaled, poly_degree)
            else:
                X_train_poly = X_train_scaled
                X_test_poly = X_test_scaled

            # Add bias term (intercept)
            X_train_final = np.hstack((np.ones((X_train_poly.shape[0], 1)), X_train_poly))
            X_test_final = np.hstack((np.ones((X_test_poly.shape[0], 1)), X_test_poly))

            # Initialize theta (parameters)
            theta = np.zeros(X_train_final.shape[1])

            # Cross-validation if enabled
            cv_mse = None
            if use_cv:
                cv_mse = k_fold_cross_validation(
                    X_train_poly, y_train_scaled, k=5,
                    alpha=adjusted_alpha, num_iters=num_iters,
                    lambda_=lambda_
                )
                # Convert CV MSE back to original scale
                cv_mse = cv_mse * (y_scale ** 2)

            # Train the model with gradient descent
            theta, cost_history, theta_history = gradient_descent(
                X_train_final, y_train_scaled, theta,
                adjusted_alpha, num_iters, lambda_
            )

            # Make predictions (and rescale back to original values)
            y_pred_scaled = predict(X_test_final, theta)
            y_pred = y_pred_scaled * y_scale

            # Calculate metrics
            mse, r2 = calculate_metrics(y_test_np, y_pred)

            # Create visualizations
            plot_url = plot_actual_vs_predicted(y_test_np, y_pred)
            cost_plot_url = plot_cost_history(cost_history)
            gd_plot_url = plot_gradient_descent_path(theta_history)

            # Prepare metrics for display
            metrics = {
                'mse': f"{mse:.2f}",
                'r2': f"{r2:.2f}",
                'poly_degree': poly_degree,
                'features_used': selected_features,
                'algorithm_type': algorithm_type,
                'regularization': use_regularization,
                'lambda': lambda_ if use_regularization else None,
                'cross_validation': use_cv,
                'cv_mse': f"{cv_mse:.2f}" if cv_mse is not None else None,
                'scaling_method': scaling_method,
                'alpha': alpha,
                'num_iters': num_iters
            }

            return render_template("results.html",
                                   metrics=metrics,
                                   plot_url=plot_url,
                                   cost_plot_url=cost_plot_url,
                                   gd_plot_url=gd_plot_url)

        except Exception as e:
            flash(f"An error occurred: {str(e)}")
            import traceback
            traceback.print_exc()

    return render_template("index.html", available_features=available_features)


# ======= MAIN =======
if __name__ == "__main__":
    # Print the features in the dataset
    df = load_data()
    print("Features available in the dataset:")
    print(df.columns.tolist())
    app.run(debug=True)
