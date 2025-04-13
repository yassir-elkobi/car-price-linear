from flask import Flask, render_template, request, flash
import pandas as pd
import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from io import BytesIO
import base64

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score

app = Flask(__name__)
app.secret_key = 'your_secret_key'


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
    Generate a list of available features after encoding categorical columns
    and removing unwanted columns.
    """
    categorical_features = ['fuel', 'seller_type', 'transmission', 'owner']
    # Encode categorical features for determining available feature names
    df_encoded = pd.get_dummies(df, columns=categorical_features, drop_first=True)
    available_features = list(df_encoded.drop(['selling_price', 'name'], axis=1).columns)
    available_features.sort()
    return available_features


def preprocess_data(df, selected_features):
    """
    Preprocess the data:
      - Encode categorical features.
      - Drop columns not needed ('name' and target variable).
      - Filter the features based on user selection.
    Returns features X and target y.
    """
    categorical_features = ['fuel', 'seller_type', 'transmission', 'owner']
    df_encoded = pd.get_dummies(df, columns=categorical_features, drop_first=True)
    X = df_encoded.drop(['selling_price', 'name'], axis=1)
    y = df_encoded['selling_price']

    # If the user selected a subset, filter X accordingly.
    if selected_features:
        X = X[selected_features]
    return X, y


def split_and_scale_data(X, y, test_size=0.2, random_state=42):
    """Split the data and scale the features using StandardScaler."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


# ======= POLYNOMIAL TRANSFORMATION =======

def apply_polynomial_transform(X_train_scaled, X_test_scaled, degree):
    """
    Apply polynomial feature transformation if degree > 1.
    Otherwise, return the scaled features as is.
    """
    if degree > 1:
        poly_transformer = PolynomialFeatures(degree=degree, include_bias=False)
        X_train_trans = poly_transformer.fit_transform(X_train_scaled)
        X_test_trans = poly_transformer.transform(X_test_scaled)
        return X_train_trans, X_test_trans, poly_transformer
    else:
        return X_train_scaled, X_test_scaled, None


# ======= REGRESSION & GRADIENT DESCENT =======

def train_regression_model(X_train, y_train):
    """
    Train a linear regression model using scikit-learn.
    """
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def run_gradient_descent(X, y, alpha=0.01, num_iters=1000):
    """
    Run gradient descent to optimize parameters for linear regression.
    X is assumed to already include a bias term.
    Returns the final theta and the cost history.
    """
    m = len(y)
    theta = np.zeros(X.shape[1])
    cost_history = []

    for i in range(num_iters):
        predictions = X.dot(theta)
        errors = predictions - y
        theta -= (alpha / m) * (X.T.dot(errors))
        cost = (1 / (2 * m)) * np.sum(errors ** 2)
        cost_history.append(cost)
    return theta, cost_history


# ======= PLOTTING FUNCTIONS =======

def plot_actual_vs_predicted(y_test, y_pred):
    """
    Plot actual vs predicted selling prices.
    """
    img = BytesIO()
    plt.figure(figsize=(8, 5))
    plt.scatter(y_test, y_pred, alpha=0.5, color='blue')
    plt.xlabel('Actual Selling Price')
    plt.ylabel('Predicted Selling Price')
    plt.title('Actual vs Predicted Selling Price')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
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


def plot_gradient_descent_path(X, y, cost_history, theta_history):
    """
    For a simple scenario (1 feature + bias), visualize the gradient descent path on the cost surface.
    This function only works when there is exactly one feature (X shape = [m, 2] after adding bias).
    """
    if X.shape[1] != 2:
        return None  # Cannot visualize high dimensional parameter space

    # Create a grid for contour plot
    theta0_vals = np.linspace(-50, 50, 100)
    theta1_vals = np.linspace(-50, 50, 100)
    J_vals = np.zeros((len(theta0_vals), len(theta1_vals)))
    m = len(y)

    # Calculate cost for each combination
    for i, t0 in enumerate(theta0_vals):
        for j, t1 in enumerate(theta1_vals):
            t = np.array([t0, t1])
            predictions = X.dot(t)
            errors = predictions - y
            J_vals[i, j] = (1 / (2 * m)) * np.sum(errors ** 2)

    # Plot contour and overlay the gradient descent path
    img = BytesIO()
    plt.figure(figsize=(8, 5))
    T0, T1 = np.meshgrid(theta0_vals, theta1_vals)
    plt.contour(T0, T1, J_vals.T, levels=np.logspace(0, 5, 35), cmap='jet')

    # Extract theta history for plotting
    theta_history = np.array(theta_history)
    plt.plot(theta_history[:, 0], theta_history[:, 1], 'o-', color='red')
    plt.xlabel('Theta 0')
    plt.ylabel('Theta 1')
    plt.title("Gradient Descent Path")
    plt.tight_layout()
    plt.savefig(img, format='png')
    plt.close()
    img.seek(0)
    return base64.b64encode(img.getvalue()).decode('utf8')


# ======= GLOBAL DATA LOAD =======
df_original = load_data()
available_features = get_available_features(df_original)


# ======= FLASK ROUTES =======
@app.route('/', methods=["GET", "POST"])
def index():
    if request.method == "POST":
        try:
            # Validate polynomial degree input (must be an integer in a valid range)
            poly_degree = int(request.form.get("poly_degree", 1))
            max_degree = 4  # maximum allowed degree
            if poly_degree < 1 or poly_degree > max_degree:
                flash(f"Polynomial degree must be between 1 and {max_degree}. Using degree=1 as default.")
                poly_degree = 1
        except ValueError:
            flash("Invalid polynomial degree input. Using degree=1 as default.")
            poly_degree = 1

        # Option to use gradient descent manually (checkbox in form)
        use_gd = request.form.get("use_gd") == "on"

        # Get selected features (default to all if none chosen)
        selected_features = request.form.getlist("features")
        if not selected_features:
            selected_features = available_features

        # Preprocess and split data
        X, y = preprocess_data(df_original, selected_features)
        X_train_scaled, X_test_scaled, y_train, y_test, scaler = split_and_scale_data(X, y)

        # Apply polynomial transformation (if degree > 1)
        X_train_trans, X_test_trans, poly_transformer = apply_polynomial_transform(X_train_scaled, X_test_scaled,
                                                                                   poly_degree)

        # Use scikit-learn model OR gradient descent based on checkbox.
        if use_gd:
            # For gradient descent visualization I require a bias term.
            m_train = X_train_trans.shape[0]
            X_train_gd = np.hstack((np.ones((m_train, 1)), X_train_trans))
            y_train_array = y_train.values if isinstance(y_train, pd.Series) else y_train

            # Hyperparameters for gradient descent
            alpha = 0.01
            num_iters = 1000

            # For capturing theta history (only for visualization when one feature is used after scaling)
            theta_history = []
            m, n = X_train_gd.shape
            theta = np.zeros(n)
            cost_history = []

            for i in range(num_iters):
                predictions = X_train_gd.dot(theta)
                errors = predictions - y_train_array
                theta = theta - (alpha / m) * (X_train_gd.T.dot(errors))
                cost = (1 / (2 * m)) * np.sum(errors ** 2)
                cost_history.append(cost)
                # Save history for visualization for the first two parameters only.
                if n == 2:
                    theta_history.append(theta.copy())

            # Use the final theta to generate predictions on test set.
            m_test = X_test_trans.shape[0]
            X_test_gd = np.hstack((np.ones((m_test, 1)), X_test_trans))
            y_pred = X_test_gd.dot(theta)

            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            # Visualizations: cost function history and (if available) gradient descent path.
            cost_plot_url = plot_cost_history(cost_history)
            gd_plot_url = None
            if X_train_gd.shape[1] == 2 and theta_history:
                gd_plot_url = plot_gradient_descent_path(X_train_gd, y_train_array, cost_history, theta_history)
        else:
            # Train using scikit-learn's LinearRegression
            model = train_regression_model(X_train_trans, y_train)
            y_pred = model.predict(X_test_trans)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            cost_plot_url = None
            gd_plot_url = None

        # Visualize actual vs predicted
        plot_url = plot_actual_vs_predicted(y_test, y_pred)

        # Prepare metrics for display
        metrics = {
            'mse': f"{mse:.2f}",
            'r2': f"{r2:.2f}",
            'poly_degree': poly_degree,
            'features_used': selected_features,
            'used_gradient_descent': use_gd
        }

        return render_template("results.html",
                               metrics=metrics,
                               plot_url=plot_url,
                               cost_plot_url=cost_plot_url,
                               gd_plot_url=gd_plot_url)
    return render_template("index.html", available_features=available_features)


# ======= MAIN =======
if __name__ == "__main__":
    app.run(debug=True)
