from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from io import BytesIO
import base64

# Scikit-learn libraries
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score

app = Flask(__name__)


# Load and preprocess dataset once at startup
def load_data():
    df = pd.read_csv('static/car_details_from_car_dehkho.csv')
    return df


def preprocess_data(df, selected_features):
    # I will drop 'name' and the target from features.
    # The target is selling_price.
    # For categorical features, I create dummy variables.
    categorical_features = ['fuel', 'seller_type', 'transmission', 'owner']
    # Only encode categorical features that are among the selected features
    # In this example, if the user selects categorical features, I will encode them.
    df_encoded = pd.get_dummies(df, columns=categorical_features, drop_first=True)

    # Prepare feature list (remove target and irrelevant columns)
    X = df_encoded.drop(['selling_price', 'name'], axis=1)
    y = df_encoded['selling_price']

    # If the user did not choose all features, filter the X dataframe
    if selected_features:
        # It is assumed that the selected_features list contains columns in X.
        X = X[selected_features]

    return X, y


# Global dataset load
df_original = load_data()


@app.route('/', methods=["GET", "POST"])
def index():
    # Define the features available for modeling (after encoding for categorical variables).
    # I show the original numeric columns and some encoded features (if they exist)
    # First, I create the dummy variables from the original data to get full column names
    df_encoded = pd.get_dummies(df_original, columns=['fuel', 'seller_type', 'transmission', 'owner'], drop_first=True)
    # Remove columns that should not be selectable
    available_features = list(df_encoded.drop(['selling_price', 'name'], axis=1).columns)

    # Sort them for a cleaner UI
    available_features.sort()

    if request.method == "POST":
        # Get polynomial degree from form; default to 1 if not provided.
        poly_degree = int(request.form.get("poly_degree", 1))

        # Get list of selected features from form (as a list of strings)
        selected_features = request.form.getlist("features")
        if not selected_features:
            # If none selected, use all available features
            selected_features = available_features

        # Preprocess data using the selected features
        X, y = preprocess_data(df_original, selected_features)
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Apply polynomial transformation if degree > 1
        if poly_degree > 1:
            poly_transformer = PolynomialFeatures(degree=poly_degree, include_bias=False)
            X_train_trans = poly_transformer.fit_transform(X_train_scaled)
            X_test_trans = poly_transformer.transform(X_test_scaled)
        else:
            # For degree 1, no transformation is necessary.
            X_train_trans = X_train_scaled
            X_test_trans = X_test_scaled

        # Train linear regression model
        model = LinearRegression()
        model.fit(X_train_trans, y_train)
        y_pred = model.predict(X_test_trans)

        # Evaluate model performance
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # Generate a plot of Actual vs Predicted for visualization
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
        plot_url = base64.b64encode(img.getvalue()).decode('utf8')

        # Prepare metrics to send to the results page
        metrics = {
            'mse': f"{mse:.2f}",
            'r2': f"{r2:.2f}",
            'poly_degree': poly_degree,
            'features_used': selected_features
        }

        return render_template('results.html', metrics=metrics, plot_url=plot_url)
    # For GET, render the main form page.
    return render_template('index.html', available_features=available_features)


if __name__ == "__main__":
    app.run(debug=True)
