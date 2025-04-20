# Car Price Predictor with Gradient Descent

**Car Price Predictor with Gradient Descent** is a Flask web application that predicts the selling price of a used car
based on various features. It implements both linear and polynomial regression models *from scratch* using gradient
descent (instead of relying on scikit-learn for model training). This project is part of a machine learning learning
journey, aiming to solidify understanding of regression algorithms, gradient descent optimization, and end-to-end model
deployment.

## Features

- **Linear & Polynomial Regression**
- **Gradient Descent Optimization**
- **L2 Regularization (Ridge)**
- **Feature Selection**
- **Feature Scaling Options**
- **Cross-Validation and MSE Evaluation**
- **Visualizations**
- **Interactive Predictions**

## Tech Stack and Tools

- **Python 3**
- **Flask**
- **Pandas**
- **NumPy**
- **Matplotlib**
- **scikit-learn (for preprocessing only)**

## Best Model Configuration

Through experimentation, the following configuration was found to produce the best results for this car price prediction
task:

- **Algorithm Type:** Polynomial Regression (Degree = 2).
- **Regularization:** Enabled L2 regularization (Ridge) with λ = 0.04.
- **Cross-Validation:** Enabled
- **Feature Scaling:** Standardization (Z-score normalization of features) was applied.
- **Gradient Descent Parameters:** Learning Rate α = 0.04, and Number of Iterations = 10,000.

## Dataset Summary

The model is trained on a dataset of used cars (originally provided by
CarDekho.com. The dataset contains **4,340 rows** (car entries) and **8 columns** of features describing each car, along
with the
target selling price. It’s a mix of numerical and categorical data. Key features include:

- **Car Age or Year**
- **Car Brand/Model**
- **Kms Driven (Mileage)**
- **Fuel Type:** Petrol, Diesel, CNG, etc
- **Transmission:** Manual or Automatic
- **Seller Type:** Whether the car is being sold by a **Dealer** or an **Individual (private seller)**
- **Owner** (Number of Previous Owners): Indicates if the car is first-hand, second-hand, etc.

The **target** variable is the *Selling_Price* of the car (the price at which the car is being sold in the used car
market).

## How to Run the App Locally

If you want to run this web app on your local machine, follow these steps:

1. **Clone the repository** or download the project source code to your local machine.
2. **Install Python dependencies:** Make sure you have Python 3 installed, then install the required libraries. You can
   do this by running:
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the Flask application:** In your terminal, navigate to the project directory and run the app. For example:
   ```bash
   python app.py
   ```  
   (Or use `flask run` if the project is set up with Flask CLI). You should see Flask start up and indicate that it’s
   running on a local address (by default http://127.0.0.1:5000 or http://localhost:5000).
4. **Open the web app in your browser:** Launch a web browser and navigate to the local URL (e.g.,
   `http://localhost:5000`). You should see the Car Price Predictor web interface.

## Learning Goals

This project was developed as part of the author’s journey to learn and demonstrate machine learning engineering skills.
The primary learning goal was to **implement regression models from scratch** and integrate them into a full web
application.

**Note:** This README is written to be clear and informative for any new visitor (or recruiter) checking out the
project. It highlights the project’s purpose, capabilities, technical implementation, and the learning outcomes
associated with it.