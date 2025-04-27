# Car Price Predictor with Linear Regression

**Car Price Predictor with Gradient Descent** is a Flask web application that predicts the selling price of a used car
based on various features. It implements both linear and polynomial regression models *from scratch* using gradient
descent (instead of relying on scikit-learn for model training). This project is part of my machine learning
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
target selling price. It's a mix of numerical and categorical data. Key features include:

- **Car Age or Year**
- **Car Brand/Model**
- **Kms Driven (Mileage)**
- **Fuel Type:** Petrol, Diesel, CNG, etc
- **Transmission:** Manual or Automatic
- **Seller Type:** Whether the car is being sold by a **Dealer** or an **Individual (private seller)**
- **Owner** (Number of Previous Owners): Indicates if the car is first-hand, second-hand, etc.

The **target** variable is the *Selling_Price* of the car (the price at which the car is being sold in the used car
market).

## Project Structure

```
├── app.py                 # Main Flask application
├── model.py               # Linear & polynomial regression model implementation
├── preprocessing.py       # Data loading and preprocessing functions
├── utils.py               # Evaluation metrics and visualization utilities
├── requirements.txt       # Project dependencies
├── static/                # Static files (CSS, data files)
│   └── car_details_from_car_dehkho.csv  # Car dataset
└── templates/             # HTML templates
    ├── index.html         # Settings page
    └── results.html       # Results visualization page
```

## How to Run the App Locally

1. **Clone the repository** or download the project source code to your local machine.
2. **Create a virtual environment:**  
```  
python -m venv .venv  
source .venv/bin/activate  # On Windows: .venv\Scripts\activate  
```
3. **Install Python dependencies:**  
```  
pip install -r requirements.txt  
```
4. **Run the Flask application:**  
```  
python app.py  
```
5. **Open the web app in your browser:** Navigate to <http://127.0.0.1:5000/>

## Learning Goals

This project was developed as part of my journey to learn and demonstrate machine learning engineering skills.
The primary learning goal was to **implement regression models from scratch** and integrate them into a full web
application.

**Note:** This README is written to be clear and informative for any new visitor (or recruiter) checking out the
project. It highlights the project's purpose, capabilities, technical implementation, and the learning outcomes
associated with it.