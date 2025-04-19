import datetime
from io import BytesIO
import base64
import math
from flask import Flask, render_template, request, flash
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures

app = Flask(__name__)
app.secret_key = "secret_key"


##########################
# 1. DATA LOADING        #
##########################

def load_data(csv_path: str = "static/car_details_from_car_dehkho.csv") -> pd.DataFrame:
    try:
        return pd.read_csv(csv_path)
    except Exception as exc:
        raise RuntimeError(f"Error loading data: {exc}") from exc


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Adds **car_age**, **km_per_year**, and **brand**."""
    out = df.copy()
    this_year = datetime.datetime.now().year
    out["car_age"] = this_year - out["year"]
    out.loc[out["car_age"] == 0, "car_age"] = 1  # avoid 0‑div for new cars
    out["km_per_year"] = out["km_driven"] / out["car_age"]
    out["brand"] = out["name"].str.split().str[0]
    return out


##########################
# 2. FEATURE UTILITIES   #
##########################

CATEGORICAL_RAW = ["fuel", "seller_type", "transmission", "owner", "brand"]


def get_available_features(df: pd.DataFrame) -> list[str]:
    df = add_engineered_features(df)
    df_enc = pd.get_dummies(df, columns=CATEGORICAL_RAW, drop_first=True)
    return sorted(df_enc.drop(columns=["selling_price", "name"]).columns)


##########################
# 3. PRE‑PROCESSING      #
##########################

def preprocess_data(df: pd.DataFrame, selected: list[str]):
    df = add_engineered_features(df)
    df_enc = pd.get_dummies(df, columns=CATEGORICAL_RAW, drop_first=True)

    X_all = df_enc.drop(columns=["selling_price", "name"])
    y_log = np.log1p(df_enc["selling_price"]).astype(np.float64)

    if selected:
        X_all = X_all[[c for c in selected if c in X_all.columns]]
    return X_all, y_log


##########################
# 4. SCALING             #
##########################

def standard_scale(df_train: pd.DataFrame, df_test: pd.DataFrame):
    """Z‑score numeric columns; dummies unchanged."""
    tr, te = df_train.copy(), df_test.copy()
    num_cols = tr.select_dtypes(include=[np.number]).columns
    mean = tr[num_cols].mean()
    std = tr[num_cols].std().replace(0, 1)
    tr[num_cols] = (tr[num_cols] - mean) / std
    te[num_cols] = (te[num_cols] - mean) / std
    return tr, te, num_cols


##########################
# 5. TRAIN/TEST SPLIT    #
##########################

def train_test_split(X, y, test_size=0.2, seed=42):
    np.random.seed(seed)
    idx = np.random.permutation(len(X))
    cut = int(len(X) * test_size)
    te, tr = idx[:cut], idx[cut:]
    return X.iloc[tr], X.iloc[te], y.iloc[tr], y.iloc[te]


##########################
# 6. DESIGN MATRIX       #
##########################

def design_matrix_builder(X_df: pd.DataFrame, numeric_cols: list[str], degree: int):
    poly = PolynomialFeatures(degree=degree, include_bias=False) if degree > 1 else None

    def transform(df: pd.DataFrame):
        num = df[numeric_cols].to_numpy(dtype=np.float64)
        num_poly = poly.fit_transform(num) if poly else num
        dummies = df.drop(columns=numeric_cols).to_numpy(dtype=np.float64)
        return np.hstack([np.ones((len(df), 1)), num_poly, dummies])

    return transform


##########################
# 7. LR WITH GD          #
##########################

def compute_cost(X, y, theta, lam=0.0):
    diff = X @ theta - y
    mse = np.dot(diff, diff) / (2 * len(y))
    if lam:
        mse += (lam / (2 * len(y))) * np.sum(theta[1:] ** 2)
    return mse


def gradient_descent(X, y, theta, alpha, iters, lam=0.0, tol=1e-8, clip=1e3):
    m = len(y)
    cost_hist, theta_hist = [], []
    for _ in range(iters):
        pred = X @ theta
        grad = (X.T @ (pred - y)) / m
        if lam:
            reg = (lam / m) * theta
            reg[0] = 0
            grad += reg
        # gradient clipping
        g_norm = np.linalg.norm(grad)
        if g_norm > clip:
            grad *= clip / g_norm
        theta -= alpha * grad
        cost = compute_cost(X, y, theta, lam)
        cost_hist.append(cost)
        theta_hist.append(theta.copy())
        if len(cost_hist) > 1 and abs(cost_hist[-2] - cost) < tol:
            break
    return theta, cost_hist, theta_hist


def predict(X, theta):
    return X @ theta


##########################
# 8. K‑FOLD CV           #
##########################

def k_fold_cv(X, y, k, alpha, iters, lam):
    """Return average **MSE in rupees** across k folds."""
    m = len(y)
    fold = m // k
    idx = np.random.permutation(m)
    scores = []
    for i in range(k):
        te = idx[i * fold:(i + 1) * fold] if i < k - 1 else idx[i * fold:]
        tr = np.setdiff1d(idx, te)
        Xtr, Xte, ytr, yte = X[tr], X[te], y[tr], y[te]
        theta = np.zeros(X.shape[1])
        theta, _, _ = gradient_descent(Xtr, ytr, theta, alpha, iters, lam)
        # back‑transform to rupees
        y_pred = np.expm1(predict(Xte, theta))
        y_true = np.expm1(yte)
        scores.append(np.mean((y_pred - y_true) ** 2))
    return float(np.mean(scores))


##########################
# 9. PLOTS & METRICS     #
##########################

def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return 0.0 if ss_tot == 0 else 1 - ss_res / ss_tot


def to_b64(fig_buf):
    fig_buf.seek(0)
    return base64.b64encode(fig_buf.read()).decode("utf-8")


def plot_pred(y_true, y_pred):
    buf = BytesIO()
    plt.figure(figsize=(6, 4))
    plt.scatter(y_true, y_pred, alpha=.5)
    lim = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    plt.plot(lim, lim, 'r--')
    plt.tight_layout()
    plt.savefig(buf, format='png')
    plt.close()
    return to_b64(buf)


def plot_cost(history):
    buf = BytesIO()
    plt.figure(figsize=(6, 4))
    plt.plot(history)
    plt.xlabel('Iterations')
    plt.ylabel('Cost')
    plt.title('Cost Function History')
    plt.tight_layout()
    plt.savefig(buf, format='png')
    plt.close()
    return to_b64(buf)


def plot_gradient_descent(theta_history):
    """Plots the evolution of parameters during gradient descent."""
    buf = BytesIO()
    plt.figure(figsize=(8, 5))
    
    # Limit to first 10 parameters for clarity
    num_params = min(10, len(theta_history[0]))
    for i in range(num_params):
        param_values = [theta[i] for theta in theta_history]
        plt.plot(param_values, label=f'θ{i}')
    
    plt.xlabel('Iterations')
    plt.ylabel('Parameter Value')
    plt.title('Parameter Evolution During Gradient Descent')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(buf, format='png')
    plt.close()
    return to_b64(buf)


##########################
# 10. GLOBALS            #
##########################
DF = load_data()
AVAILABLE = get_available_features(DF)


##########################
# 11. ROUTES             #
##########################

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            algo = request.form.get('algorithm_type', 'linear')
            deg = int(request.form.get('poly_degree', 1))
            deg = max(1, min(deg, 4))
            use_reg = request.form.get('use_regularization') == 'on'
            lam = float(request.form.get('lambda', 0.1)) if use_reg else 0.0
            use_cv = request.form.get('use_cv') == 'on'
            sel_feats = request.form.getlist('features') or AVAILABLE
            alpha = float(request.form.get('alpha', 0.001))
            iters = int(request.form.get('num_iters', 2000))
            scale_method = request.form.get('scaling_method', 'standardization')

            X_df, y_log = preprocess_data(DF, sel_feats)
            Xtr_df, Xte_df, ytr, yte = train_test_split(X_df, y_log)
            Xtr_df, Xte_df, num_cols = standard_scale(Xtr_df, Xte_df)

            deg_used = deg if algo == 'polynomial' else 1
            transform = design_matrix_builder(Xtr_df, num_cols, deg_used)
            Xtr, Xte = transform(Xtr_df), transform(Xte_df)

            theta = np.zeros(Xtr.shape[1])
            cv_mse = None
            if use_cv:
                cv_mse = k_fold_cv(Xtr, ytr.to_numpy(), k=5, alpha=alpha, iters=iters, lam=lam)

            theta, cost_hist, theta_hist = gradient_descent(Xtr, ytr.to_numpy(), theta, alpha, iters, lam)

            y_pred_log = predict(Xte, theta)
            y_pred, y_true = np.expm1(y_pred_log), np.expm1(yte)

            mse = np.mean((y_pred - y_true) ** 2)
            r2 = r2_score(y_true, y_pred)

            return render_template('results.html',
                                   metrics={
                                       'mse': f"{mse:,.0f}", 'r2': f"{r2:.3f}",
                                       'poly_degree': deg_used, 'algorithm_type': algo,
                                       'regularization': use_reg, 'lambda': lam if use_reg else None,
                                       'cross_validation': use_cv, 'cv_mse': f"{cv_mse:,.0f}" if cv_mse else None,
                                       'alpha': alpha, 'num_iters': len(cost_hist),
                                       'features_used': sel_feats,
                                       'scaling_method': scale_method
                                   },
                                   plot_url=plot_pred(y_true, y_pred),
                                   cost_plot_url=plot_cost(cost_hist),
                                   gd_plot_url=plot_gradient_descent(theta_hist))
        except Exception as exc:
            flash(f"Error: {exc}")
    return render_template('index.html', available_features=AVAILABLE)


if __name__ == '__main__':
    print('Features available:', AVAILABLE)
    app.run(debug=True)
