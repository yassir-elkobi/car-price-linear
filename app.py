from flask import Flask, render_template, request, flash
import numpy as np
import pandas as pd
from preprocessing import (
    load_data, add_engineered_features, get_available_features,
    preprocess_data, standard_scale, train_test_split, design_matrix_builder
)
from model import compute_cost, gradient_descent, predict, k_fold_cv, r2_score
from utils import plot_pred, plot_cost, plot_gradient_descent, safe_expm1

app = Flask(__name__)

DF = load_data()
AVAILABLE = get_available_features(DF)


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
                cv_mse = k_fold_cv(Xtr, ytr.to_numpy(), k=5, alpha=alpha, num_iters=iters, lambda_=lam)

            theta, cost_hist, theta_hist = gradient_descent(Xtr, ytr.to_numpy(), theta, alpha, iters, lam)

            y_pred_log = predict(Xte, theta)
            # Clip values before applying expm1 to prevent overflow
            y_pred_log = np.clip(y_pred_log, -709, 709)  # log(1e308) ≈ 709, clipping to avoid overflow
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
