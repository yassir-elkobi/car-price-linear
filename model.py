import numpy as np


def compute_cost(X, y, theta, lambda_=0.0):
    """
    Compute the cost for linear regression with optional L2 regularization.

    Args:
        X (np.ndarray): Design matrix, shape (m, n).
        y (np.ndarray): Target values, shape (m,).
        theta (np.ndarray): Parameters, shape (n,).
        lambda_ (float): Regularization strength (default 0: no regularization).

    Returns:
        float: The computed cost.
    """
    m = len(y)
    predictions = X.dot(theta)
    errors = predictions - y
    # Clip errors to prevent numerical overflow
    errors = np.clip(errors, -1e15, 1e15)
    squared_errors = errors ** 2
    cost = (1 / (2 * m)) * np.sum(squared_errors)

    if lambda_ > 0:
        # Clip theta to prevent overflow in regularization term
        theta_reg = np.clip(theta[1:], -1e15, 1e15)
        reg_term = (lambda_ / (2 * m)) * np.sum(theta_reg ** 2)
        cost += reg_term

    # Check for numerical issues
    if np.isnan(cost) or np.isinf(cost):
        print("Warning: Cost function resulted in NaN or Inf.")

    return cost


def gradient_descent(X, y, theta, alpha, num_iters, lambda_=0.0, tol=1e-6):
    """
    Perform gradient descent to learn theta parameters.

    Args:
        X (np.ndarray): Design matrix, shape (m, n).
        y (np.ndarray): Target values, shape (m,).
        theta (np.ndarray): Initial parameters, shape (n,).
        alpha (float): Learning rate.
        num_iters (int): Maximum number of iterations.
        lambda_ (float): Regularization strength.
        tol (float): Tolerance for early stopping based on cost improvement.

    Returns:
        theta (np.ndarray): Learned parameters.
        cost_history (list): Cost at each iteration.
        theta_history (list): Parameter values at each iteration.
    """
    m = len(y)
    cost_history = []
    theta_history = [theta.copy()]  # Save initial theta

    prev_cost = float('inf')

    for i in range(num_iters):
        # Compute predictions and error
        predictions = X.dot(theta)
        errors = predictions - y

        # Clip extremely large errors to prevent overflow
        errors = np.clip(errors, -1e15, 1e15)

        # Compute gradient
        gradient = (1 / m) * (X.T.dot(errors))
        if lambda_ > 0:
            # Regularize all but the bias term
            gradient[1:] += (lambda_ / m) * theta[1:]

        # Clip extremely large gradients to prevent overflow
        gradient = np.clip(gradient, -1e15, 1e15)

        # Update parameters
        update = alpha * gradient
        new_theta = theta - update

        # Check for NaN or inf values in theta
        if np.any(np.isnan(new_theta)) or np.any(np.isinf(new_theta)):
            print("Warning: NaN or Inf values detected in parameters. Stopping early.")
            break

        theta = new_theta
        theta_history.append(theta.copy())

        # Save cost and check for convergence
        try:
            cost = compute_cost(X, y, theta, lambda_)
            # Check for NaN cost
            if np.isnan(cost) or np.isinf(cost):
                print("Warning: NaN or Inf cost detected. Stopping early.")
                break

            cost_history.append(cost)

            # Check for convergence using valid values only
            if i > 0 and prev_cost != float('inf') and abs(prev_cost - cost) < tol:
                break

            prev_cost = cost

        except (RuntimeWarning, OverflowError) as e:
            print(f"Warning: {e}. Stopping early.")
            break

    return theta, cost_history, theta_history


def predict(X, theta):
    """
    Generate predictions given inputs X and parameters theta.

    Args:
        X (np.ndarray): Design matrix, shape (m, n).
        theta (np.ndarray): Parameters, shape (n,).

    Returns:
        np.ndarray: Predicted values, shape (m,).
    """
    return X.dot(theta)


def k_fold_cv(X, y, k=5, alpha=0.01, num_iters=1000, lambda_=0.0):
    """
    Perform k-fold cross-validation using gradient descent.

    Args:
        X (np.ndarray): Full design matrix, shape (m, n).
        y (np.ndarray): Full target array (log-transformed), shape (m,).
        k (int): Number of folds.
        alpha (float): Learning rate for gradient descent.
        num_iters (int): Maximum iterations for gradient descent.
        lambda_ (float): Regularization strength.

    Returns:
        float: Average MSE (in original scale) across folds.
    """
    m = len(y)
    indices = np.arange(m)
    np.random.shuffle(indices)
    fold_size = m // k
    mse_scores = []

    for fold in range(k):
        start = fold * fold_size
        end = start + fold_size if fold < k - 1 else m
        test_idx = indices[start:end]
        train_idx = np.hstack((indices[:start], indices[end:]))

        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        # Initialize theta for this fold
        theta_init = np.zeros(X.shape[1])

        # Train model
        theta_trained, _, _ = gradient_descent(
            X_train, y_train, theta_init, alpha, num_iters, lambda_)

        # Make predictions and back-transform
        y_pred = predict(X_test, theta_trained)

        # Handle potential NaNs or Infs
        if np.any(np.isnan(y_pred)) or np.any(np.isinf(y_pred)):
            print(f"Warning: NaN or Inf in predictions for fold {fold}. Skipping this fold.")
            continue

        # Safely back-transform
        try:
            # Clip y_pred to prevent overflow in expm1
            y_pred = np.clip(y_pred, -709, 709)  # log(1e308) ≈ 709, clipping to avoid overflow
            y_pred = np.expm1(y_pred)
            y_true = np.expm1(y_test)

            # Compute fold MSE with clipping to prevent extreme values
            y_pred = np.clip(y_pred, 0, 1e15)  # Clip to reasonable values
            mse = np.mean((y_pred - y_true) ** 2)

            if not np.isnan(mse) and not np.isinf(mse):
                mse_scores.append(mse)
        except (RuntimeWarning, OverflowError) as e:
            print(f"Warning in fold {fold}: {e}")
            continue

    # Return average if we have scores, otherwise a high value to indicate failure
    if len(mse_scores) > 0:
        return float(np.mean(mse_scores))
    else:
        print("Warning: All folds failed. Returning high MSE.")
        return float(1e10)  # Return a high MSE to indicate failure


def r2_score(y_true, y_pred):
    """
    Calculate the coefficient of determination (R²) for predictions.
    
    Args:
        y_true: Ground truth target values
        y_pred: Predicted target values
        
    Returns:
        R² score (float)
    """
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return 0.0 if ss_tot == 0 else 1 - ss_res / ss_tot
