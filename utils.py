import base64
from io import BytesIO
import numpy as np
import matplotlib.pyplot as plt


def to_b64(fig_buf):
    """
    Convert a figure buffer to base64 string for embedding in HTML.
    
    Args:
        fig_buf: BytesIO buffer containing the figure
        
    Returns:
        base64 encoded string
    """
    fig_buf.seek(0)
    return base64.b64encode(fig_buf.read()).decode("utf-8")


def plot_pred(y_true, y_pred):
    """
    Create a scatter plot comparing true vs predicted values.
    
    Args:
        y_true: True target values
        y_pred: Predicted target values
        
    Returns:
        base64 encoded string of the plot
    """
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
    """
    Plot the cost function history during training.
    
    Args:
        history: List of cost values at each iteration
        
    Returns:
        base64 encoded string of the plot
    """
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
    """
    Plot the evolution of parameters during gradient descent.
    
    Args:
        theta_history: List of parameter arrays at each iteration
        
    Returns:
        base64 encoded string of the plot
    """
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


def safe_expm1(x, min_val=-709, max_val=709):
    """
    Safely apply expm1 to avoid overflow/underflow.
    
    Args:
        x: Input array or value
        min_val: Minimum allowed value
        max_val: Maximum allowed value
        
    Returns:
        expm1 result with clipping
    """
    # Clip values to prevent overflow in expm1
    x_clipped = np.clip(x, min_val, max_val)
    return np.expm1(x_clipped)


def safe_log1p(x, min_val=1e-15):
    """
    Safely apply log1p to avoid issues with zero or negative values.
    
    Args:
        x: Input array or value
        min_val: Minimum allowed value
        
    Returns:
        log1p result with clipping
    """
    # Ensure x is non-negative
    x_clipped = np.maximum(x, min_val)
    return np.log1p(x_clipped)
