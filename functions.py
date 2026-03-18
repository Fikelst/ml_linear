import numpy as np

def predict(X, theta):
    return np.dot(X, theta)

def compute_cost(X, y, theta):
    m = len(y)
    predictions = predict(X, theta)
    cost = (1 / (2 * m)) * np.sum(np.square(predictions - y))
    return cost

def gradient_descent(X, y, theta, alpha, num_iters):
    m = X.shape[0]
    J_history = []
    X_T = np.transpose(X)
    for i in range(num_iters):
        predictions = np.dot(X, theta)
        error = predictions - y
        gradient = (1 / m) * np.dot(X_T, error)
        theta = theta - alpha * gradient
        cost = (1 / (2 * m)) * np.sum(np.square(error))
        J_history.append(cost)
    return theta, J_history

def calculate_r2(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    return 1 - (ss_res / ss_tot)