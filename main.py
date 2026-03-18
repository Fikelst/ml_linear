import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from functions import predict, compute_cost, gradient_descent, calculate_r2
from wizualizacja import plot_regression_results

# Ładowanie danych i parametrów
data = pd.read_csv('insurance.csv')
with open('config.json', 'r') as f:
    params = json.load(f)

# Preprocessing
data['sex'] = data['sex'].map({'male': 1, 'female': 0})
data['smoker'] = data['smoker'].map({'yes': 1, 'no': 0})
X = data[['age', 'sex', 'bmi', 'children', 'smoker']].copy()
y = data['charges'].copy()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalizacja
mean_vals = X_train.mean()
std_vals = X_train.std()
X_train_norm = (X_train - mean_vals) / std_vals
X_test_norm = (X_test - mean_vals) / std_vals

X_train_final = np.c_[np.ones(X_train_norm.shape[0]), X_train_norm.values]
X_test_final = np.c_[np.ones(X_test_norm.shape[0]), X_test_norm.values]
y_train_final = y_train.values.reshape(-1, 1)
y_test_final = y_test.values.reshape(-1, 1)

# Model
theta = np.zeros((X_train_final.shape[1], 1))
theta_final, cost_history = gradient_descent(X_train_final, y_train_final, theta, params['alpha'], params['num_iters'])

# Wyniki
predictions_test = predict(X_test_final, theta_final)
r2_score = calculate_r2(y_test_final, predictions_test)

print(f"Współczynnik R2: {r2_score:.4f}")
plot_regression_results(y_test_final, predictions_test)