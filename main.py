import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from functions import predict, compute_cost, gradient_descent

data = pd.read_csv('insurance.csv')

with open('config.json', 'r') as f:
    params = json.load(f)

alpha = params['alpha']
num_iters = params['num_iters']

data['sex'] = data['sex'].map({'male': 1, 'female': 0})
data['smoker'] = data['smoker'].map({'yes': 1, 'no': 0})

X = data[['age', 'sex', 'bmi', 'children', 'smoker']].copy()
y = data['charges'].copy()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

mean_vals = X_train.mean()
std_vals = X_train.std()

X_train_norm = (X_train - mean_vals) / std_vals
X_test_norm = (X_test - mean_vals) / std_vals

X_train_final = np.c_[np.ones(X_train_norm.shape[0]), X_train_norm.values]
X_test_final = np.c_[np.ones(X_test_norm.shape[0]), X_test_norm.values]
y_train_final = y_train.values.reshape(-1, 1)
y_test_final = y_test.values.reshape(-1, 1)

theta = np.zeros((X_train_final.shape[1], 1))
initial_cost = compute_cost(X_train_final, y_train_final, theta)

theta_final, cost_history = gradient_descent(
    X_train_final, y_train_final, theta, alpha, num_iters
)

predictions_test = predict(X_test_final, theta_final)
final_test_cost = compute_cost(X_test_final, y_test_final, theta_final)

ss_res = np.sum((y_test_final - predictions_test)**2)
ss_tot = np.sum((y_test_final - np.mean(y_test_final))**2)
r2 = 1 - (ss_res / ss_tot)

# --- WYŚWIETLANIE WYNIKÓW ---
print(f"Początkowy błąd: {initial_cost:.2f}")
print(f"Końcowy błąd na testach: {final_test_cost:.2f}")
print(f"Współczynnik R2: {r2:.4f}")

# Wizualizacja: Rzeczywistość vs Predykcja
plt.scatter(y_test_final, predictions_test, color='blue', alpha=0.5, label='Dane testowe')
max_val = float(max(y_test_final.max(), predictions_test.max()))
plt.plot([0, max_val], [0, max_val], color="black", linewidth=2, label='Idealne dopasowanie')
plt.xlabel('Rzeczywiste koszty')
plt.ylabel('Przewidziane koszty')
plt.title('Skuteczność modelu regresji')
plt.legend()
plt.show()