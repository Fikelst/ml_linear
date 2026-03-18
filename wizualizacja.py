import matplotlib.pyplot as plt

def plot_regression_results(y_test, predictions):
    plt.scatter(y_test, predictions, color='blue', alpha=0.5, label='Dane testowe')
    max_val = float(max(y_test.max(), predictions.max()))
    plt.plot([0, max_val], [0, max_val], color="black", linewidth=2, label='Idealne dopasowanie')
    plt.xlabel('Rzeczywiste koszty')
    plt.ylabel('Przewidziane koszty')
    plt.title('Skuteczność modelu regresji')
    plt.legend()
    plt.show()

def plot_cost_history(cost_history):
    plt.plot(cost_history)
    plt.xlabel('Iteracja')
    plt.ylabel('Koszt (J)')
    plt.title('Historia funkcji kosztu')
    plt.show()