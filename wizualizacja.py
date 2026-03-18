import matplotlib.pyplot as plt
from main import *
plt.scatter(y_test_final, predictions_test, color='blue', alpha=0.5, label='Dane testowe')
max_val = float(max(y_test_final.max(), predictions_test.max()))
plt.plot([0, max_val], [0, max_val], color="black", linewidth=2, label='Idealne dopasowanie')
plt.xlabel('Rzeczywiste koszty')
plt.ylabel('Przewidziane koszty')
plt.title('Skuteczność modelu regresji')
plt.legend()
plt.show()