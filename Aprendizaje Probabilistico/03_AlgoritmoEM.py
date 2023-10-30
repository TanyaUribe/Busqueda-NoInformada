"""
El algoritmo EM (Expectation-Maximization) es un método de optimización utilizado en estadísticas
y aprendizaje automático para estimar los parámetros de modelos probabilísticos cuando se tienen
datos incompletos o se desconoce la distribución de los datos observados. 
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Generar datos de ejemplo de una mezcla de dos distribuciones gaussianas
np.random.seed(0)
n = 1000
mu_true = [2.0, 7.0]
sigma_true = [1.0, 1.5]
weights_true = [0.4, 0.6]
component = np.random.choice(2, n, p=weights_true)
data = np.random.normal(mu_true[component], sigma_true[component])

# Inicialización de los parámetros
mu_1, sigma_1, weight_1 = 0.0, 1.0, 0.5
mu_2, sigma_2, weight_2 = 5.0, 1.0, 0.5

# Función para calcular la responsabilidad (E-step)
def calculate_responsibility(data, mu_1, sigma_1, weight_1, mu_2, sigma_2, weight_2):
    likelihood_1 = norm.pdf(data, loc=mu_1, scale=sigma_1)
    likelihood_2 = norm.pdf(data, loc=mu_2, scale=sigma_2)
    total_likelihood = weight_1 * likelihood_1 + weight_2 * likelihood_2
    responsibility_1 = weight_1 * likelihood_1 / total_likelihood
    responsibility_2 = weight_2 * likelihood_2 / total_likelihood
    return responsibility_1, responsibility_2

# Algoritmo EM
num_iterations = 50
for i in range(num_iterations):
    # E-step: Calcular las responsabilidades
    responsibility_1, responsibility_2 = calculate_responsibility(data, mu_1, sigma_1, weight_1, mu_2, sigma_2, weight_2)

    # M-step: Actualizar los parámetros
    mu_1 = np.sum(responsibility_1 * data) / np.sum(responsibility_1)
    sigma_1 = np.sqrt(np.sum(responsibility_1 * (data - mu_1)**2) / np.sum(responsibility_1))
    weight_1 = np.mean(responsibility_1)

    mu_2 = np.sum(responsibility_2 * data) / np.sum(responsibility_2)
    sigma_2 = np.sqrt(np.sum(responsibility_2 * (data - mu_2)**2) / np.sum(responsibility_2))
    weight_2 = np.mean(responsibility_2)

# Visualizar los resultados
x = np.linspace(-5, 15, 1000)
density_1 = norm.pdf(x, loc=mu_1, scale=sigma_1)
density_2 = norm.pdf(x, loc=mu_2, scale=sigma_2)
final_mixture = weight_1 * density_1 + weight_2 * density_2

plt.figure(figsize=(12, 6))
plt.hist(data, bins=30, density=True, alpha=0.5, label='Datos de Muestra')
plt.plot(x, final_mixture, label='Distribución Estimada (EM)')
plt.legend()
plt.title('Estimación de una Mezcla de Distribuciones Gaussianas con EM')
plt.show()
